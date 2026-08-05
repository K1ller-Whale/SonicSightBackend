import os
import librosa
import numpy as np
import torch
import soundfile as sf
import cv2
from PIL import Image
import torch.nn.functional as F
import logging
from contextlib import nullcontext

from models import ModelBuilder, activate
from config import (
    ARCH_SOUND,
    ARCH_FRAME,
    ARCH_SYNTH,
    NUM_CHANNELS,
    NUM_MIX,
    NUM_FRAMES,
    STRIDE_FRAMES,
    FRAME_RATE,
    IMG_SIZE,
    AUD_LEN,
    AUD_RATE,
    STFT_FRAME,
    STFT_HOP,
    LOG_FREQ,
    BINARY_MASK,
    MASK_THRES,
    MASK_RENORM,
    MASK_RENORM_FLOOR,
    MASK_RENORM_EPS,
    AUDIO_NORM,
    AUDIO_NORM_TARGET_RMS,
    AUDIO_NORM_MAX_GAIN,
    AUDIO_NORM_MIN_RMS,
    AUDIO_NORM_SMOOTH,
    STREAM_DUMP,
    IMG_POOL,
    SEPARATION_MODE,
    SEP_TOPK,
    MASK_SMOOTH,
    CKPT_ROOT,
)
import utils.video_transforms as vtransforms
from utils.utils import warpgrid, istft_reconstruction
from utils.debugging_clock import DebuggingClock

logger = logging.getLogger(__name__)


class StreamingBuffer:
    """Manages a rolling buffer of frames and audio for near real-time inference."""

    def __init__(
        self,
        target_audio_len=AUD_LEN,
        target_sr=AUD_RATE,
        num_frames=NUM_FRAMES,
        frame_ring_cap=60,
        early_min_samples=None,
        frame_selection="centered_triple",
        window_min_advance=1,
    ):
        self.target_audio_len = target_audio_len
        self.target_sr = target_sr

        # Per-model window configuration (instance state, NOT class state, so
        # two concurrently open streams with different models cannot collide).
        self.num_frames = int(num_frames)
        self.frame_ring_cap = int(frame_ring_cap)
        # Minimum audio before early inference is attempted. Below ~1 second
        # the spectrogram is too empty for meaningful separation. Setting it
        # equal to target_audio_len disables early mode entirely.
        self.early_min_samples = (
            int(early_min_samples) if early_min_samples is not None else int(self.target_sr)
        )
        # "centered_triple": [center-1, center, center+1] left/right halves
        #                    (Sound of Pixels), returned as (left[3], right[3]).
        # "consecutive_span": num_frames consecutive full frames around the
        #                     center, returned as a flat list of PIL images.
        self.frame_selection = frame_selection
        # Minimum sample advance between consecutive windows (the model's hop
        # for high-frame-rate models; 1 = pace by frame arrival). Enforced
        # only after the first window so startup latency is unaffected.
        self.window_min_advance = int(window_min_advance)

        # Audio buffer (PCM 16-bit float)
        self.audio_chunks = []
        self.total_audio_samples = 0
        self.audio_base_sample = 0

        # Frame buffer (list of dicts with 'timestamp_ms' and 'image' (PIL))
        self.frame_buffer = []

        # Track absolute timestamps
        self.audio_start_ms = None
        self.last_processed_timestamp_ms = -1
        self.last_audio_timestamp_ms = -1
        self.last_frame_timestamp_ms = -1
        self.last_processed_window_start_sample = -1

    def add_audio_chunk(self, pcm_bytes, timestamp_ms):
        """Adds raw PCM 16-bit 11025Hz bytes to the audio buffer."""
        timestamp_ms = int(timestamp_ms)
        chunk_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        chunk_float32 = chunk_int16.astype(np.float32) / 32768.0

        if self.audio_start_ms is None:
            chunk_duration_ms = (len(chunk_float32) / self.target_sr) * 1000.0
            self.audio_start_ms = timestamp_ms - chunk_duration_ms
        elif (
            self.last_audio_timestamp_ms != -1
            and timestamp_ms < self.last_audio_timestamp_ms
        ):
            logger.warning(
                "Out-of-order audio timestamp detected: current=%sms last=%sms",
                timestamp_ms,
                self.last_audio_timestamp_ms,
            )

        self.audio_chunks.append(chunk_float32)
        self.total_audio_samples += len(chunk_float32)
        self.last_audio_timestamp_ms = max(self.last_audio_timestamp_ms, timestamp_ms)

        # Keep a max buffer of ~15 seconds to avoid memory bloat
        max_samples = self.target_sr * 15
        if self.total_audio_samples > max_samples:
            flat_audio = np.concatenate(self.audio_chunks)
            discard_samples = len(flat_audio) - max_samples
            self.audio_start_ms += (discard_samples / self.target_sr) * 1000.0
            self.audio_base_sample += discard_samples
            self.audio_chunks = [flat_audio[-max_samples:]]
            self.total_audio_samples = max_samples

    def decode_images(self, timestamp_ms, left_jpeg, right_jpeg):
        import io

        left_img = Image.open(io.BytesIO(left_jpeg)).convert("RGB")
        right_img = Image.open(io.BytesIO(right_jpeg)).convert("RGB")
        return timestamp_ms, left_img, right_img

    def decode_full_image(self, timestamp_ms, full_jpeg):
        import io

        full_img = Image.open(io.BytesIO(full_jpeg)).convert("RGB")
        return timestamp_ms, full_img

    def add_decoded_frame(self, timestamp_ms, left_img, right_img, full_img=None):
        timestamp_ms = int(timestamp_ms)

        if timestamp_ms <= self.last_processed_timestamp_ms:
            logger.warning(
                "Dropping stale frame older than last processed window: frame=%sms processed=%sms",
                timestamp_ms,
                self.last_processed_timestamp_ms,
            )
            return

        if self.last_frame_timestamp_ms != -1 and timestamp_ms <= self.last_frame_timestamp_ms:
            logger.warning(
                "Dropping out-of-order frame timestamp: current=%sms last=%sms",
                timestamp_ms,
                self.last_frame_timestamp_ms,
            )
            return

        self.frame_buffer.append(
            {
                "timestamp_ms": timestamp_ms,
                "left_img": left_img,
                "right_img": right_img,
                "full_img": full_img,
            }
        )
        self.last_frame_timestamp_ms = timestamp_ms

        # Ring cap to bound memory (60 for Sound of Pixels; larger for models
        # that consume long consecutive frame spans)
        if len(self.frame_buffer) > self.frame_ring_cap:
            self.frame_buffer.pop(0)

    def add_frame(self, timestamp_ms, left_jpeg, right_jpeg, width, height):
        """Decodes JPEG bytes and adds the PIL Image to the frame buffer."""
        try:
            timestamp_ms, left_img, right_img = self.decode_images(
                timestamp_ms, left_jpeg, right_jpeg
            )
            self.add_decoded_frame(timestamp_ms, left_img, right_img)
        except Exception as e:
            logger.info(f"Error decoding frame: {e}")

    def has_enough_data(self):
        """Checks if we have basic minimum requirements to even attempt finding a window."""
        if self.total_audio_samples < self.early_min_samples:
            return False
        if len(self.frame_buffer) < self.num_frames:
            return False
        return True

    def _select_frames(self, center_idx):
        """Build the model's frame input around center_idx.

        Returns (frames, trim_start): the frames object handed to the engine
        and the buffer index older frames can be trimmed from after a
        successful window.
        """
        if self.frame_selection == "consecutive_span":
            half = self.num_frames // 2
            start = center_idx - half
            start = max(0, min(start, len(self.frame_buffer) - self.num_frames))
            sel = self.frame_buffer[start : start + self.num_frames]
            return [entry["full_img"] for entry in sel], start

        if self.frame_selection == "centered_triple_full":
            # Pixel mode: same 3-consecutive-frame selection as the deployed
            # streaming halves path (NOT the offline STRIDE_FRAMES spread),
            # but returning the full letterboxed frames.
            full_frames = [
                self.frame_buffer[center_idx - 1]["full_img"],
                self.frame_buffer[center_idx]["full_img"],
                self.frame_buffer[center_idx + 1]["full_img"],
            ]
            return full_frames, center_idx - 1

        # "centered_triple" — the original Sound of Pixels selection, verbatim.
        left_frames = [
            self.frame_buffer[center_idx - 1]["left_img"],
            self.frame_buffer[center_idx]["left_img"],
            self.frame_buffer[center_idx + 1]["left_img"],
        ]
        right_frames = [
            self.frame_buffer[center_idx - 1]["right_img"],
            self.frame_buffer[center_idx]["right_img"],
            self.frame_buffer[center_idx + 1]["right_img"],
        ]
        return (left_frames, right_frames), center_idx - 1

    def get_latest_window(self):
        """
        Returns the most recent valid window for inference:
        - target_audio_len samples of audio (zero-padded if in early mode)
        - num_frames frames centered precisely in the audio stream
          (selection shape per self.frame_selection)

        Sets self._last_window_mode to:
          'early'  — audio was zero-padded (< target_audio_len available)
          'normal' — full audio window available

        In early mode, also sets self._last_early_audio_len to the number of
        real (non-zero-padded) samples, so the caller can adjust the OLA
        center-slice extraction point.
        """
        if not self.has_enough_data():
            return None, None, None, None

        available_samples = self.total_audio_samples
        is_early = available_samples < self.target_audio_len

        if is_early:
            # ── EARLY INFERENCE MODE ──
            # Use ALL available audio, zero-pad to target_audio_len.
            # Place real audio at the START of the window.
            if len(self.audio_chunks) > 1:
                flat_audio = np.concatenate(self.audio_chunks)
                self.audio_chunks = [flat_audio]
            else:
                flat_audio = self.audio_chunks[0]

            real_audio = flat_audio[:available_samples].copy()
            audio_window = np.zeros(self.target_audio_len, dtype=np.float32)
            audio_window[:len(real_audio)] = real_audio

            # Use the latest frames as visual context
            # (we guaranteed >= num_frames in has_enough_data)
            valid_center_idx = len(self.frame_buffer) - 2
            if valid_center_idx < 1:
                valid_center_idx = 1
            if valid_center_idx >= len(self.frame_buffer) - 1:
                valid_center_idx = len(self.frame_buffer) - 2

            frames_tuple, _ = self._select_frames(valid_center_idx)

            center_timestamp = self.frame_buffer[valid_center_idx]["timestamp_ms"]

            # Use audio_base_sample as the absolute start
            absolute_window_start_sample = self.audio_base_sample

            # Guard against non-monotonic windows
            if (
                self.last_processed_window_start_sample >= 0
                and absolute_window_start_sample
                <= self.last_processed_window_start_sample
            ):
                return None, None, None, None

            self.last_processed_timestamp_ms = center_timestamp
            self.last_processed_window_start_sample = absolute_window_start_sample

            # Communicate early-mode metadata to the caller
            self._last_window_mode = "early"
            self._last_early_audio_len = len(real_audio)

            logger.info(
                "Early inference: %d/%d samples (%.1f%% real), ts=%dms",
                len(real_audio),
                self.target_audio_len,
                100.0 * len(real_audio) / self.target_audio_len,
                center_timestamp,
            )

            return (
                audio_window,
                frames_tuple,
                int(center_timestamp),
                absolute_window_start_sample,
            )

        # ── NORMAL MODE ── (existing logic)
        self._last_window_mode = "normal"
        self._last_early_audio_len = 0

        half_audio_ms = ((self.target_audio_len / 2) / self.target_sr) * 1000.0
        current_audio_end_ms = (
            self.audio_start_ms + (self.total_audio_samples / self.target_sr) * 1000.0
        )

        valid_center_idx = -1
        audio_start_idxFor_slice = -1
        audio_end_idxFor_slice = -1
        absolute_window_start_sample = -1

        # Search backward from the end for the first frame that has valid audio surrounding it
        for i in range(len(self.frame_buffer) - 2, 0, -1):
            t_frame = self.frame_buffer[i]["timestamp_ms"]

            if t_frame <= self.last_processed_timestamp_ms:
                continue

            req_start_time = t_frame - half_audio_ms
            req_end_time = t_frame + half_audio_ms

            if (
                req_start_time >= self.audio_start_ms
                and req_end_time <= current_audio_end_ms
            ):
                valid_center_idx = i
                start_sample = int(
                    ((req_start_time - self.audio_start_ms) / 1000.0) * self.target_sr
                )
                
                # ENFORCE STFT PHASE ALIGNMENT!
                # By forcing the slice start to be an exact multiple of the STFT hop size (256),
                # the STFT frames calculated by inference will exactly match the un-sliced global 
                # STFT frames. This prevents sub-sample phase shifts and guarantees a perfectly 
                # contiguous audio waveform across independent batches!
                start_sample = start_sample - (start_sample % 256)
                end_sample = start_sample + self.target_audio_len

                # Correct bounds
                if end_sample > self.total_audio_samples:
                    end_sample = self.total_audio_samples
                    start_sample = end_sample - self.target_audio_len
                    start_sample = start_sample - (start_sample % 256)
                    end_sample = start_sample + self.target_audio_len

                audio_start_idxFor_slice = start_sample
                audio_end_idxFor_slice = end_sample
                absolute_window_start_sample = self.audio_base_sample + start_sample

                last_start = self.last_processed_window_start_sample
                min_required = (
                    last_start + self.window_min_advance if last_start >= 0 else 0
                )
                if absolute_window_start_sample < max(min_required, last_start + 1):
                    # Not enough timeline advance yet — either audio starvation
                    # (frames ahead of buffered audio) or, for high-frame-rate
                    # models, the enforced hop. Candidate starts only decrease
                    # for older frames, so bail instead of sweeping the whole
                    # ring and logging per frame on the latency path.
                    logger.debug(
                        "No eligible window yet: newest candidate start_sample=%s last_start=%s min_advance=%s frame=%sms",
                        absolute_window_start_sample,
                        last_start,
                        self.window_min_advance,
                        t_frame,
                    )
                    # valid_center_idx was already assigned above; clear it so
                    # this stale candidate is not returned as a window.
                    valid_center_idx = -1
                    break
                
                # Recompute the exact center timestamp of this sample-aligned window
                # so the gRPC server can extract absolute chunks accurately
                true_start_ms = self.audio_start_ms + (start_sample / self.target_sr) * 1000.0
                t_frame = true_start_ms + half_audio_ms
                break

        if valid_center_idx != -1:
            if len(self.audio_chunks) > 1:
                flat_audio = np.concatenate(self.audio_chunks)
                self.audio_chunks = [flat_audio]

            audio_window = self.audio_chunks[0][
                audio_start_idxFor_slice:audio_end_idxFor_slice
            ]
            frames_tuple, trim_start = self._select_frames(valid_center_idx)

            center_timestamp = int(t_frame)
            self.last_processed_timestamp_ms = center_timestamp
            self.last_processed_window_start_sample = absolute_window_start_sample

            # Remove older frames to prevent memory bloat (never trim past the
            # frames the current selection still references)
            self.frame_buffer = self.frame_buffer[trim_start:]

            return (
                audio_window,
                frames_tuple,
                center_timestamp,
                absolute_window_start_sample,
            )

        return None, None, None, None


class InferenceEngine:
    def __init__(self):
        self.is_loaded = False
        self.device = None
        self.use_amp = False

    def load_model(self):
        if self.is_loaded:
            return

        logger.info("Loading models...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = False

        # Optimization: Enable CUDNN autotuner for faster convolutions
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            gpu_name = torch.cuda.get_device_name(0).lower()
            # GTX 16xx (no tensor cores) often runs slower with autocast due to cast overhead.
            # Keep AMP enabled on GPUs that benefit from it.
            self.use_amp = "gtx 16" not in gpu_name
            logger.info(f"CUDA device: {gpu_name} | AMP enabled: {self.use_amp}")

        self.weights = {
            "sound": os.path.join(CKPT_ROOT, "sound_best.pth"),
            "frame": os.path.join(CKPT_ROOT, "frame_best.pth"),
            "synth": os.path.join(CKPT_ROOT, "synthesizer_best.pth"),
        }

        for k, v in self.weights.items():
            if not os.path.exists(v):
                raise FileNotFoundError(f"Missing weights: {v}")

        builder = ModelBuilder()
        self.net_sound = builder.build_sound(
            arch=ARCH_SOUND, fc_dim=NUM_CHANNELS, weights=self.weights["sound"]
        )
        self.net_frame = builder.build_frame(
            arch=ARCH_FRAME,
            fc_dim=NUM_CHANNELS,
            pool_type=IMG_POOL,
            weights=self.weights["frame"],
        )
        self.net_synth = builder.build_synthesizer(
            arch=ARCH_SYNTH, fc_dim=NUM_CHANNELS, weights=self.weights["synth"]
        )

        self.net_sound.to(self.device).eval()
        self.net_frame.to(self.device).eval()
        self.net_synth.to(self.device).eval()

        # Pre-calculate Warp Grids for GPU STFT
        # We force dimensions to exactly 256x256 for the AI model
        self.T_target = 256
        self.F_target = 256
        self.grid_warp = torch.from_numpy(
            warpgrid(1, self.F_target, self.T_target, warp=True)
        ).to(self.device)
        self.grid_unwarp = torch.from_numpy(
            warpgrid(1, 512, self.T_target, warp=False)
        ).to(self.device)

        # Pre-compute STFT window
        self.hann_window = torch.hann_window(STFT_FRAME).to(self.device)

        # Pre-compute video transform
        self._eval_transform = self.build_vid_transform_eval()

        self.is_loaded = True
        mask_mode = "BINARY" if BINARY_MASK else "SOFT (ratio)"
        if MASK_RENORM:
            mask_mode += f" +renorm(floor={MASK_RENORM_FLOOR})"
        if AUDIO_NORM:
            mask_mode += (
                f" | input rms -> {AUDIO_NORM_TARGET_RMS}"
                f" (max {AUDIO_NORM_MAX_GAIN}x, smooth {AUDIO_NORM_SMOOTH})"
            )
        if SEPARATION_MODE == "pixel":
            sep_desc = f"PIXEL-level (top-{SEP_TOPK} activation pixels)"
        else:
            sep_desc = f"VIDEO-level ({IMG_POOL}) [legacy]"
        smooth_desc = (
            f"EMA {MASK_SMOOTH}" if 0.0 < MASK_SMOOTH < 1.0 else "off"
        )
        logger.info(f"Models loaded on {self.device} | AUD_LEN={AUD_LEN} | Mask mode: {mask_mode}")
        logger.info(f"Separation conditioning: {sep_desc} | mask smoothing: {smooth_desc}")

        # Warm up the model
        self._warmup()

    def _warmup(self):
        """Perform a dummy inference to warm up the GPU/cache."""
        logger.info("Warming up inference engine...")
        with torch.no_grad():
            # Use the exact dimensions used in live streaming: [B, 1, F_warped, T_warped]
            dummy_mag = torch.zeros((1, 1, 256, 256)).to(self.device)
            dummy_frame = torch.zeros((1, 3, 3, 224, 224)).to(self.device)
            # Just run the core networks
            feat_sound = self.net_sound(dummy_mag)
            feat_frame = self.net_frame.forward_multiframe(dummy_frame)
            self.net_synth(feat_frame, feat_sound)
        logger.info("Warm-up complete.")

    def _audio_norm_gain(self, y_segment, smooth=False):
        """Gain that brings this window to AUDIO_NORM_TARGET_RMS.

        The network is not scale invariant, and a quiet capture sits well
        outside the level its batch-norm statistics were trained on. Scaling
        the input is exactly invertible: the masks are ratios and the phase is
        untouched, so dividing the reconstruction by the same gain returns the
        original level with no colouration.

        In the streaming path the gain is smoothed across windows. Giving each
        overlap-add window its own gain would make the masks jump at every
        seam, which is audible as pumping.
        """
        if not AUDIO_NORM:
            return 1.0

        y = np.asarray(y_segment, dtype=np.float64)
        if y.size == 0:
            return 1.0

        rms = float(np.sqrt(np.mean(np.square(y))))
        if not np.isfinite(rms) or rms < AUDIO_NORM_MIN_RMS:
            # Silence. Leave it alone rather than amplify the noise floor.
            return 1.0

        gain = AUDIO_NORM_TARGET_RMS / rms
        gain = float(
            np.clip(gain, 1.0 / AUDIO_NORM_MAX_GAIN, AUDIO_NORM_MAX_GAIN)
        )

        if smooth:
            prev = getattr(self, "_norm_gain_ema", None)
            if prev is None:
                self._norm_gain_ema = gain
            else:
                a = float(np.clip(AUDIO_NORM_SMOOTH, 0.0, 1.0))
                self._norm_gain_ema = (1.0 - a) * prev + a * gain
            gain = self._norm_gain_ema

        return gain

    def prepare_audio(self, audio_path):
        y, sr = librosa.load(audio_path, sr=AUD_RATE, mono=True)
        if y.size == 0:
            raise RuntimeError("Empty audio.")

        if y.shape[0] < AUD_LEN:
            n = int(AUD_LEN / float(y.shape[0])) + 1
            y = np.tile(y, n)
        center = y.shape[0] // 2
        start = max(0, center - AUD_LEN // 2)
        end = start + AUD_LEN
        y_segment = y[start:end]

        # Bring the window to the level the network was trained at. One-shot
        # path, so no smoothing.
        audio_gain = self._audio_norm_gain(y_segment, smooth=False)

        # Use GPU STFT for consistency and speed
        y_torch = torch.from_numpy(y_segment).to(self.device).float()
        if audio_gain != 1.0:
            y_torch = y_torch * audio_gain
        window = self.hann_window
        stft = torch.stft(
            y_torch,
            n_fft=STFT_FRAME,
            hop_length=STFT_HOP,
            window=window,
            return_complex=True,
        )

        # Force same dimensions as streaming mode
        stft = stft[:512, :256]

        mag = torch.abs(stft)
        mag_lin_torch = mag.unsqueeze(0).unsqueeze(0)
        phase_np = torch.angle(stft).cpu().numpy()

        return mag_lin_torch, phase_np, y_segment, audio_gain

    def prepare_audio_from_buffer(self, y_segment):
        """Prepare STFT using Torch on GPU for maximum speed."""
        if len(y_segment) != AUD_LEN:
            if len(y_segment) < AUD_LEN:
                y_segment = np.pad(y_segment, (0, AUD_LEN - len(y_segment)))
            else:
                y_segment = y_segment[:AUD_LEN]

        # Bring the window to the level the network was trained at. Streaming
        # path, so the gain is smoothed across consecutive windows.
        audio_gain = self._audio_norm_gain(y_segment, smooth=True)

        # Move raw audio to GPU
        y_torch = torch.from_numpy(y_segment).to(self.device).float()
        if audio_gain != 1.0:
            y_torch = y_torch * audio_gain

        # Use Torch STFT (GPU accelerated)
        window = self.hann_window
        stft = torch.stft(
            y_torch,
            n_fft=STFT_FRAME,
            hop_length=STFT_HOP,
            window=window,
            return_complex=True,
        )

        # Optimization: Ensure exactly 512 frequency bins and 256 time steps
        # stft is [Freq (512), Time (257)]
        # We slice to [512, 256] to match the model's expected power-of-2 dimensions
        stft = stft[:512, :256]

        mag = torch.abs(stft)
        mag_lin_torch = mag.unsqueeze(0).unsqueeze(0)  # [1, 1, 512, 256]
        phase_np = torch.angle(stft).cpu().numpy()

        return mag_lin_torch, phase_np, y_segment, audio_gain

    def build_vid_transform_eval(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # Note: Resize and CenterCrop have been natively relocated to the mobile client Android application.
        # Mobile delivers pristine 224x224 JPEGs per side natively. Let's just normalize!
        transform_list = [
            vtransforms.ToTensor(),
            vtransforms.Normalize(mean, std),
            vtransforms.Stack(),
        ]

        def _transform(frames):
            out = frames
            for t in transform_list:
                out = t(out)
            return out

        return _transform

    def prepare_video_inputs(self, frames_dir, num_frames):
        indices = []
        center = max(1, num_frames // 2)
        for i in range(NUM_FRAMES):
            offset = (i - NUM_FRAMES // 2) * STRIDE_FRAMES
            idx = max(1, min(num_frames, center + offset))
            indices.append(idx)

        transform = self._eval_transform
        left_frames, right_frames = [], []

        # Helper to safely load images
        files = sorted(os.listdir(frames_dir))

        for idx in indices:
            # Handle potential index out of bounds by clamping
            safe_idx = min(len(files), max(1, idx))
            fn = os.path.join(frames_dir, "{:06d}.jpg".format(safe_idx))

            try:
                img = Image.open(fn).convert("RGB")
            except (FileNotFoundError, IOError, OSError):
                # Fallback if file not found (though backend should ensure it exists)
                img = Image.new("RGB", (IMG_SIZE * 2, IMG_SIZE))

            w, h = img.size
            mid = w // 2
            left_frames.append(img.crop((0, 0, mid, h)))
            right_frames.append(img.crop((mid, 0, w, h)))

        left_tensor = transform(left_frames).unsqueeze(0).to(self.device)
        right_tensor = transform(right_frames).unsqueeze(0).to(self.device)

        center_idx = len(indices) // 2
        return (
            left_tensor,
            right_tensor,
            left_frames[center_idx],
            right_frames[center_idx],
        )

    def prepare_video_inputs_from_buffer(self, frames_list):
        """Prepare video tensors directly from a list of PIL Images."""
        transform = self._eval_transform
        left_frames, right_frames = [], []

        for img in frames_list:
            w, h = img.size
            mid = w // 2
            left_frames.append(img.crop((0, 0, mid, h)))
            right_frames.append(img.crop((mid, 0, w, h)))

        left_tensor = transform(left_frames).unsqueeze(0).to(self.device)
        right_tensor = transform(right_frames).unsqueeze(0).to(self.device)

        center_idx = len(frames_list) // 2
        return (
            left_tensor,
            right_tensor,
            left_frames[center_idx],
            right_frames[center_idx],
        )

    def logic_separate_audio(
        self, mag_lin_torch, phase_np, frames_left, frames_right, audio_gain=1.0
    ):
        with torch.no_grad():
            B, _, HS, T_lin = mag_lin_torch.size()
            mag_for_net = mag_lin_torch
            if LOG_FREQ:
                grid_warp = torch.from_numpy(warpgrid(B, 256, T_lin, warp=True)).to(
                    self.device
                )
                mag_for_net = F.grid_sample(mag_for_net, grid_warp, align_corners=True)

            log_mag = torch.log(mag_for_net + 1e-10)
            feat_sound = self.net_sound(log_mag)
            feat_sound = activate(feat_sound, "no")

            feat_frames = [
                activate(self.net_frame.forward_multiframe(f), "sigmoid")
                for f in (frames_left, frames_right)
            ]

            pred_masks = [
                activate(self.net_synth(f, feat_sound), "sigmoid") for f in feat_frames
            ]

            if LOG_FREQ:
                grid_unwarp = torch.from_numpy(
                    warpgrid(B, HS, pred_masks[0].size(3), warp=False)
                ).to(self.device)
                pred_masks = [
                    F.grid_sample(pm, grid_unwarp, align_corners=True)
                    for pm in pred_masks
                ]

            mag_lin_np = mag_lin_torch.squeeze().cpu().numpy()

            # Renormalize across BOTH masks before reconstructing either one,
            # so the pair partitions the mixture's energy. This must happen
            # outside the per-mask loop, and must match eval_stream_window
            # exactly or the file and streaming paths will diverge.
            pm_nps = [pm.squeeze().cpu().numpy() for pm in pred_masks]
            if MASK_RENORM:
                pm_sum = np.clip(
                    np.sum(pm_nps, axis=0),
                    max(MASK_RENORM_FLOOR, MASK_RENORM_EPS),
                    None,
                )
                pm_nps = [pm_np / pm_sum for pm_np in pm_nps]

            wavs = []
            for pm_np in pm_nps:
                if BINARY_MASK:
                    pm_np = (pm_np > MASK_THRES).astype(np.float32)
                pred_mag = mag_lin_np * pm_np
                wav = istft_reconstruction(pred_mag, phase_np, hop_length=STFT_HOP)
                # Undo the input normalization so the caller gets the original
                # level back.
                if audio_gain != 1.0:
                    wav = wav / audio_gain
                wavs.append(np.clip(wav, -1.0, 1.0))

        return wavs

    def logic_compute_heatmap(self, mag_lin_torch, frames_tensor):
        with torch.no_grad():
            B, _, HS, T_lin = mag_lin_torch.size()
            mag_for_net = mag_lin_torch
            if LOG_FREQ:
                grid_warp = torch.from_numpy(warpgrid(B, 256, T_lin, warp=True)).to(
                    self.device
                )
                mag_for_net = F.grid_sample(mag_for_net, grid_warp, align_corners=True)

            log_mag = torch.log(mag_for_net + 1e-10)
            feat_sound = self.net_sound(log_mag)
            feat_sound = activate(feat_sound, "no")

            feats_video = self.net_frame.forward_multiframe(frames_tensor, pool=False)
            feats_img, _ = feats_video.max(dim=2)  # Temporal max-pooling as per paper
            feats_img = activate(
                feats_img, "sigmoid"
            )  # Sigmoid activation as per paper

            # Compute in float32 to prevent float16 underflow to 0.0000
            amp_off_ctx = (
                torch.amp.autocast(device_type="cuda", enabled=False)
                if self.device.type == "cuda"
                else nullcontext()
            )
            with amp_off_ctx:
                feats_img_32 = feats_img.float()
                feat_sound_32 = feat_sound.float()

                z = self.net_synth.forward_pixelwise(feats_img_32, feat_sound_32)
                # Sound of Pixels Localization (Section 3.3):
                # Apply sigmoid to get per-pixel mask, then average across spectral dims.
                # Do NOT multiply by magnitude — that biases toward low-frequency energy.
                mask = torch.sigmoid(z)
                z_spatial = mask.mean(dim=(3, 4))  # [B, HI, WI]

                # Normalize per batch element to avoid batch coupling artifacts
                B_m, HI_m, WI_m = z_spatial.shape
                z_flat = z_spatial.view(B_m, -1)
                z_min = z_flat.min(dim=1, keepdim=True)[0]
                z_norm = z_flat - z_min
                z_max = z_norm.max(dim=1, keepdim=True)[0]
                z_norm = torch.where(z_max > 0, z_norm / z_max, z_norm)
                z_spatial = z_norm.view(B_m, HI_m, WI_m)

            z_spatial = z_spatial.unsqueeze(1)
            z_up = F.interpolate(
                z_spatial,
                size=(IMG_SIZE, IMG_SIZE),
                mode="bilinear",
                align_corners=False,
            )

            return z_up.squeeze(1).squeeze(0).cpu().numpy()

    def get_audio_segment(self, full_audio, frame_idx):
        """
        Extracts the AUD_LEN window centered around the time of frame_idx.
        """
        # Calculate center sample based on frame rate and audio rate
        # e.g., Frame 10 at 8fps = 1.25s. 1.25s * 11025Hz = Sample 13781
        samples_per_frame = AUD_RATE / float(FRAME_RATE)
        center_sample = int(frame_idx * samples_per_frame)

        half_len = AUD_LEN // 2
        start = center_sample - half_len
        end = start + AUD_LEN

        # Padding logic if the window goes out of bounds
        pad_front = 0
        pad_back = 0

        if start < 0:
            pad_front = abs(start)
            start = 0

        if end > len(full_audio):
            pad_back = end - len(full_audio)
            end = len(full_audio)

        y_segment = full_audio[start:end]

        if pad_front > 0 or pad_back > 0:
            y_segment = np.pad(y_segment, (pad_front, pad_back), mode="constant")

        # Ensure exact length
        if len(y_segment) != AUD_LEN:
            # Fallback resizing if off by a sample due to rounding
            y_segment = np.resize(y_segment, (AUD_LEN,))

        # Create STFT Spectrogram
        spec = librosa.stft(y_segment, n_fft=STFT_FRAME, hop_length=STFT_HOP)
        mag = np.abs(spec)
        mag_lin_torch = (
            torch.from_numpy(mag).unsqueeze(0).unsqueeze(0).float().to(self.device)
        )

        return mag_lin_torch

    # --- HELPER: VIDEO TENSOR FOR SPECIFIC FRAME (WITH PADDING) ---
    def _load_frame_safe(self, frames_dir, idx, max_frames, files_list):
        """
        Loads a frame. If idx is out of bounds (padding), returns the closest valid frame.
        """
        # Clamp index to [0, max_frames - 1] (Edge Padding)
        # This acts like "blank frames" but works better for ResNet features than pure black
        safe_idx = max(0, min(max_frames - 1, idx))

        # Assuming files are sorted, use list index
        filename = files_list[safe_idx]
        path = os.path.join(frames_dir, filename)

        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logger.info(f"Error loading frame {filename}: {e}")
            return Image.new("RGB", (IMG_SIZE * 2, IMG_SIZE))

    def get_video_window(
        self, frames_dir, center_idx, max_frames, files_list, transform
    ):
        """
        Prepares the 3-frame stack [center-stride, center, center+stride]
        """
        frames_indices = []
        for i in range(NUM_FRAMES):
            # Calculate offset: -24, 0, +24
            offset = (i - NUM_FRAMES // 2) * STRIDE_FRAMES
            frames_indices.append(center_idx + offset)

        left_frames, right_frames = [], []

        for idx in frames_indices:
            img = self._load_frame_safe(frames_dir, idx, max_frames, files_list)

            w, h = img.size
            mid = w // 2
            left_frames.append(img.crop((0, 0, mid, h)))
            right_frames.append(img.crop((mid, 0, w, h)))

        left_tensor = transform(left_frames).unsqueeze(0).to(self.device)
        right_tensor = transform(right_frames).unsqueeze(0).to(self.device)

        # Return tensors and the center PIL images for overlay
        return left_tensor, right_tensor, left_frames[1], right_frames[1]

    # --- NEW FUNCTION: SLIDING WINDOW EVALUATION ---
    def eval_video_sliding_window(self, audio_path, frames_dir, output_dir):
        """
        Evaluates the model on EVERY frame of the video.
        Generates a sequence of heatmap overlaid images.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Load FULL Audio
        logger.info("Loading full audio...")
        full_audio, sr = librosa.load(audio_path, sr=AUD_RATE, mono=True)

        # 2. List Frames
        files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        logger.info(files)
        num_frames = len(files)
        logger.info(f"Processing {num_frames} frames via sliding window...")

        transform = self._eval_transform

        # 3. Loop through EVERY frame
        # We index from 0 to num_frames-1. The helper functions handle padding.

        for i in range(num_frames):

            if i % 10 == 0:
                logger.info(f"Processing frame {i}/{num_frames}")

            # A. Get Audio Window for this frame
            mag_lin_torch = self.get_audio_segment(full_audio, i)

            # B. Get Video Window (with padding)
            t_left, t_right, p_left, p_right = self.get_video_window(
                frames_dir, i, num_frames, files, transform
            )

            # C. Compute Heatmaps
            map_left = self.logic_compute_heatmap(mag_lin_torch, t_left)
            map_right = self.logic_compute_heatmap(mag_lin_torch, t_right)

            # D. Overlay
            overlay_left = self.overlay_heatmap(p_left, map_left)
            overlay_right = self.overlay_heatmap(p_right, map_right)

            # E. Stitch and Save
            w, h = overlay_left.size
            combined = Image.new("RGB", (w * 2, h))
            combined.paste(overlay_left, (0, 0))
            combined.paste(overlay_right, (w, 0))

            save_path = os.path.join(output_dir, f"heatmap_{i:06d}.jpg")
            combined.save(save_path)

        logger.info(f"Done! Frames saved to {output_dir}")
        return output_dir

    def overlay_heatmap(self, img_pil, heatmap, alpha=0.6):
        img = np.array(img_pil.convert("RGB"))
        h, w, _ = img.shape
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_norm = 1.0 - np.clip(heatmap_resized, 0.0, 1.0)
        heatmap_color = cv2.applyColorMap(
            (heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(img[:, :, ::-1], 1.0 - alpha, heatmap_color, alpha, 0)
        return Image.fromarray(overlay[:, :, ::1])

    def eval_stream_window(self, audio_array, frames_tuple):
        """Inference that runs synchronously on memory buffers for bidirectional streaming. (B=2 Stereo Batching)"""
        import gc

        # Disable automatic GC during this critical latency path to prevent 300ms OS stutters
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
            # Raise gen-1/2 thresholds so they trigger far less frequently
            # Default (700, 10, 10) causes gen-1 every ~10 inferences → 350ms spike
            # (700, 50, 50) pushes spikes out to every ~50 inferences
            gc.set_threshold(700, 50, 50)

        try:
            with torch.inference_mode():
                # Use AMP for throughput on GTX-class GPUs; keep heatmap math in fp32.
                amp_ctx = (
                    torch.amp.autocast(
                        device_type="cuda",
                        enabled=(self.device.type == "cuda" and self.use_amp),
                    )
                    if self.device.type == "cuda"
                    else nullcontext()
                )
                with amp_ctx:
                    # 1. Prepare Data
                    (
                        mag_lin_torch,
                        phase_np,
                        _,
                        audio_gain,
                    ) = self.prepare_audio_from_buffer(audio_array)

                    left_frames, right_frames = frames_tuple
                    transform = self._eval_transform

                    left_tensor = transform(left_frames).unsqueeze(0).to(self.device)
                    right_tensor = transform(right_frames).unsqueeze(0).to(self.device)

                    # Merge streams sequentially to batch dimensions: Shape: [2, 3, T, H, W]
                    frames_tensor = torch.cat([left_tensor, right_tensor], dim=0)

                    # For legacy fallback contexts (unlikely strictly used)
                    pil_left_center = left_frames[1]
                    pil_right_center = right_frames[1]

                    # --- A. AUDIO FEATURES ---
                    mag_for_net = F.grid_sample(
                        mag_lin_torch, self.grid_warp, align_corners=True
                    )
                    log_mag = torch.log(mag_for_net + 1e-10)
                    feat_sound = self.net_sound(log_mag)
                    feat_sound = activate(feat_sound, "no")

                    # --- B. VISION FEATURES (Single Pass) ---
                    feats_video = self.net_frame.forward_multiframe(
                        frames_tensor, pool=False
                    )

                    # --- C. AUDIO SEPARATION ---
                    feat_sound_expanded = feat_sound.expand(
                        2, -1, -1, -1
                    )  # Broadcast audio parameters for both images

                    # Cached for the heatmap stage further down, so the expensive
                    # forward_pixelwise call happens exactly once per cycle.
                    cached_spatial_act = None

                    if SEPARATION_MODE == "pixel":
                        # ------------------------------------------------------
                        # PIXEL-LEVEL CONDITIONING (paper Section 3.3, test-time)
                        #
                        # The previous code used adaptive_max_pool3d to collapse
                        # T, H and W into ONE 32-d vector per crop, then sigmoid.
                        # Taking the max over 3x14x14 = 588 positions saturates
                        # nearly every channel, so both crops - each just "a
                        # musician in a room" - produced almost the same vector
                        # (measured cosine similarity 0.85 to 0.98).
                        #
                        # The synthesizer is a plain inner product with scale ~= 1
                        # (measured min 0.971, max 1.052), so near-identical vision
                        # vectors give near-identical masks. After renormalisation
                        # that is a 50/50 split: both instruments in both channels.
                        #
                        # Instead, condition on the most sound-active PIXELS of
                        # each crop, which is what the paper prescribes at test
                        # time. forward_pixelwise gives the mask at every pixel;
                        # we pick the top-K by activation and average them.
                        # ------------------------------------------------------
                        feats_img_px = feats_video.max(dim=2)[0]  # [2, C, HI, WI]
                        feats_img_px = activate(feats_img_px, "sigmoid")

                        px_ctx = (
                            torch.amp.autocast(device_type="cuda", enabled=False)
                            if self.device.type == "cuda"
                            else nullcontext()
                        )
                        with px_ctx:
                            z_px = self.net_synth.forward_pixelwise(
                                feats_img_px.float(),
                                feat_sound_expanded.float(),
                            )  # [2, HI, WI, HS, WS]
                            mask_px = torch.sigmoid(z_px)

                            # Spatial activation = how much sound this pixel claims.
                            # This is exactly the heatmap, so cache it for reuse.
                            spatial_act = mask_px.mean(dim=(3, 4))  # [2, HI, WI]
                            cached_spatial_act = spatial_act

                            B_px, HI_px, WI_px = spatial_act.shape
                            HS_px, WS_px = mask_px.shape[3], mask_px.shape[4]

                            act_flat = spatial_act.reshape(B_px, -1)
                            k = int(max(1, min(SEP_TOPK, act_flat.shape[1])))
                            top_v, top_i = act_flat.topk(k, dim=1)  # [2, k]

                            mask_flat = mask_px.reshape(
                                B_px, HI_px * WI_px, HS_px, WS_px
                            )
                            gather_idx = top_i.view(B_px, k, 1, 1).expand(
                                -1, -1, HS_px, WS_px
                            )
                            sel = mask_flat.gather(1, gather_idx)  # [2, k, HS, WS]

                            # Activation-weighted average of the top-K pixel masks.
                            w = top_v / top_v.sum(dim=1, keepdim=True).clamp_min(1e-8)
                            pred_mask = (sel * w.view(B_px, k, 1, 1)).sum(dim=1)
                            pred_mask = pred_mask.unsqueeze(1)  # [2, 1, HS, WS]

                            del mask_px, z_px, mask_flat, sel
                    else:
                        # Legacy video-level pooling (the paper's TRAIN-time mode).
                        def pool_vision(x):
                            B = x.size(0)
                            if IMG_POOL == "avgpool":
                                return F.adaptive_avg_pool3d(x, 1).view(B, -1)
                            else:
                                return F.adaptive_max_pool3d(x, 1).view(B, -1)

                        feat_vision_pooled = activate(
                            pool_vision(feats_video), "sigmoid"
                        )  # [2, C_feat]

                        pred_mask = activate(
                            self.net_synth(feat_vision_pooled, feat_sound_expanded),
                            "sigmoid",
                        )

                    # Unwarp
                    grid_unwarp_expanded = self.grid_unwarp.expand(2, -1, -1, -1)
                    pred_mask = F.grid_sample(
                        pred_mask.to(mag_lin_torch.dtype),
                        grid_unwarp_expanded,
                        align_corners=True,
                    )

                    # --- AUDIO RECONSTRUCTION ON GPU ---
                    pm_torch = pred_mask.squeeze(1)  # [2, HS, T]

                    # --- MASK RENORMALIZATION ---
                    # Ratio masks must partition the mixture: maskL + maskR = 1
                    # at every T-F bin. Two independent sigmoids do not
                    # guarantee that, and when they undershoot, the output
                    # loses nearly all of its energy.
                    #
                    # This cannot amplify anything: m_i / (m_L + m_R) is in
                    # [0, 1] by construction, so each output bin is at most the
                    # mixture bin and the pair sums to exactly the mixture. The
                    # epsilon only guards 0/0 in digitally silent bins.
                    # --- MASK TEMPORAL SMOOTHING ---
                    # Consecutive windows overlap by 97.9% (hop 1378 of 65536),
                    # so their audio content is virtually identical and their
                    # masks should be too. They are computed independently and
                    # flicker, and overlap-add of flickering masks is exactly
                    # what a robotic / warbling artifact sounds like. An EMA
                    # across windows removes the flicker without adding latency.
                    if 0.0 < MASK_SMOOTH < 1.0:
                        prev_mask = getattr(self, "_mask_ema", None)
                        if prev_mask is None or prev_mask.shape != pm_torch.shape:
                            self._mask_ema = pm_torch.clone()
                        else:
                            self._mask_ema = (
                                MASK_SMOOTH * pm_torch
                                + (1.0 - MASK_SMOOTH) * prev_mask
                            )
                        pm_torch = self._mask_ema

                    if MASK_RENORM:
                        pm_sum = pm_torch.sum(dim=0, keepdim=True)
                        pm_torch = pm_torch / torch.clamp(
                            pm_sum, min=max(MASK_RENORM_FLOOR, MASK_RENORM_EPS)
                        )

                    # Apply Binary Mask if configured (now on GPU)
                    if BINARY_MASK:
                        pm_torch = (pm_torch > MASK_THRES).float()

                    mag_lin_torch_sq = mag_lin_torch.squeeze(0).squeeze(0)  # [HS, T]
                    pred_mag_torch = (
                        mag_lin_torch_sq.unsqueeze(0) * pm_torch
                    )  # [2, HS, T]

                    # Use phase from earlier (it was a numpy array, move back to torch)
                    phase_torch = torch.from_numpy(phase_np).to(self.device)

                    # Reconstruct complex spectrogram using batched parameters
                    spec_reconstruct = torch.polar(
                        pred_mag_torch, phase_torch.unsqueeze(0)
                    )  # [2, HS, T]

                    # Inverse STFT on GPU natively handles massive batching mathematically
                    window = self.hann_window
                    wav_torch = torch.istft(
                        spec_reconstruct,
                        n_fft=STFT_FRAME,
                        hop_length=STFT_HOP,
                        window=window,
                        length=AUD_LEN,
                    )

                    # Undo the input normalization so playback level matches
                    # the microphone level.
                    if audio_gain != 1.0:
                        wav_torch = wav_torch / audio_gain

                    # --- D. VISUAL HEATMAPS ---
                    # Fast batched heatmaps using already computed features to keep
                    # per-cycle latency under real-time streaming budget.
                    # Keep heatmap branch in full fp32 to avoid underflow/saturation artifacts.
                    heatmap_ctx = (
                        torch.amp.autocast(device_type="cuda", enabled=False)
                        if self.device.type == "cuda"
                        else nullcontext()
                    )
                    with heatmap_ctx:
                        if cached_spatial_act is not None:
                            # Pixel-mode separation already computed the pixelwise
                            # mask and reduced it over the spectral dims. Reuse it
                            # instead of running forward_pixelwise a second time
                            # (saves ~103 MB of allocation and a large bmm per cycle).
                            z_spatial = cached_spatial_act  # [2, HI, WI]
                        else:
                            feats_img, _ = feats_video.max(dim=2)  # [2, C, HI, WI]
                            feats_img = activate(feats_img, "sigmoid")
                            feat_sound_for_heatmap = feat_sound.expand(2, -1, -1, -1)
                            z = self.net_synth.forward_pixelwise(
                                feats_img.float(),
                                feat_sound_for_heatmap.float(),
                            )  # [2, HI, WI, HS, WS]
                            # Sound of Pixels Localization (Section 3.3):
                            # Sigmoid mask averaged across spectral dims.
                            # mean (not amax): amax over 256x256 = 65536 sigmoid
                            # values saturates every pixel to ~1.0 and washes out
                            # all localization contrast. mean matches /predict.
                            mask = torch.sigmoid(z)
                            z_spatial = mask.mean(dim=(3, 4))  # [2, HI, WI]
                            del mask, z

                        # Normalize each side independently
                        B_hm, HI_hm, WI_hm = z_spatial.shape
                        z_flat = z_spatial.view(B_hm, -1)
                        z_min = z_flat.min(dim=1, keepdim=True)[0]
                        z_norm = z_flat - z_min
                        z_max = z_norm.max(dim=1, keepdim=True)[0]
                        z_norm = torch.where(z_max > 0, z_norm / z_max, z_norm)
                        z_spatial = z_norm.view(B_hm, 1, HI_hm, WI_hm)

                        # FIX: raise resolution from 24x24 to 56x56 to match the
                        # detail level of the /predict endpoint (224x224). 56x56
                        # = 3136 bytes/side, still negligible vs gRPC bandwidth.
                        z_up = F.interpolate(
                            z_spatial,
                            size=(56, 56),
                            mode="bilinear",
                            align_corners=False,
                        )
                        heatmap_tensor_batch = (
                            z_up.squeeze(1).cpu().numpy().astype(np.float32)
                        )
                    wav_final_batch = (
                        torch.clamp(wav_torch, -1.0, 1.0).cpu().numpy()
                    )  # [2, T]

                    # Guard against NaN from AMP float16 edge cases (transient on first inferences)
                    if np.isnan(heatmap_tensor_batch).any():
                        heatmap_tensor_batch = np.nan_to_num(
                            heatmap_tensor_batch, nan=0.0
                        )
                    if np.isnan(wav_final_batch).any():
                        wav_final_batch = np.nan_to_num(wav_final_batch, nan=0.0)
        finally:
            if gc_was_enabled:
                gc.enable()
            # Perform a generation 0 collection only so it completes instantly and prevents large buildups
            gc.collect(0)

        # Scalars for the streaming diagnostic recorder. Each .item() forces a
        # GPU sync, so only compute them when the recorder is actually on.
        diag = None
        if STREAM_DUMP:
            try:
                diag = {
                    "audio_gain": float(audio_gain),
                    "mask_mean_left": float(pm_torch[0].float().mean().item()),
                    "mask_mean_right": float(pm_torch[1].float().mean().item()),
                    "in_rms": float(
                        np.sqrt(np.mean(np.square(np.asarray(audio_array, dtype=np.float64))))
                    ),
                    "out_rms_left": float(
                        np.sqrt(np.mean(np.square(wav_final_batch[0].astype(np.float64))))
                    ),
                    "out_rms_right": float(
                        np.sqrt(np.mean(np.square(wav_final_batch[1].astype(np.float64))))
                    ),
                }
            except Exception:
                diag = None

        return {
            "left_audio": wav_final_batch[0],
            "right_audio": wav_final_batch[1],
            "left_heatmap": heatmap_tensor_batch[0],
            "right_heatmap": heatmap_tensor_batch[1],
            "left_center_frame": pil_left_center,
            "right_center_frame": pil_right_center,
            "diag": diag,
        }

    def eval_pixel_window(self, audio_array, full_frames):
        """Pixel-mode feature pass: run the two expensive nets ONCE for a
        window of full letterboxed frames and return the fp32 intermediates
        the region cache needs. No masks are synthesized here — every spatial
        selection is served later against the cache (PIXEL_PLAN section 2).

        audio_array: float32 mono [-1, 1], AUD_LEN samples.
        full_frames: 3 PIL RGB 224x224 letterboxed frames (centered triple).

        Returns dict:
            feat_sound [1, K, 256, 256] fp32, feats_img [1, K, 14, 14] fp32
            (post-sigmoid, temporal-maxed — exactly the deployed pixel-path
            recipe), mag_lin [512, 256] fp32, phase [512, 256] fp32,
            audio_gain float.
        """
        with torch.inference_mode():
            amp_ctx = (
                torch.amp.autocast(
                    device_type="cuda",
                    enabled=(self.device.type == "cuda" and self.use_amp),
                )
                if self.device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                # Audio prep mirrors prepare_audio_from_buffer, but with a
                # PIXEL-PRIVATE gain EMA. prepare_audio_from_buffer(smooth=True)
                # mutates self._norm_gain_ema — the halves streams' state — and
                # sharing it would break the halves byte-exactness promise the
                # moment a pixel stream and a halves stream coexist in one
                # process (caught in review).
                y_segment = audio_array
                if len(y_segment) != AUD_LEN:
                    if len(y_segment) < AUD_LEN:
                        y_segment = np.pad(y_segment, (0, AUD_LEN - len(y_segment)))
                    else:
                        y_segment = y_segment[:AUD_LEN]

                inst_gain = self._audio_norm_gain(y_segment, smooth=False)
                prev = getattr(self, "_pixel_gain_ema", None)
                if prev is None:
                    audio_gain = inst_gain
                else:
                    a = float(np.clip(AUDIO_NORM_SMOOTH, 0.0, 1.0))
                    audio_gain = (1.0 - a) * prev + a * inst_gain
                self._pixel_gain_ema = audio_gain

                y_torch = torch.from_numpy(y_segment).to(self.device).float()
                if audio_gain != 1.0:
                    y_torch = y_torch * audio_gain
                stft = torch.stft(
                    y_torch,
                    n_fft=STFT_FRAME,
                    hop_length=STFT_HOP,
                    window=self.hann_window,
                    return_complex=True,
                )[:512, :256]
                mag_lin_torch = stft.abs().unsqueeze(0).unsqueeze(0)  # [1, 1, 512, 256]
                phase_np = torch.angle(stft).cpu().numpy()

                frames_tensor = (
                    self._eval_transform(full_frames).unsqueeze(0).to(self.device)
                )  # [1, 3, T, 224, 224]

                mag_for_net = F.grid_sample(
                    mag_lin_torch, self.grid_warp, align_corners=True
                )
                log_mag = torch.log(mag_for_net + 1e-10)
                feat_sound = activate(self.net_sound(log_mag), "no")

                feats_video = self.net_frame.forward_multiframe(
                    frames_tensor, pool=False
                )  # [1, K, T, 14, 14]
                # Temporal max then sigmoid — the deployed pixel-path recipe
                # (and the reason region MAX-pooling later equals video-level
                # pooling restricted to the region: sigmoid is monotonic).
                feats_img = activate(feats_video.max(dim=2)[0], "sigmoid")

        return {
            "feat_sound": feat_sound.float().clone(),
            "feats_img": feats_img.float().clone(),
            "mag_lin": mag_lin_torch.squeeze(0).squeeze(0).float().clone(),
            "phase": torch.from_numpy(phase_np).to(self.device).float(),
            "audio_gain": float(audio_gain),
        }

    def eval_for_grpc(self, audio_path, frames_dir, num_frames):
        """Inference that returns raw arrays instead of files (for gRPC response).

        Returns:
            dict with:
                - left_audio: numpy array of separated left audio (float32, clipped to [-1,1])
                - right_audio: numpy array of separated right audio
                - left_heatmap: numpy array of left heatmap (float32, [224x224])
                - right_heatmap: numpy array of right heatmap
                - left_center_frame: PIL Image of the left center frame
                - right_center_frame: PIL Image of the right center frame
        """
        # 1. Prepare Data
        mag_lin_torch, phase_np, _, audio_gain = self.prepare_audio(audio_path)
        frames_left, frames_right, pil_left, pil_right = self.prepare_video_inputs(
            frames_dir, num_frames
        )

        # 2. Audio Separation
        wavs = self.logic_separate_audio(
            mag_lin_torch, phase_np, frames_left, frames_right, audio_gain=audio_gain
        )

        # 3. Visual Heatmaps
        map_left = self.logic_compute_heatmap(mag_lin_torch, frames_left)
        map_right = self.logic_compute_heatmap(mag_lin_torch, frames_right)

        return {
            "left_audio": wavs[0],  # numpy float32 array
            "right_audio": wavs[1],  # numpy float32 array
            "left_heatmap": map_left,  # numpy float32 [224x224]
            "right_heatmap": map_right,  # numpy float32 [224x224]
            "left_center_frame": pil_left,  # PIL Image
            "right_center_frame": pil_right,  # PIL Image
        }

    def eval_single_heatmap(self, audio_path, frames_dir, num_frames, output_dir):
        """
        Generates Left Audio, Right Audio, and Combined Heatmap Image.
        """
        # 1. Prepare Data
        mag_lin_torch, phase_np, _, audio_gain = self.prepare_audio(audio_path)
        frames_left, frames_right, pil_left, pil_right = self.prepare_video_inputs(
            frames_dir, num_frames
        )

        # 2. Audio Separation
        wavs = self.logic_separate_audio(
            mag_lin_torch, phase_np, frames_left, frames_right, audio_gain=audio_gain
        )

        path_aud_left = os.path.join(output_dir, "pred_left.wav")
        path_aud_right = os.path.join(output_dir, "pred_right.wav")

        # Save Audio
        sf.write(path_aud_left, wavs[0], AUD_RATE)
        sf.write(path_aud_right, wavs[1], AUD_RATE)

        # 3. Visual Heatmaps
        map_left = self.logic_compute_heatmap(mag_lin_torch, frames_left)
        map_right = self.logic_compute_heatmap(mag_lin_torch, frames_right)

        # 4. Create Overlays
        overlay_left = self.overlay_heatmap(pil_left, map_left)
        overlay_right = self.overlay_heatmap(pil_right, map_right)

        # 5. Stitch Images Together (Left + Right)
        w, h = overlay_left.size
        # Create a new image twice as wide
        combined_img = Image.new("RGB", (w * 2, h))
        combined_img.paste(overlay_left, (0, 0))
        combined_img.paste(overlay_right, (w, 0))

        path_combined = os.path.join(output_dir, "pred_combined.jpg")
        combined_img.save(path_combined)

        return {
            "audio_left": path_aud_left,
            "audio_right": path_aud_right,
            "image_combined": path_combined,
        }


# Global Instance
inference = InferenceEngine()


if __name__ == "__main__":
    audio_path = r"src\testing_model\audio\video.mp3"
    frames_dir = r"src\testing_model\frames"
    output_dir = r"src\testing_model\output_frames"

    inference.load_model()

    # Run Sliding Window
    inference.eval_video_sliding_window(audio_path, frames_dir, output_dir)
