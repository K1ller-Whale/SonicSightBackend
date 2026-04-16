import os
import subprocess
import numpy as np
import cv2
import soundfile as sf
import logging

logger = logging.getLogger(__name__)

# MoviePy compatibility
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

from config import AUD_RATE


class PreprocessResult:
    """Result container for video preprocessing."""

    def __init__(self, video_path, audio_path, frames_dir, frame_count):
        self.video_path = video_path
        self.audio_path = audio_path
        self.frames_dir = frames_dir
        self.frame_count = frame_count


class VideoPreprocessor:
    """Handles FFmpeg preprocessing and audio/frame extraction.

    The FFmpeg configuration matches the mobile VideoProcessor.kt exactly:
      -r 8 -vf scale=256:256 -preset ultrafast -c:a aac -b:a 32k -ac 1 -ar 11025
    """

    TARGET_SAMPLE_RATE = AUD_RATE  # 11025

    def preprocess(self, raw_input_path, output_dir):
        """Full preprocessing pipeline: FFmpeg → extract audio → extract frames.

        Args:
            raw_input_path: Path to the raw input video file.
            output_dir: Base directory for all output files.

        Returns:
            PreprocessResult with paths to processed video, audio, frames dir, and frame count.
        """
        processed_path = os.path.join(output_dir, "processed.mp4")
        audio_dir = os.path.join(output_dir, "audio")
        frames_dir = os.path.join(output_dir, "frames")

        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)

        # Step 1: FFmpeg preprocess (identical to mobile VideoProcessor.kt)
        self._run_ffmpeg(raw_input_path, processed_path)

        # Step 2: Extract audio
        audio_path = os.path.join(audio_dir, "audio.wav")
        self._extract_audio(processed_path, audio_path)

        # Step 3: Extract frames
        frame_count = self._extract_frames(processed_path, frames_dir)

        return PreprocessResult(processed_path, audio_path, frames_dir, frame_count)

    def preprocess_already_processed(self, video_path, output_dir):
        """For backward compatibility: skip FFmpeg, just extract audio and frames.

        Used when the video has already been preprocessed (e.g., by mobile FFmpeg).

        Args:
            video_path: Path to the already-processed video file.
            output_dir: Base directory for all output files.

        Returns:
            PreprocessResult with paths.
        """
        audio_dir = os.path.join(output_dir, "audio")
        frames_dir = os.path.join(output_dir, "frames")

        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)

        audio_path = os.path.join(audio_dir, "audio.wav")
        self._extract_audio(video_path, audio_path)

        frame_count = self._extract_frames(video_path, frames_dir)

        return PreprocessResult(video_path, audio_path, frames_dir, frame_count)

    def _run_ffmpeg(self, input_path, output_path):
        """Run FFmpeg with the exact same config as mobile VideoProcessor.kt."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-r", "8",
            "-vf", "scale=256:256",
            "-preset", "ultrafast",
            "-c:a", "aac",
            "-b:a", "32k",
            "-ac", "1",
            "-ar", str(self.TARGET_SAMPLE_RATE),
            output_path,
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg preprocessing failed: {stderr}")

        logger.info(f"FFmpeg preprocessing complete: {output_path}")

    def _extract_audio(self, video_path, audio_path):
        """Extract audio from video using MoviePy (same as current main.py)."""
        clip = VideoFileClip(video_path)
        try:
            if clip.audio is None:
                # Handle silent video (create silence matching video duration)
                sr = self.TARGET_SAMPLE_RATE
                silent_audio = np.zeros(
                    (int(clip.duration * sr),), dtype=np.float32
                )
                sf.write(audio_path, silent_audio, sr)
            else:
                clip.audio.write_audiofile(
                    audio_path,
                    fps=self.TARGET_SAMPLE_RATE,
                    codec="pcm_s16le",
                    nbytes=2,
                    logger=None,
                )
        finally:
            clip.close()

        logger.info(f"Audio extracted: {audio_path}")

    def _extract_frames(self, video_path, frames_dir):
        """Extract all frames from video using OpenCV (same as current main.py)."""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        idx = 1

        while True:
            success, frame = cap.read()
            if not success:
                break
            cv2.imwrite(os.path.join(frames_dir, f"{idx:06d}.jpg"), frame)
            idx += 1
            frame_count += 1

        cap.release()
        logger.info(f"Extracted {frame_count} frames to {frames_dir}")
        return frame_count
