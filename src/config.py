import os
from pathlib import Path

ARCH_SOUND = "unet7"
ARCH_FRAME = "resnet18dilated"
ARCH_SYNTH = "linear"
NUM_CHANNELS = 32
NUM_MIX = 2
NUM_FRAMES = 3
STRIDE_FRAMES = 24
FRAME_RATE = 8
IMG_SIZE = 224
AUD_LEN = 65536
AUD_RATE = 11025
STFT_FRAME = 1022
STFT_HOP = 256
LOG_FREQ = 1
# Configurable via environment variable: 0 = soft (ratio) masking, 1 = binary masking.
# Soft masking produces smoother separation; binary produces sharper but harsher separation.
BINARY_MASK = int(os.environ.get("SONICSIGHT_BINARY_MASK", "0"))
MASK_THRES = 0.5

# Ratio masks for N sources satisfy sum_n M_n = 1 by construction, but the
# network predicts each mask with an independent sigmoid, so nothing enforces
# that identity at inference time. Renormalizing restores it.
# Validated on two real clips: transformative on a quiet one (energy retention
# 5.4% -> 85%, output correlation 0.665 -> 0.251) and harmless on a healthy one
# (retention 0.907 -> 0.922, correlation 0.202 -> 0.185).
MASK_RENORM = int(os.environ.get("SONICSIGHT_MASK_RENORM", "1"))

# Optional floor on the divisor: divide by max(maskL + maskR, FLOOR).
#
# DISABLED BY DEFAULT, and the reasoning that originally motivated it was
# wrong. The floor was added to stop near-silent bins being "amplified without
# bound". No such amplification is possible:
#
#     normalized mask = m_i / (m_L + m_R)
#     m_i >= 0 and m_i <= (m_L + m_R)  =>  ratio is always in [0, 1]
#
# So each output bin is at most the mixture bin, and the two outputs sum to
# exactly the mixture. Renormalization can only redistribute energy, never
# create it. Confirmed empirically: measured noise gain on the quietest 25% of
# frames was 0.80-0.88 on both test clips, i.e. below 1.0.
#
# What the floor actually does is attenuate the fix in precisely the low-sum
# bins the fix exists to repair:
#     mask sum 0.243 -> floor 0.25 keeps 97% of the correction
#     mask sum 0.10  -> floor 0.25 keeps 40%
#     mask sum 0.01  -> floor 0.25 keeps  4%
# Measured on the quiet clip: floor 0.25 -> retention 0.459, no floor -> 0.837.
#
# The genuine edge case is 0/0, and a plain epsilon handles that. Leave this at
# 0.0 unless you have a measurement that says otherwise.
MASK_RENORM_FLOOR = float(os.environ.get("SONICSIGHT_MASK_RENORM_FLOOR", "0.0"))

# Guards division by zero in digitally silent bins. Not a tunable.
MASK_RENORM_EPS = 1e-8

# --- SEPARATION CONDITIONING MODE ---
# "pixel"  : condition the mask on the most sound-active PIXELS of each crop
#            (Sound of Pixels paper, Section 3.3 - the prescribed TEST-time mode).
# "video"  : condition on a single globally max-pooled vector per crop
#            (the paper's TRAIN-time mode; saturates and de-discriminates at test).
SEPARATION_MODE = os.environ.get("SONICSIGHT_SEPARATION_MODE", "video")

# How many top-activation pixels to average the mask over in "pixel" mode.
# 1 = single peak pixel (sharpest separation, most sensitive to jitter)
# 4-8 = robust average over the instrument region (recommended)
SEP_TOPK = int(os.environ.get("SONICSIGHT_SEP_TOPK", "4"))

# --- PIXEL-MODE REGION CONDITIONING (pixel_cache.synthesize_regions) ---
# "mask_topk"       : a region's mask is the energy-weighted average of its
#                     most sound-active CELL masks — the same fix applied to
#                     the halves path above, restricted to the region. A
#                     max-pooled complement feature vector saturates (191 of
#                     196 post-sigmoid cells), which made a long-press play a
#                     mildly re-weighted mixture: both instruments audible.
# "feature_maxpool" : legacy — one mask from a max-pooled region feature
#                     vector (kept for A/B listening via pixel_ab_harness).
REGION_CONDITIONING = os.environ.get("SONICSIGHT_REGION_CONDITIONING", "mask_topk")

# Verbose per-query/per-window pixel diagnostics (cell choices, gate
# threshold, mask statistics). Costs a few host syncs and log lines per
# synthesis — debugging sessions only.
PIXEL_DEBUG = int(os.environ.get("SONICSIGHT_PIXEL_DEBUG", "0"))

# Default mask variant for pixel-path region synthesis (the deferred B1
# decision, resolved by measurement on canon_d.mp4 2026-08-15, real ckpt):
# soft ratio taps leak — a cello tap's region correlated 0.989 with its own
# complement (the complement's top-energy cells are the same instrument, so
# renorm flattens); binary taps separated cleanly (guitar tap corr vs
# A_left/right 0.553/0.047, cello tap 0.165/0.932). Pixel-only: the halves
# path keeps its own BINARY_MASK gate.
PIXEL_TAP_BINARY = int(os.environ.get("SONICSIGHT_PIXEL_TAP_BINARY", "1"))

# EMA weight for the NEW mask in sticky-follow synthesis (1.0 = off).
# Rationale mirrors halves MASK_SMOOTH (unwarp -> EMA -> renorm -> binary):
# sticky windows overlap 97.9% but masks are computed independently, and
# under the binary default the renormed ratio rides the 0.5 threshold —
# independent per-window masks flip bins wholesale. Measured on canon_d
# (2026-08-16, 48 consecutive hop windows, cello follow): 18.6% of ALL
# spectrogram bins flip per 125 ms window without smoothing (max 44%) —
# heard live as robotic/static (log 2026-08-15: raw region/complement
# cos 0.97-0.99, knife-edge). EMA 0.3 cuts the flip rate to 7.2% mean.
# (Track-level rms is NOT the metric — those swings are the music's own
# dynamics and EMA rightly leaves them alone.)
PIXEL_STICKY_SMOOTH = float(os.environ.get("SONICSIGHT_PIXEL_STICKY_SMOOTH", "0.3"))

# --- MASK TEMPORAL SMOOTHING ---
# Consecutive streaming windows overlap by 97.9% (hop 1378 of 65536 samples),
# so their masks SHOULD be nearly identical. They are computed independently
# and flicker instead, which is what produces the robotic/warbling artifact.
# EMA weight for the new mask: 1.0 = no smoothing, 0.3 = heavy smoothing.
MASK_SMOOTH = float(os.environ.get("SONICSIGHT_MASK_SMOOTH", "1.0"))

# Input level normalization before the STFT.
# The net is not scale invariant: scaling the waveform by s shifts its log
# spectrogram by log(s), which moves the whole input off the distribution the
# batch-norm statistics were trained on. A quiet capture (rms 0.025) separated
# far worse than the same content at rms 0.10.
# Measured with renormalization enabled:
#   quiet clip   none -> rms 0.10 : correlation 0.503 -> 0.247, retention 0.837 -> 0.921
#   healthy clip none -> rms 0.10 : correlation 0.189 -> 0.188 (no change)
# Helps a lot where it is needed, does nothing where it is not.
AUDIO_NORM = int(os.environ.get("SONICSIGHT_AUDIO_NORM", "1"))
AUDIO_NORM_TARGET_RMS = float(
    os.environ.get("SONICSIGHT_AUDIO_NORM_TARGET_RMS", "0.10")
)

# Cap the gain so a near-silent window cannot be blown up into noise.
AUDIO_NORM_MAX_GAIN = float(os.environ.get("SONICSIGHT_AUDIO_NORM_MAX_GAIN", "20.0"))

# Windows below this rms are treated as silence and left alone.
AUDIO_NORM_MIN_RMS = float(os.environ.get("SONICSIGHT_AUDIO_NORM_MIN_RMS", "1e-4"))

# EMA coefficient applied to the gain in the STREAMING path only, so that
# consecutive overlap-add windows do not each get a different gain. Without
# this, the masks would shift window to window and the seams would pump.
# 1.0 = no smoothing (instant), lower = slower. The file path is one-shot and
# never smooths.
AUDIO_NORM_SMOOTH = float(os.environ.get("SONICSIGHT_AUDIO_NORM_SMOOTH", "0.3"))

# --- STREAMING DIAGNOSTIC RECORDER ---
# Set SONICSIGHT_DUMP_STREAM=1 to record a full diagnostic bundle for each
# streaming session (microphone input, identity-mask control, the exact PCM
# sent to the client, raw model windows, and per-chunk / per-cycle CSV logs).
# Buffered in memory and written when the stream ends, so it does not add
# disk I/O to the latency path. Leave off for normal use.
STREAM_DUMP = int(os.environ.get("SONICSIGHT_DUMP_STREAM", "0"))
STREAM_DUMP_DIR = os.environ.get(
    "SONICSIGHT_DUMP_DIR",
    str(Path(__file__).resolve().parent / "outputs" / "streamdump"),
)

IMG_POOL = "maxpool"
CKPT_ROOT = str(Path(__file__).resolve().parent / "ckpt")
