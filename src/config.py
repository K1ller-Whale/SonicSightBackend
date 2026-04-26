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
IMG_POOL = "maxpool"
CKPT_ROOT = str(Path(__file__).resolve().parent / "ckpt")
