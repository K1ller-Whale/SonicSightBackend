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
AUD_LEN = 65535
AUD_RATE = 11025
STFT_FRAME = 1022
STFT_HOP = 256
LOG_FREQ = 1
BINARY_MASK = 1
MASK_THRES = 0.5
IMG_POOL = "maxpool"
CKPT_ROOT = str(Path(__file__).resolve().parent / "ckpt")
