import os
import sys
import cv2
import numpy as np
import shutil
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from video_preprocessor import VideoPreprocessor

def create_dummy_video(path, duration=2, fps=30, size=(640, 480)):
    """Creates a dummy MP4 video file for testing."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, size)

    for i in range(duration * fps):
        # Create a frame with some text/color change
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)

    out.release()

# Use a try-except or check for pytest for the fixture/decorator
try:
    import pytest
    @pytest.fixture
    def test_env():
        test_dir = Path("tests/temp_data")
        test_dir.mkdir(parents=True, exist_ok=True)
        input_path = test_dir / "dummy_input.mp4"
        output_dir = test_dir / "output"

        create_dummy_video(str(input_path))

        yield input_path, output_dir

        # Cleanup after test
        shutil.rmtree(test_dir)
except ImportError:
    pass

def test_video_preprocessor_config(test_env):
    input_path, output_dir = test_env
    preprocessor = VideoPreprocessor()

    # Run preprocessing
    result = preprocessor.preprocess(str(input_path), str(output_dir))

    # 1. Verify Processed Video Properties
    cap = cv2.VideoCapture(result.video_path)

    # Verify Resolution (256x256)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert width == 256
    assert height == 256

    # Verify Frame Rate (8fps)
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert round(fps) == 8

    cap.release()

    # 2. Verify Frames Extraction
    extracted_frames = list(Path(result.frames_dir).glob("*.jpg"))
    assert len(extracted_frames) > 0
    assert result.frame_count == len(extracted_frames)

    # 3. Verify Audio Extraction
    assert os.path.exists(result.audio_path)
    # Note: Our dummy video didn't have audio,
    # so the preprocessor should have generated a silent WAV.
    import soundfile as sf
    data, samplerate = sf.read(result.audio_path)
    assert samplerate == 11025
    assert len(data) > 0

if __name__ == "__main__":
    # Manual run if pytest is not used
    class MockEnv:
        def __enter__(self):
            test_dir = Path("tests/temp_data")
            test_dir.mkdir(parents=True, exist_ok=True)
            input_path = test_dir / "dummy_input.mp4"
            output_dir = test_dir / "output"
            create_dummy_video(str(input_path))
            return input_path, output_dir
        def __exit__(self, *args):
            pass

    with MockEnv() as (ip, od):
        print("Running manual preprocessor test...")
        preprocessor = VideoPreprocessor()
        res = preprocessor.preprocess(str(ip), str(od))
        print(f"Success! Processed video: {res.video_path}")
        print(f"Frames: {res.frame_count}, Audio: {res.audio_path}")
