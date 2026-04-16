import asyncio
import base64
import logging
import os
import shutil
from uuid import uuid4

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from inference import inference
from video_preprocessor import VideoPreprocessor


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    try:
        inference.load_model()
    except Exception as e:
        logger.info(f"Failed to load model: {e}")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

preprocessor = VideoPreprocessor()

MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB


def remove_directory(path: str):
    """Helper to clean up temp files after request"""
    try:
        shutil.rmtree(path)
        logger.info(f"Cleaned up: {path}")
    except Exception as e:
        logger.info(f"Error cleaning up {path}: {e}")


def file_to_base64(file_path: str) -> str:
    """Reads a file and converts it to a base64 string"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@app.get("/")
def health_check():
    return {"status": "active", "model_loaded": inference.is_loaded}


@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    # 1. Setup Directories
    request_id = uuid4().hex
    base_dir = os.path.join("src", "outputs", request_id)
    results_dir = os.path.join(base_dir, "results")

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 2. Save Input Video (with size limit)
    filename = f"input_{request_id}.mp4"
    video_path = os.path.join(base_dir, filename)

    try:
        total_size = 0
        with open(video_path, "wb") as out_f:
            while chunk := await video.read(1024 * 1024):  # Read 1MB at a time
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024*1024)}MB"
                    )
                out_f.write(chunk)
    finally:
        await video.close()

    # 3. Preprocess (audio + frame extraction via VideoPreprocessor)
    # The video from mobile is already FFmpeg-processed, so skip FFmpeg step
    try:
        processed = await asyncio.to_thread(
            preprocessor.preprocess_already_processed, video_path, base_dir
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}")

    # 4. Run Inference
    try:
        paths = await asyncio.to_thread(
            inference.eval_single_heatmap,
            processed.audio_path,
            processed.frames_dir,
            processed.frame_count,
            results_dir,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # 5. Convert Results to Base64
    try:
        b64_audio_left = await asyncio.to_thread(file_to_base64, paths["audio_left"])
        b64_audio_right = await asyncio.to_thread(file_to_base64, paths["audio_right"])
        b64_image = await asyncio.to_thread(file_to_base64, paths["image_combined"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {e}")

    # 6. Cleanup Task (Happens after response is sent)
    background_tasks.add_task(remove_directory, base_dir)

    # 7. Return JSON
    return {
        "left_audio": b64_audio_left,  # Base64 string of WAV
        "right_audio": b64_audio_right,  # Base64 string of WAV
        "heatmap_image": b64_image,  # Base64 string of JPG
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
