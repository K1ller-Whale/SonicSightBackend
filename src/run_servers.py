import os
import sys
import asyncio
import threading
import uvicorn
import logging

# Ensure the 'src' directory is in the path so imports work when run from project root
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from grpc_server import serve as grpc_serve
from model_registry import load_all_engines


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_fastapi(host="0.0.0.0", port=8000):
    """Run FastAPI server in a separate thread."""
    from main import app

    uvicorn.run(app, host=host, port=port)


async def run_grpc(port=50051):
    """Run gRPC server as an async task."""
    await grpc_serve(port=port)


def main():
    # Load every registered model once (shared singletons). A model that
    # fails to load is skipped and advertised as absent via HealthCheck.
    load_all_engines()

    # Start FastAPI in a background thread
    fastapi_thread = threading.Thread(
        target=run_fastapi,
        kwargs={"host": "0.0.0.0", "port": 8000},
        daemon=True,
    )
    fastapi_thread.start()
    logger.info("FastAPI server started on port 8000")

    # Run gRPC server on the main asyncio loop
    logger.info("Starting gRPC server on port 50051...")
    asyncio.run(run_grpc(port=50051))


if __name__ == "__main__":
    main()
