import os
import sys
import asyncio
import grpc
import time

# Add src to path to import generated stubs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import sonicsight_pb2
import sonicsight_pb2_grpc

async def test_health_check(stub):
    print("Testing HealthCheck...")
    try:
        response = await stub.HealthCheck(sonicsight_pb2.Empty())
        print(f"HealthCheck response: model_loaded={response.model_loaded}, device={response.device}")
        return True
    except Exception as e:
        print(f"HealthCheck failed: {e}")
        return False

async def test_process_video(stub, video_path):
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}. Skipping ProcessVideo test.")
        return False

    print(f"Testing ProcessVideo with {video_path}...")

    def chunk_generator():
        chunk_size = 256 * 1024
        file_size = os.path.getsize(video_path)

        with open(video_path, "rb") as f:
            chunk_index = 0
            # First chunk with metadata
            data = f.read(chunk_size)
            is_last = f.tell() >= file_size

            yield sonicsight_pb2.VideoChunk(
                metadata=sonicsight_pb2.VideoMetadata(
                    total_size=file_size,
                    filename=os.path.basename(video_path),
                    mime_type="video/mp4"
                ),
                data=data,
                chunk_index=chunk_index,
                is_last=is_last
            )
            chunk_index += 1

            while not is_last:
                data = f.read(chunk_size)
                is_last = f.tell() >= file_size
                yield sonicsight_pb2.VideoChunk(
                    data=data,
                    chunk_index=chunk_index,
                    is_last=is_last
                )
                chunk_index += 1

    try:
        start_time = time.time()
        # gRPC streams usually take an async iterator
        async def request_iterator():
            for chunk in chunk_generator():
                yield chunk

        response = await stub.ProcessVideo(request_iterator())
        elapsed = time.time() - start_time

        if response.success:
            print(f"ProcessVideo success in {elapsed:.2f}s!")
            print(f"Received: {len(response.left_audio_pcm)} bytes left audio")
            print(f"Received: {len(response.left_heatmap)} bytes left heatmap")
            print(f"Processing time on server: {response.processing_time_ms}ms")
        else:
            print(f"ProcessVideo reported failure: {response.error_message}")
        return response.success
    except Exception as e:
        print(f"ProcessVideo RPC failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    server_address = 'localhost:50051'
    print(f"Connecting to {server_address}...")

    async with grpc.aio.insecure_channel(server_address) as channel:
        stub = sonicsight_pb2_grpc.SonicSightServiceStub(channel)

        # 1. Test Health
        health_ok = await test_health_check(stub)

        # 2. Test Video (if provided via args)
        if len(sys.argv) > 1:
            await test_process_video(stub, sys.argv[1])
        else:
            print("\nTo test video processing, run: python src/test_grpc_client.py <path_to_video.mp4>")

if __name__ == "__main__":
    asyncio.run(main())
