import os

from sam3_service.app import app
from sam3_service.eiseg_tcp import MAX_FRAME_BYTES


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("SAM3_HOST", "0.0.0.0")
    port = int(os.getenv("SAM3_PORT", "8006"))

    print("Starting SAM3 TensorRT HTTP + TCP Server...")
    print(f"Server URL: http://{host}:{port}")
    print(f"OpenAPI docs: http://{host}:{port}/docs")
    print(f"TCP: {os.getenv('SAM3_TCP_HOST', '0.0.0.0')}:{os.getenv('SAM3_TCP_PORT', '6063')}")

    # Keep one process so both transports use the same model and encoder context.
    # Match the TCP image limit; uvicorn's default 16 MiB is too small for Base64 uploads.
    uvicorn.run(app, host=host, port=port, ws_max_size=MAX_FRAME_BYTES)
