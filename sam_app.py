import os

from sam3_service.app import app


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("SAM3_HOST", "0.0.0.0")
    port = int(os.getenv("SAM3_PORT", "8006"))

    print("Starting SAM3 Detection Server...")
    print(f"Server URL: http://{host}:{port}")
    print(f"OpenAPI docs: http://{host}:{port}/docs")

    uvicorn.run(app, host=host, port=port)
