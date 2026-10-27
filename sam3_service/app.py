import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import RESULT_DIR, STATIC_DIR
from .routes import register_routes
from .tcp_bridge import router as tcp_test_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .eiseg_tcp import InteractiveSegmentationServer
    from .trt_runtime import predictor

    address = (os.getenv("SAM3_TCP_HOST", "0.0.0.0"), int(os.getenv("SAM3_TCP_PORT", "6063")))
    # FastAPI owns both listeners and injects the already-loaded model into TCP.
    with InteractiveSegmentationServer(address, backend=predictor.interactive) as server:
        app.state.tcp_server = server
        thread = threading.Thread(target=server.start_server, daemon=True)
        thread.start()
        print(f"SAM3 TCP 服务已启动：{server.server_address}", flush=True)
        try:
            yield
        finally:
            server.shutdown()
            thread.join()


def create_app() -> FastAPI:
    app = FastAPI(
        title="SAM3 OpenAI-style Segmentation API",
        description="TensorRT HTTP service for SAM3 image segmentation with API key authentication.",
        version="2.0.0",
        lifespan=lifespan,
    )

    allow_origins_env = os.getenv("SAM3_ALLOW_ORIGINS", "*")
    allow_origins = [item.strip() for item in allow_origins_env.split(",") if item.strip()] or ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if RESULT_DIR.exists():
        app.mount("/results", StaticFiles(directory=str(RESULT_DIR)), name="results")
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    app.include_router(tcp_test_router)
    register_routes(app)
    return app


app = create_app()
