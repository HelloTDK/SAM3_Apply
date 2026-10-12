import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import BACKEND, RESULT_DIR, STATIC_DIR
from .routes import register_routes
from .runtime import MODEL_LOCK, predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        if BACKEND == "om":
            with MODEL_LOCK:
                predictor.close(finalize=True)


def create_app() -> FastAPI:
    app = FastAPI(
        title="SAM3 OpenAI-style Segmentation API",
        description="HTTP service for SAM3 image segmentation with API key authentication.",
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

    register_routes(app)
    return app


app = create_app()
