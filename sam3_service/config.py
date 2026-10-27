"""Shared configuration for the SAM3 service.

This module is intentionally lightweight: importing it must not load the SAM3
model.  Routes, auth and runtime code can all depend on these constants without
creating a model-loading side effect.
"""

import math
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
VENDOR_DIR = ROOT_DIR / "_vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

UPLOAD_DIR = ROOT_DIR / "uploads"
RESULT_DIR = ROOT_DIR / "results"
STATIC_DIR = ROOT_DIR / "static"
API_KEYS_FILE = Path(os.getenv("SAM3_API_KEYS_FILE", str(ROOT_DIR / "api_keys.json")))


def _env_bool(name: str, default: str) -> bool:
    """Parse common truthy environment variable values."""
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


# 默认只读取程序自身携带的 engine，避免部署时依赖仓库中的其他目录。
DEFAULT_ENGINE_DIR = ROOT_DIR / "engines"
TRT_ENGINE_DIR = Path(os.getenv("SAM3_TRT_ENGINE_DIR", str(DEFAULT_ENGINE_DIR)))
NATIVE_ENGINE_DIR = Path(os.getenv("SAM3_NATIVE_ENGINE_DIR", str(TRT_ENGINE_DIR / "native_interactive")))
MAX_IMAGE_BYTES = int(os.getenv("SAM3_MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))
TCP_POLYGON_EPSILON = float(os.getenv("SAM3_TCP_POLYGON_EPSILON", "2.0"))
if not math.isfinite(TCP_POLYGON_EPSILON) or not 0 <= TCP_POLYGON_EPSILON <= 50:
    raise ValueError("SAM3_TCP_POLYGON_EPSILON must be between 0 and 50 pixels")
SAVE_UPLOADS = _env_bool("SAM3_SAVE_UPLOADS", "1")
MAX_CONCURRENT_INFERENCES = max(1, int(os.getenv("SAM3_MAX_CONCURRENT_INFERENCES", "4")))
CUDA_CLEANUP_AFTER_REQUEST = _env_bool("SAM3_CUDA_CLEANUP_AFTER_REQUEST", "1")
TRT_NMS_IOU = float(os.getenv("SAM3_TRT_NMS_IOU", "0.7"))
VISUAL_PROMPT_MAX_CANDIDATES = max(1, int(os.getenv("SAM3_VISUAL_PROMPT_MAX_CANDIDATES", "32")))
SIMILAR_MODES = {"feature_match", "same_image_prompt"}


MODEL_LABEL = (os.getenv("SAM3_MODEL_LABEL", "") or "").strip() or "sam3-tensorrt"

# Runtime outputs are written to these folders and static mounting checks them.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
