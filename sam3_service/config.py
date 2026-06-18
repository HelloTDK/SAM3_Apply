"""Shared configuration for the SAM3 service.

This module is intentionally lightweight: importing it must not load the SAM3
model.  Routes, auth and runtime code can all depend on these constants without
creating a model-loading side effect.
"""

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


def _resolve_default_checkpoint_path() -> Path:
    weights_dir = ROOT_DIR / "weights"
    candidates = [
        weights_dir / "sam3.pt",
        weights_dir / "sam3.1_multiplex.pt",
    ]
    for one_path in candidates:
        if one_path.exists():
            return one_path
    return candidates[0]


DEFAULT_CHECKPOINT_PATH = _resolve_default_checkpoint_path()
CHECKPOINT_PATH = Path(os.getenv("SAM3_CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT_PATH)))
MAX_IMAGE_BYTES = int(os.getenv("SAM3_MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))
SAVE_UPLOADS = _env_bool("SAM3_SAVE_UPLOADS", "1")
MAX_CONCURRENT_INFERENCES = max(1, int(os.getenv("SAM3_MAX_CONCURRENT_INFERENCES", "4")))
SERIALIZE_MODEL_ACCESS = _env_bool("SAM3_SERIALIZE_MODEL_ACCESS", "1")
EMPTY_CUDA_CACHE_EACH_REQUEST = _env_bool("SAM3_EMPTY_CUDA_CACHE_EACH_REQUEST", "0")
CUDA_CLEANUP_AFTER_REQUEST = _env_bool("SAM3_CUDA_CLEANUP_AFTER_REQUEST", "1")
IDLE_MODEL_UNLOAD_SECONDS = int(os.getenv("SAM3_IDLE_MODEL_UNLOAD_SECONDS", "0"))
CUDA_CLEANUP_LOG = _env_bool("SAM3_CUDA_CLEANUP_LOG", "0")
INFER_DTYPE_STR = (os.getenv("SAM3_INFER_DTYPE", "bfloat16") or "bfloat16").strip().lower()
ENABLE_AUTOCAST = _env_bool("SAM3_ENABLE_AUTOCAST", "1")
ULTRALYTICS_IMGSZ = int(os.getenv("SAM3_ULTRALYTICS_IMGSZ", "1036"))
ULTRALYTICS_IOU = float(os.getenv("SAM3_ULTRALYTICS_IOU", "0.7"))
ULTRALYTICS_VERBOSE = _env_bool("SAM3_ULTRALYTICS_VERBOSE", "0")
SIMILAR_MODES = {"feature_match", "same_image_prompt"}
MULTI_NEGATIVE_FILTER_IOU = float(os.getenv("SAM3_MULTI_NEGATIVE_FILTER_IOU", "0.5"))


def _infer_model_label(checkpoint_path: Path) -> str:
    lowered_name = checkpoint_path.name.lower()
    if "sam3.1" in lowered_name or "multiplex" in lowered_name:
        return "ultralytics-sam3.1"
    return "ultralytics-sam3"


MODEL_LABEL = (os.getenv("SAM3_MODEL_LABEL", "") or "").strip() or _infer_model_label(CHECKPOINT_PATH)

# Runtime outputs are written to these folders and static mounting checks them.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
