import asyncio
import base64
import binascii
import contextlib
import hashlib
import html
import io
import json
import os
import re
import secrets
import shutil
import threading
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cv2
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field, model_validator

try:
    import argostranslate.package as argos_package
    import argostranslate.translate as argos_translate

    ARGOS_AVAILABLE = True
except Exception:
    argos_package = None
    argos_translate = None
    ARGOS_AVAILABLE = False


ROOT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = ROOT_DIR / "_vendor"
if VENDOR_DIR.exists():
    import sys

    sys.path.insert(0, str(VENDOR_DIR))

UPLOAD_DIR = ROOT_DIR / "uploads"
RESULT_DIR = ROOT_DIR / "results"
STATIC_DIR = ROOT_DIR / "static"
API_KEYS_FILE = Path(os.getenv("SAM3_API_KEYS_FILE", str(ROOT_DIR / "api_keys.json")))


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
SAVE_UPLOADS = os.getenv("SAM3_SAVE_UPLOADS", "1").lower() in {"1", "true", "yes", "on"}
MAX_CONCURRENT_INFERENCES = max(1, int(os.getenv("SAM3_MAX_CONCURRENT_INFERENCES", "4")))
SERIALIZE_MODEL_ACCESS = os.getenv("SAM3_SERIALIZE_MODEL_ACCESS", "1").lower() in {"1", "true", "yes", "on"}
EMPTY_CUDA_CACHE_EACH_REQUEST = os.getenv("SAM3_EMPTY_CUDA_CACHE_EACH_REQUEST", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
INFER_DTYPE_STR = (os.getenv("SAM3_INFER_DTYPE", "bfloat16") or "bfloat16").strip().lower()
ENABLE_AUTOCAST = os.getenv("SAM3_ENABLE_AUTOCAST", "1").lower() in {"1", "true", "yes", "on"}
ULTRALYTICS_IMGSZ = int(os.getenv("SAM3_ULTRALYTICS_IMGSZ", "1036"))
ULTRALYTICS_IOU = float(os.getenv("SAM3_ULTRALYTICS_IOU", "0.7"))
ULTRALYTICS_VERBOSE = os.getenv("SAM3_ULTRALYTICS_VERBOSE", "0").lower() in {"1", "true", "yes", "on"}
SIMILAR_CANDIDATE_MULTIPLIER = max(1, int(os.getenv("SAM3_SIMILAR_CANDIDATE_MULTIPLIER", "2")))
MAX_SIMILAR_PROMPT_CANDIDATES = max(1, int(os.getenv("SAM3_MAX_SIMILAR_PROMPT_CANDIDATES", "12")))
SIMILAR_CANDIDATE_NMS_IOU = float(os.getenv("SAM3_SIMILAR_CANDIDATE_NMS_IOU", "0.25"))
SIMILAR_PEAK_NMS_KERNEL = max(1, int(os.getenv("SAM3_SIMILAR_PEAK_NMS_KERNEL", "5")))
SIMILAR_CANDIDATE_PREFILTER_MULTIPLIER = max(
    1,
    int(os.getenv("SAM3_SIMILAR_CANDIDATE_PREFILTER_MULTIPLIER", "8")),
)
CONCAT_PROMPT_SCALES = [
    float(item.strip())
    for item in os.getenv("SAM3_CONCAT_PROMPT_SCALES", "1.0").split(",")
    if item.strip()
]
CONCAT_PROMPT_PADDING = max(0, int(os.getenv("SAM3_CONCAT_PROMPT_PADDING", "16")))
CONCAT_PROMPT_SEPARATOR = max(0, int(os.getenv("SAM3_CONCAT_PROMPT_SEPARATOR", "16")))
SIMILAR_MODES = {"feature_match", "concat_prompt", "same_image_prompt"}


def _infer_model_label(checkpoint_path: Path) -> str:
    lowered_name = checkpoint_path.name.lower()
    if "sam3.1" in lowered_name or "multiplex" in lowered_name:
        return "ultralytics-sam3.1"
    return "ultralytics-sam3"


MODEL_LABEL = (os.getenv("SAM3_MODEL_LABEL", "") or "").strip() or _infer_model_label(CHECKPOINT_PATH)


def _resolve_infer_dtype(device_name: str, requested: str) -> torch.dtype:
    normalized = requested.strip().lower()
    alias = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    kind = alias.get(normalized)
    if kind is None:
        raise ValueError(
            f"Unsupported SAM3_INFER_DTYPE='{requested}'. "
            "Use one of: bfloat16, float16, float32."
        )

    is_cuda = "cuda" in device_name and torch.cuda.is_available()
    if kind == "bfloat16":
        if is_cuda:
            return torch.bfloat16
        print("bfloat16 requested on non-CUDA device; fallback to float32")
        return torch.float32
    if kind == "float16":
        if is_cuda:
            return torch.float16
        print("float16 requested on non-CUDA device; fallback to float32")
        return torch.float32
    return torch.float32


def _inference_autocast_context():
    use_autocast = (
        ENABLE_AUTOCAST
        and torch.cuda.is_available()
        and "cuda" in device
        and MODEL_DTYPE in {torch.bfloat16, torch.float16}
    )
    if not use_autocast:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=MODEL_DTYPE)


def _patch_openai_clip_tokenizer() -> None:
    """Make PyPI openai-clip compatible with Ultralytics SAM3's tokenizer call shape."""
    try:
        import clip
    except Exception:
        return

    tokenizer_module = getattr(clip, "simple_tokenizer", None)
    tokenizer_cls = getattr(tokenizer_module, "SimpleTokenizer", None)
    if tokenizer_cls is None or callable(tokenizer_cls()):
        return

    def _call(self, texts: Any, context_length: int = 77) -> torch.Tensor:
        return clip.tokenize(texts, context_length=context_length, truncate=True)

    tokenizer_cls.__call__ = _call

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def setup_chinese_font() -> str:
    """Configure matplotlib for Chinese labels."""
    chinese_fonts = [
        "GB_KT_GB18030",
        "Noto Sans CJK JP",
        "Noto Serif CJK JP",
        "GB_SS_GB18030",
        "GB_HT_GB18030",
        "SimHei",
        "Microsoft YaHei",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]

    available_fonts = [f.name for f in fm.fontManager.ttflist]

    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams["font.sans-serif"] = [font] + plt.rcParams["font.sans-serif"]
            print(f"Using font: {font}")
            break

    plt.rcParams["axes.unicode_minus"] = False
    return plt.rcParams["font.sans-serif"][0]


CURRENT_FONT = setup_chinese_font()

app = FastAPI(
    title="SAM3 OpenAI-style Segmentation API",
    description="HTTP service for SAM3 image segmentation with API key authentication.",
    version="2.0.0",
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


device = os.getenv("SAM3_DEVICE", "") or ("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ.setdefault("ARGOS_DEVICE_TYPE", "cuda" if "cuda" in device else "cpu")
MODEL_DTYPE = _resolve_infer_dtype(device, INFER_DTYPE_STR)

print("Loading SAM3 model...")
print(f"Using device: {device}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Model label: {MODEL_LABEL}")
print(f"Inference dtype: {MODEL_DTYPE}")
print(f"Autocast enabled: {ENABLE_AUTOCAST}")
print(f"Ultralytics image size: {ULTRALYTICS_IMGSZ}")
print(f"Ultralytics NMS IoU: {ULTRALYTICS_IOU}")

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")

try:
    from ultralytics.models.sam import SAM3SemanticPredictor
except Exception as exc:
    raise RuntimeError(
        "Ultralytics with SAM3 support is required. Install it in the sam3 conda environment, "
        "for example: python -m pip install -U 'ultralytics>=8.4.51'"
    ) from exc

_patch_openai_clip_tokenizer()

predictor = SAM3SemanticPredictor(
    overrides={
        "model": str(CHECKPOINT_PATH),
        "device": device,
        "conf": 0.01,
        "iou": ULTRALYTICS_IOU,
        "imgsz": ULTRALYTICS_IMGSZ,
        "half": MODEL_DTYPE == torch.float16,
        "verbose": ULTRALYTICS_VERBOSE,
        "save": False,
    }
)
predictor.setup_model()

print("Model loaded successfully")

translation_available = False
ARGOS_SOURCE_CODE = (os.getenv("SAM3_ARGOS_SOURCE_CODE", "zh") or "zh").strip()
ARGOS_TARGET_CODE = (os.getenv("SAM3_ARGOS_TARGET_CODE", "en") or "en").strip()
ARGOS_TRANSLATOR: Optional[Any] = None
ARGOS_AUTO_INSTALL = os.getenv("SAM3_ARGOS_AUTO_INSTALL", "1").lower() in {"1", "true", "yes", "on"}
ARGOS_MODEL_PATH = (os.getenv("SAM3_ARGOS_MODEL_PATH", "") or "").strip()
ARGOS_MODEL_DIR = (os.getenv("SAM3_ARGOS_MODEL_DIR", "") or "").strip()
ARGOS_FORCE_OFFLINE = os.getenv("SAM3_ARGOS_FORCE_OFFLINE", "0").lower() in {"1", "true", "yes", "on"}
ARGOS_INSTALL_LOCK = threading.Lock()
ARGOS_LOCAL_INSTALL_ATTEMPTED = False
ARGOS_ONLINE_INSTALL_ATTEMPTED = False
ARGOS_TRANSLATION_ERROR_LOGGED = False
ARGOS_OFFLINE_LOCK = threading.Lock()
ARGOS_OFFLINE_TRANSLATOR: Optional[Any] = None
ARGOS_OFFLINE_PACKAGE_PATH: Optional[str] = None
ARGOS_CLEAN_BROKEN_PACKAGES = os.getenv("SAM3_ARGOS_CLEAN_BROKEN_PACKAGES", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _normalize_lang_code(code: str) -> str:
    return (code or "").replace("-", "_").lower()


def _normalize_prompt_label(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text or "").strip()
    return collapsed.strip(" \t\r\n,;:，；。.!?！？\"'`")


def _lang_matches(code: str, expected_code: str) -> bool:
    normalized = _normalize_lang_code(code)
    expected = _normalize_lang_code(expected_code)
    return normalized == expected or normalized.startswith(f"{expected}_")


def _pick_installed_language(installed_languages: List[Any], expected_code: str) -> Optional[Any]:
    for one_language in installed_languages:
        language_code = getattr(one_language, "code", "")
        if _lang_matches(language_code, expected_code):
            return one_language
    return None


def _pick_available_package(available_packages: List[Any], from_code: str, to_code: str) -> Optional[Any]:
    for one_package in available_packages:
        package_from = getattr(one_package, "from_code", "")
        package_to = getattr(one_package, "to_code", "")
        if _lang_matches(package_from, from_code) and _lang_matches(package_to, to_code):
            return one_package
    return None


def _pick_installed_package(from_code: str, to_code: str) -> Optional[Any]:
    if not ARGOS_AVAILABLE:
        return None
    try:
        installed_packages = argos_package.get_installed_packages()
    except Exception:
        return None
    for one_package in installed_packages:
        package_from = getattr(one_package, "from_code", "")
        package_to = getattr(one_package, "to_code", "")
        if _lang_matches(package_from, from_code) and _lang_matches(package_to, to_code):
            return one_package
    return None


def _discover_argos_package_roots() -> List[Path]:
    roots: List[Path] = []
    env_keys = ["ARGOS_PACKAGE_DIR", "ARGOS_PACKAGES_DIR", "ARGOS_TRANSLATE_PACKAGES_DIR"]
    for one_key in env_keys:
        one_value = (os.getenv(one_key, "") or "").strip()
        if one_value:
            roots.append(Path(one_value))

    roots.append(Path.home() / ".local" / "share" / "argos-translate" / "packages")

    unique_roots: List[Path] = []
    seen: set = set()
    for one_root in roots:
        normalized = str(one_root)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_roots.append(one_root)
    return unique_roots


def _sanitize_argos_package_dirs() -> int:
    if not ARGOS_CLEAN_BROKEN_PACKAGES:
        return 0

    fixed = 0
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for root in _discover_argos_package_roots():
        if not root.exists() or not root.is_dir():
            continue
        for one_child in sorted(root.iterdir()):
            if not one_child.is_dir():
                continue
            if ".broken_" in one_child.name:
                continue

            metadata_path = one_child / "metadata.json"
            if metadata_path.exists():
                continue

            quarantine = one_child.with_name(f"{one_child.name}.broken_{ts}")
            suffix_index = 1
            while quarantine.exists():
                quarantine = one_child.with_name(f"{one_child.name}.broken_{ts}_{suffix_index}")
                suffix_index += 1
            try:
                shutil.move(str(one_child), str(quarantine))
                fixed += 1
                print(
                    "Sanitized broken Argos package directory "
                    f"'{one_child}' -> '{quarantine}' (missing metadata.json)"
                )
            except Exception as exc:
                print(f"Failed to sanitize broken Argos package directory '{one_child}': {exc}")
    return fixed


def _configure_stanza_resources_for_offline() -> None:
    # Argos package may include stanza resources under package_path/stanza.
    # Point Stanza there to avoid network downloads in offline environments.
    one_package = _pick_installed_package(ARGOS_SOURCE_CODE, ARGOS_TARGET_CODE)
    if one_package is None:
        return

    stanza_dir = Path(getattr(one_package, "package_path", "")) / "stanza"
    resources_json = stanza_dir / "resources.json"
    if not stanza_dir.exists() or not resources_json.exists():
        return

    resolved = str(stanza_dir)
    os.environ.setdefault("STANZA_RESOURCES_DIR", resolved)
    try:
        import stanza.resources.common as stanza_common  # type: ignore

        stanza_common.DEFAULT_MODEL_DIR = resolved
    except Exception:
        pass

    print(f"Configured STANZA_RESOURCES_DIR for offline use: {resolved}")


def _discover_local_argos_models() -> List[Path]:
    candidates: List[Path] = []

    if ARGOS_MODEL_PATH:
        candidates.append(Path(ARGOS_MODEL_PATH))

    candidate_dirs: List[Path] = []
    if ARGOS_MODEL_DIR:
        candidate_dirs.append(Path(ARGOS_MODEL_DIR))
    candidate_dirs.extend(
        [
            ROOT_DIR / "argos_models",
            ROOT_DIR / "models",
            ROOT_DIR,
        ]
    )

    for one_dir in candidate_dirs:
        if not one_dir.exists() or not one_dir.is_dir():
            continue
        for one_file in sorted(one_dir.glob("*.argosmodel")):
            candidates.append(one_file)

    unique_candidates: List[Path] = []
    seen: set = set()
    for one_path in candidates:
        normalized = str(one_path)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_candidates.append(one_path)

    return unique_candidates


def _install_argos_from_local_models() -> bool:
    local_models = _discover_local_argos_models()
    if local_models:
        print(f"Discovered local Argos model files: {[str(one_model) for one_model in local_models]}")
    else:
        print("No local Argos model file discovered (*.argosmodel)")

    installed_any = False
    for model_path in local_models:
        if not model_path.exists() or not model_path.is_file():
            continue
        try:
            print(f"Installing Argos model from local path: {model_path}")
            argos_package.install_from_path(str(model_path))
            print(f"Argos model installed from local path: {model_path}")
            installed_any = True
        except Exception as exc:
            print(f"Failed to install local Argos model {model_path}: {exc}")

    return installed_any


def _load_argos_translator() -> Optional[Any]:
    if not ARGOS_AVAILABLE:
        return None

    try:
        installed_languages = argos_translate.get_installed_languages()
        source_language = _pick_installed_language(installed_languages, ARGOS_SOURCE_CODE)
        target_language = _pick_installed_language(installed_languages, ARGOS_TARGET_CODE)
        if source_language is None or target_language is None:
            return None
        return source_language.get_translation(target_language)
    except Exception as exc:
        print(f"Failed to initialize argostranslate translator: {exc}")
        return None


def _translate_with_argos_model_offline(text: str) -> Optional[str]:
    global ARGOS_OFFLINE_TRANSLATOR, ARGOS_OFFLINE_PACKAGE_PATH

    if not ARGOS_AVAILABLE:
        return None

    with ARGOS_OFFLINE_LOCK:
        one_package = _pick_installed_package(ARGOS_SOURCE_CODE, ARGOS_TARGET_CODE)
        if one_package is None:
            return None

        package_path = str(getattr(one_package, "package_path", ""))
        if not package_path:
            return None

        try:
            import ctranslate2
            from argostranslate import settings as argos_settings
        except Exception:
            return None

        if ARGOS_OFFLINE_TRANSLATOR is None or ARGOS_OFFLINE_PACKAGE_PATH != package_path:
            model_path = str(Path(package_path) / "model")
            ARGOS_OFFLINE_TRANSLATOR = ctranslate2.Translator(
                model_path,
                device=argos_settings.device,
                inter_threads=argos_settings.inter_threads,
                intra_threads=argos_settings.intra_threads,
                compute_type=argos_settings.compute_type,
            )
            ARGOS_OFFLINE_PACKAGE_PATH = package_path

        tokenized = [one_package.tokenizer.encode(text)]
        target_prefix = None
        if getattr(one_package, "target_prefix", ""):
            target_prefix = [[one_package.target_prefix]]

        translated_batches = ARGOS_OFFLINE_TRANSLATOR.translate_batch(
            tokenized,
            target_prefix=target_prefix,
            replace_unknowns=True,
            max_batch_size=1,
            batch_type="tokens",
            beam_size=1,
            num_hypotheses=1,
            return_scores=False,
        )

        if not translated_batches:
            return None

        translated_tokens = translated_batches[0].hypotheses[0]
        value = one_package.tokenizer.decode(translated_tokens)

        if getattr(one_package, "target_prefix", "") and value.startswith(one_package.target_prefix):
            value = value[len(one_package.target_prefix) :]
        if value.startswith(" "):
            value = value[1:]
        return value


def _log_argos_state() -> None:
    if not ARGOS_AVAILABLE:
        return
    try:
        installed_languages = argos_translate.get_installed_languages()
        language_codes = [getattr(one_language, "code", "") for one_language in installed_languages]
        print(f"Argos installed languages: {language_codes}")
    except Exception as exc:
        print(f"Failed to read Argos installed languages: {exc}")

    try:
        installed_packages = argos_package.get_installed_packages()
        package_pairs = [
            f"{getattr(one_package, 'from_code', '')}->{getattr(one_package, 'to_code', '')}"
            for one_package in installed_packages
        ]
        print(f"Argos installed packages: {package_pairs}")
    except Exception as exc:
        print(f"Failed to read Argos installed packages: {exc}")


def _ensure_argos_package_installed() -> bool:
    global ARGOS_LOCAL_INSTALL_ATTEMPTED, ARGOS_ONLINE_INSTALL_ATTEMPTED

    if not ARGOS_AVAILABLE:
        return False

    with ARGOS_INSTALL_LOCK:
        if not ARGOS_LOCAL_INSTALL_ATTEMPTED:
            ARGOS_LOCAL_INSTALL_ATTEMPTED = True
            if _install_argos_from_local_models():
                return True

        if not ARGOS_AUTO_INSTALL:
            return False

        if ARGOS_ONLINE_INSTALL_ATTEMPTED:
            return False
        ARGOS_ONLINE_INSTALL_ATTEMPTED = True

        try:
            print(f"Attempting to auto-install Argos package {ARGOS_SOURCE_CODE}->{ARGOS_TARGET_CODE} from network ...")
            argos_package.update_package_index()
            available_packages = argos_package.get_available_packages()
            package_to_install = _pick_available_package(
                available_packages,
                from_code=ARGOS_SOURCE_CODE,
                to_code=ARGOS_TARGET_CODE,
            )
            if package_to_install is None:
                print(f"No Argos package found for {ARGOS_SOURCE_CODE}->{ARGOS_TARGET_CODE}")
                return False

            package_path = package_to_install.download()
            argos_package.install_from_path(package_path)
            print(f"Argos package {ARGOS_SOURCE_CODE}->{ARGOS_TARGET_CODE} installed from network")
            return True
        except Exception as exc:
            print(f"Failed to auto-install Argos package {ARGOS_SOURCE_CODE}->{ARGOS_TARGET_CODE} from network: {exc}")
            return False


if ARGOS_AVAILABLE:
    _sanitize_argos_package_dirs()
    _log_argos_state()
    _configure_stanza_resources_for_offline()
    ARGOS_TRANSLATOR = _load_argos_translator()
    translation_available = ARGOS_TRANSLATOR is not None
    if translation_available:
        print(f"Translation package {ARGOS_SOURCE_CODE}->{ARGOS_TARGET_CODE} is ready")
    else:
        if _ensure_argos_package_installed():
            _log_argos_state()
            ARGOS_TRANSLATOR = _load_argos_translator()
            translation_available = ARGOS_TRANSLATOR is not None

    if translation_available:
        print(f"Translation package {ARGOS_SOURCE_CODE}->{ARGOS_TARGET_CODE} is ready")
    else:
        print(
            f"Translation package {ARGOS_SOURCE_CODE}->{ARGOS_TARGET_CODE} not ready at startup; "
            "will retry during requests"
        )
        if ARGOS_AUTO_INSTALL:
            print("Argos auto-install is enabled")
        if ARGOS_MODEL_PATH:
            print(f"SAM3_ARGOS_MODEL_PATH configured: {ARGOS_MODEL_PATH}")
        if ARGOS_MODEL_DIR:
            print(f"SAM3_ARGOS_MODEL_DIR configured: {ARGOS_MODEL_DIR}")
    if ARGOS_FORCE_OFFLINE:
        print("SAM3_ARGOS_FORCE_OFFLINE is enabled")
else:
    print("argostranslate is not installed, Chinese prompt translation disabled")


class ApiKeyManager:
    """Manage API keys in a local JSON file using hashed storage."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._lock = threading.RLock()
        self._store = self._load_store()

    @staticmethod
    def _hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _load_store(self) -> Dict[str, Any]:
        if not self.file_path.exists():
            return {"version": 1, "keys": []}

        try:
            with self.file_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            if "keys" not in data or not isinstance(data["keys"], list):
                return {"version": 1, "keys": []}
            return data
        except Exception:
            return {"version": 1, "keys": []}

    def _save_store(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.file_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as fp:
            json.dump(self._store, fp, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.file_path)

    def _sanitize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": record["id"],
            "name": record["name"],
            "role": record["role"],
            "enabled": record["enabled"],
            "created_at": record["created_at"],
            "expires_at": record.get("expires_at"),
            "key_prefix": record.get("key_prefix"),
        }

    def has_admin(self) -> bool:
        with self._lock:
            return any(k.get("role") == "admin" and k.get("enabled", True) for k in self._store["keys"])

    def upsert_admin_key(self, raw_key: str, name: str = "env-admin") -> None:
        key_hash = self._hash_key(raw_key)
        with self._lock:
            existing = next((k for k in self._store["keys"] if k["key_hash"] == key_hash), None)
            if existing:
                existing["enabled"] = True
                existing["role"] = "admin"
                existing["name"] = name
            else:
                record = {
                    "id": f"key_{uuid.uuid4().hex[:12]}",
                    "name": name,
                    "role": "admin",
                    "key_hash": key_hash,
                    "key_prefix": f"{raw_key[:8]}...{raw_key[-4:]}",
                    "enabled": True,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "expires_at": None,
                }
                self._store["keys"].append(record)
            self._save_store()

    def create_key(
        self,
        name: str,
        role: Literal["client", "admin"] = "client",
        expires_in_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        raw_key = f"sam3_{secrets.token_urlsafe(32)}"
        now = datetime.now(timezone.utc)
        expires_at = None
        if expires_in_days:
            expires_at = (now + timedelta(days=expires_in_days)).isoformat()

        record = {
            "id": f"key_{uuid.uuid4().hex[:12]}",
            "name": name,
            "role": role,
            "key_hash": self._hash_key(raw_key),
            "key_prefix": f"{raw_key[:8]}...{raw_key[-4:]}",
            "enabled": True,
            "created_at": now.isoformat(),
            "expires_at": expires_at,
        }

        with self._lock:
            self._store["keys"].append(record)
            self._save_store()

        return {
            **self._sanitize_record(record),
            "api_key": raw_key,
        }

    def list_keys(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._sanitize_record(record) for record in self._store["keys"]]

    def delete_key(self, key_id: str, protect_last_admin: bool = True) -> bool:
        with self._lock:
            index = next((idx for idx, item in enumerate(self._store["keys"]) if item["id"] == key_id), None)
            if index is None:
                return False

            record = self._store["keys"][index]
            if protect_last_admin and record.get("role") == "admin" and record.get("enabled", True):
                other_enabled_admin = any(
                    item.get("id") != key_id and item.get("role") == "admin" and item.get("enabled", True)
                    for item in self._store["keys"]
                )
                if not other_enabled_admin:
                    raise ValueError("Cannot delete the last enabled admin key")

            self._store["keys"].pop(index)
            self._save_store()
            return True

    def validate_key(self, raw_key: str) -> Optional[Dict[str, Any]]:
        key_hash = self._hash_key(raw_key)

        with self._lock:
            record = next(
                (
                    item
                    for item in self._store["keys"]
                    if item.get("enabled", True) and item.get("key_hash") == key_hash
                ),
                None,
            )

        if not record:
            return None

        expires_at = record.get("expires_at")
        if expires_at:
            try:
                expires_at_dt = datetime.fromisoformat(expires_at)
                if datetime.now(timezone.utc) > expires_at_dt:
                    return None
            except Exception:
                return None

        return self._sanitize_record(record)


api_key_manager = ApiKeyManager(API_KEYS_FILE)


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]+", text or ""))


def translate_to_english(text: str) -> str:
    normalized_text = _normalize_prompt_label(text)
    if not contains_chinese(normalized_text):
        return normalized_text or text

    if not ARGOS_AVAILABLE:
        return normalized_text

    global ARGOS_TRANSLATOR, translation_available, ARGOS_TRANSLATION_ERROR_LOGGED

    try:
        if ARGOS_FORCE_OFFLINE:
            translated = _translate_with_argos_model_offline(normalized_text)
            if not translated:
                return normalized_text
        else:
            if ARGOS_TRANSLATOR is None:
                # Lazy retry avoids a startup-time false negative permanently disabling translation.
                ARGOS_TRANSLATOR = _load_argos_translator()

            if ARGOS_TRANSLATOR is None and _ensure_argos_package_installed():
                ARGOS_TRANSLATOR = _load_argos_translator()

            if ARGOS_TRANSLATOR is not None:
                translation_available = True
                translated = ARGOS_TRANSLATOR.translate(normalized_text)
            else:
                translated = argos_translate.translate(normalized_text, ARGOS_SOURCE_CODE, ARGOS_TARGET_CODE)

        translated = _normalize_prompt_label(translated).lower()
        if not translated or contains_chinese(translated):
            return normalized_text
        return translated
    except Exception as exc:
        translated_offline = _translate_with_argos_model_offline(normalized_text)
        if translated_offline:
            translated = _normalize_prompt_label(translated_offline).lower()
            if translated and not contains_chinese(translated):
                return translated

        if not ARGOS_TRANSLATION_ERROR_LOGGED:
            print(f"argostranslate failed for '{normalized_text}', using raw text: {exc}")
            ARGOS_TRANSLATION_ERROR_LOGGED = True
        return normalized_text


def split_prompt_classes(prompt_text: str) -> List[str]:
    if not prompt_text:
        return []
    return [item.strip() for item in re.split(r"[;；,，]+", prompt_text) if item.strip()]


def prepare_single_text_prompt(prompt_text: Optional[str]) -> Tuple[Optional[List[str]], Optional[str], Optional[str], bool]:
    """Prepare one SAM3 text prompt. SAM3 box+text grounding uses a single prompt id."""
    original_prompt = _normalize_prompt_label(prompt_text or "")
    if not original_prompt:
        return None, None, None, False

    translated_prompt = translate_to_english(original_prompt)
    translated_prompt = _normalize_prompt_label(translated_prompt).lower() if translated_prompt else original_prompt
    if not translated_prompt:
        translated_prompt = original_prompt

    return [translated_prompt], original_prompt, translated_prompt, translated_prompt != original_prompt


def to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        # NumPy has no native bfloat16 dtype in many runtime stacks.
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        return tensor.numpy()
    return np.asarray(value)


def save_upload_image(image: Image.Image, original_name: str) -> str:
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", original_name or "input.jpg")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"upload_{timestamp}_{safe_name}"
    image.save(UPLOAD_DIR / filename)
    return filename


def decode_base64_image(image_base64: str) -> Image.Image:
    if not image_base64:
        raise ValueError("image_base64 is required")

    raw_data = image_base64.strip()
    if raw_data.startswith("data:image") and "," in raw_data:
        raw_data = raw_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw_data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image data") from exc

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image payload too large. Max bytes: {MAX_IMAGE_BYTES}")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("Decoded base64 is not a valid image") from exc

    return image


def build_class_color(index: int) -> np.ndarray:
    hue = (index * 137.5) % 360
    if hue < 60:
        rgb = (255, int(hue * 4.25), 0)
    elif hue < 120:
        rgb = (int(255 - (hue - 60) * 4.25), 255, 0)
    elif hue < 180:
        rgb = (0, 255, int((hue - 120) * 4.25))
    elif hue < 240:
        rgb = (0, int(255 - (hue - 180) * 4.25), 255)
    elif hue < 300:
        rgb = (int((hue - 240) * 4.25), 0, 255)
    else:
        rgb = (255, 0, int(255 - (hue - 300) * 4.25))
    return np.array(rgb)


def mask_to_polygons(mask: np.ndarray, epsilon: float = 2.0, min_area: float = 10.0) -> List[Dict[str, Any]]:
    mask_arr = np.asarray(mask)
    if mask_arr.ndim > 2:
        mask_arr = np.squeeze(mask_arr)

    if mask_arr.ndim != 2:
        return []

    binary = (mask_arr > 0.5).astype(np.uint8)
    if binary.max() == 0:
        return []

    # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons: List[Dict[str, Any]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        if epsilon > 0:
            contour = cv2.approxPolyDP(contour, epsilon, closed=True)

        if contour.shape[0] < 3:
            continue

        points = [[round(float(p[0][0]), 3), round(float(p[0][1]), 3)] for p in contour]
        polygons.append(
            {
                "area": float(area),
                "points": points,
            }
        )

    polygons.sort(key=lambda item: item["area"], reverse=True)

    return polygons


def bbox_to_xywh(box: np.ndarray) -> List[float]:
    x1, y1, x2, y2 = [float(x) for x in box]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return [
        round(x1, 3),
        round(y1, 3),
        round(width, 3),
        round(height, 3),
    ]


def bbox_xywh_to_xyxy(bnd_points: List[float]) -> List[float]:
    x, y, w, h = [float(v) for v in bnd_points]
    return [
        round(x, 3),
        round(y, 3),
        round(x + w, 3),
        round(y + h, 3),
    ]


def clip_xyxy_to_image(box_xyxy: List[float], image_width: int, image_height: int) -> List[float]:
    if len(box_xyxy) != 4:
        raise ValueError("box_xyxy must have exactly 4 values: [x1, y1, x2, y2]")
    x1 = _safe_float_from_any(box_xyxy[0], "box_xyxy[0]")
    y1 = _safe_float_from_any(box_xyxy[1], "box_xyxy[1]")
    x2 = _safe_float_from_any(box_xyxy[2], "box_xyxy[2]")
    y2 = _safe_float_from_any(box_xyxy[3], "box_xyxy[3]")
    x1 = max(0.0, min(x1, image_width * 1.0))
    y1 = max(0.0, min(y1, image_height * 1.0))
    x2 = max(0.0, min(x2, image_width * 1.0))
    y2 = max(0.0, min(y2, image_height * 1.0))
    if x2 <= x1:
        x2 = min(image_width * 1.0, x1 + 1.0)
    if y2 <= y1:
        y2 = min(image_height * 1.0, y1 + 1.0)
    return [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]


def bbox_iou_xywh(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, aw, ah = [float(v) for v in box_a]
    bx1, by1, bw, bh = [float(v) for v in box_b]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def bbox_xywh_to_polygon_points(bnd_points: List[float]) -> List[List[float]]:
    x, y, w, h = bnd_points
    return [
        [round(x, 3), round(y, 3)],
        [round(x + w, 3), round(y, 3)],
        [round(x + w, 3), round(y + h, 3)],
        [round(x, 3), round(y + h, 3)],
    ]


def clip_bnd_points_to_image(bnd_points: List[float], image_width: int, image_height: int) -> List[float]:
    if len(bnd_points) != 4:
        raise ValueError("bnd_points must have exactly 4 values: [x, y, w, h]")

    x = _safe_float_from_any(bnd_points[0], "bnd_points[0]")
    y = _safe_float_from_any(bnd_points[1], "bnd_points[1]")
    w = _safe_float_from_any(bnd_points[2], "bnd_points[2]")
    h = _safe_float_from_any(bnd_points[3], "bnd_points[3]")

    if w <= 0 or h <= 0:
        raise ValueError("bnd_points width and height must be > 0")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Invalid image size")

    x = max(0.0, min(x, image_width - 1.0))
    y = max(0.0, min(y, image_height - 1.0))
    w = max(1.0, min(w, image_width - x))
    h = max(1.0, min(h, image_height - y))
    return [round(x, 3), round(y, 3), round(w, 3), round(h, 3)]


def parse_optional_bnd_points_text(raw_value: Optional[str]) -> Optional[List[float]]:
    if raw_value is None:
        return None
    normalized = raw_value.strip()
    if not normalized:
        return None

    parts = [one.strip() for one in re.split(r"[,\s]+", normalized) if one.strip()]
    if len(parts) != 4:
        raise ValueError("reference_bnd_points must be 4 numbers: x,y,w,h")

    try:
        return [float(one) for one in parts]
    except ValueError as exc:
        raise ValueError("reference_bnd_points must contain numeric values") from exc


def _safe_float_from_any(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc


def visualize_results(image: Image.Image, all_detections: List[Dict[str, Any]]) -> str:
    img_array = np.array(image)
    overlay = img_array.copy().astype(np.float32)

    for detection in all_detections:
        masks = to_numpy(detection["masks"])
        if masks.ndim == 2:
            masks = masks[None, ...]

        class_color = detection["color"].astype(np.float32)
        for mask in masks:
            if mask.ndim > 2:
                mask = np.squeeze(mask)
            mask_bool = mask > 0.5
            overlay[mask_bool] = overlay[mask_bool] * 0.5 + class_color * 0.5

    overlay = overlay.astype(np.uint8)

    fig, ax = plt.subplots(1, figsize=(12, 8), dpi=100)
    ax.imshow(overlay)

    for detection in all_detections:
        boxes = to_numpy(detection["boxes"])
        scores = to_numpy(detection["scores"])
        class_name = detection.get("original_class_name", detection["class_name"])
        color_norm = detection["color"] / 255.0

        if boxes.ndim == 1 and boxes.size == 4:
            boxes = boxes[None, ...]
        if scores.ndim == 0:
            scores = np.array([float(scores)])

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = [float(v) for v in box]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color_norm, facecolor="none")
            ax.add_patch(rect)
            ax.text(
                x1,
                max(0.0, y1 - 5),
                f"{class_name} {float(score):.2f}",
                bbox={"facecolor": color_norm, "alpha": 0.7, "edgecolor": "white", "linewidth": 1},
                fontsize=9,
                color="white",
                weight="bold",
                fontfamily="sans-serif",
            )

    ax.axis("off")
    plt.tight_layout(pad=0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result_filename = f"result_{timestamp}.jpg"
    result_path = RESULT_DIR / result_filename

    plt.savefig(result_path, dpi=150, bbox_inches="tight", format="jpg")
    plt.close(fig)

    return result_filename


def pil_to_bgr_numpy(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def run_ultralytics_prediction(
    image: Image.Image,
    *,
    text: Optional[List[str]] = None,
    bboxes: Optional[List[List[float]]] = None,
    confidence_threshold: float = 0.3,
    reset_cached_image: bool = True,
) -> Any:
    np_image = pil_to_bgr_numpy(image)
    if reset_cached_image and hasattr(predictor, "reset_image"):
        predictor.reset_image()
    predictor.args.conf = confidence_threshold
    predictor.args.iou = ULTRALYTICS_IOU
    kwargs: Dict[str, Any] = {
        "source": np_image,
        "stream": False,
    }
    if text is not None:
        kwargs["text"] = text
    if bboxes is not None:
        kwargs["bboxes"] = bboxes
        kwargs["labels"] = [1] * len(bboxes)
    results = predictor(**kwargs)
    return results[0] if results else None


def extract_ultralytics_arrays(result: Any) -> Dict[str, np.ndarray]:
    empty_masks = np.zeros((0, 0, 0), dtype=bool)
    empty_boxes = np.zeros((0, 4), dtype=np.float32)
    empty_scores = np.zeros((0,), dtype=np.float32)
    empty_classes = np.zeros((0,), dtype=np.int64)

    if result is None or getattr(result, "masks", None) is None or getattr(result, "boxes", None) is None:
        return {
            "masks": empty_masks,
            "boxes": empty_boxes,
            "scores": empty_scores,
            "classes": empty_classes,
        }

    masks_obj = result.masks
    boxes_obj = result.boxes
    masks_np = to_numpy(masks_obj.data)
    boxes_np = to_numpy(boxes_obj.xyxy)
    scores_np = to_numpy(boxes_obj.conf)
    classes_np = to_numpy(boxes_obj.cls).astype(np.int64)

    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    if boxes_np.ndim == 1 and boxes_np.size == 4:
        boxes_np = boxes_np[None, ...]
    if scores_np.ndim == 0:
        scores_np = np.array([float(scores_np)])
    if classes_np.ndim == 0 and classes_np.size > 0:
        classes_np = np.array([int(classes_np)])

    return {
        "masks": masks_np,
        "boxes": boxes_np,
        "scores": scores_np,
        "classes": classes_np,
    }


def set_ultralytics_image_and_features(image: Image.Image) -> torch.Tensor:
    predictor.set_image(pil_to_bgr_numpy(image))
    features = predictor.features
    if not isinstance(features, dict):
        raise ValueError("Ultralytics SAM3 did not return dictionary features")
    backbone_fpn = features.get("backbone_fpn")
    if not isinstance(backbone_fpn, list) or not backbone_fpn:
        raise ValueError("Ultralytics SAM3 features missing backbone_fpn")
    feature_map = backbone_fpn[-1]
    if not torch.is_tensor(feature_map) or feature_map.ndim != 4 or feature_map.shape[0] < 1:
        raise ValueError("Unexpected Ultralytics SAM3 feature map shape")
    return feature_map[0].float()


def _extract_feature_vector_from_box(
    feature_map: torch.Tensor,
    box_xywh: List[float],
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    feat_c, feat_h, feat_w = feature_map.shape
    x, y, w, h = [float(v) for v in box_xywh]
    x2 = x + w
    y2 = y + h

    fx1 = int(np.floor((x / max(image_width, 1)) * feat_w))
    fy1 = int(np.floor((y / max(image_height, 1)) * feat_h))
    fx2 = int(np.ceil((x2 / max(image_width, 1)) * feat_w))
    fy2 = int(np.ceil((y2 / max(image_height, 1)) * feat_h))

    fx1 = max(0, min(fx1, feat_w - 1))
    fy1 = max(0, min(fy1, feat_h - 1))
    fx2 = max(fx1 + 1, min(fx2, feat_w))
    fy2 = max(fy1 + 1, min(fy2, feat_h))

    roi = feature_map[:, fy1:fy2, fx1:fx2]
    if roi.numel() == 0:
        return feature_map.reshape(feat_c, -1).mean(dim=1)
    return roi.reshape(feat_c, -1).mean(dim=1)


def _extract_feature_vector_from_mask(feature_map: torch.Tensor, mask_2d: np.ndarray) -> torch.Tensor:
    feat_c, feat_h, feat_w = feature_map.shape
    mask_np = np.asarray(mask_2d)
    if mask_np.ndim > 2:
        mask_np = np.squeeze(mask_np)
    if mask_np.ndim != 2:
        return feature_map.reshape(feat_c, -1).mean(dim=1)

    mask_tensor = torch.from_numpy(mask_np.astype(np.float32)).to(feature_map.device)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    mask_low_res = F.interpolate(mask_tensor, size=(feat_h, feat_w), mode="bilinear", align_corners=False)
    mask_low_res = torch.clamp(mask_low_res.squeeze(0).squeeze(0), min=0.0, max=1.0)

    weight_sum = float(mask_low_res.sum().item())
    if weight_sum <= 1e-6:
        return feature_map.reshape(feat_c, -1).mean(dim=1)

    weighted = feature_map * mask_low_res.unsqueeze(0)
    return weighted.reshape(feat_c, -1).sum(dim=1) / mask_low_res.reshape(-1).sum()


def _cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor, eps: float = 1e-6) -> float:
    a = vec_a.float()
    b = vec_b.float()
    denom = torch.norm(a) * torch.norm(b)
    if float(denom.item()) <= eps:
        return 0.0
    return float(torch.dot(a, b).item() / float(denom.item()))


def _extract_primary_component_mask(
    mask_2d: np.ndarray,
    candidate_bnd_points: List[float],
    min_area_ratio: float = 0.0001,
) -> np.ndarray:
    mask_np = np.asarray(mask_2d)
    if mask_np.ndim > 2:
        mask_np = np.squeeze(mask_np)
    if mask_np.ndim != 2:
        return np.zeros((1, 1), dtype=np.uint8)

    binary = (mask_np > 0.5).astype(np.uint8)
    if binary.max() == 0:
        return binary

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    x, y, w, h = candidate_bnd_points
    cx = int(round(x + w / 2.0))
    cy = int(round(y + h / 2.0))
    h_img, w_img = binary.shape
    cx = max(0, min(cx, w_img - 1))
    cy = max(0, min(cy, h_img - 1))
    label_at_center = int(labels[cy, cx])

    candidate_labels: List[Tuple[int, int]] = []
    total_pixels = max(1, h_img * w_img)
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area / total_pixels < min_area_ratio:
            continue
        candidate_labels.append((label_id, area))

    if not candidate_labels:
        return np.zeros_like(binary)

    if label_at_center > 0 and any(label_id == label_at_center for label_id, _ in candidate_labels):
        chosen_label = label_at_center
    else:
        chosen_label = max(candidate_labels, key=lambda item: item[1])[0]
    return (labels == chosen_label).astype(np.uint8)


def _propose_candidate_boxes_from_similarity(
    sample_feature_vec: torch.Tensor,
    query_feature_map: torch.Tensor,
    reference_bnd_points: List[float],
    reference_size: Tuple[int, int],
    query_size: Tuple[int, int],
    top_k: int,
) -> List[Dict[str, Any]]:
    _, feat_h, feat_w = query_feature_map.shape
    query_width, query_height = query_size
    ref_width, ref_height = reference_size

    normalized_query_feats = F.normalize(query_feature_map.float(), p=2, dim=0, eps=1e-6)
    normalized_sample_vec = F.normalize(sample_feature_vec.float(), p=2, dim=0, eps=1e-6)
    sim_map = torch.einsum("c,chw->hw", normalized_sample_vec, normalized_query_feats)
    sim_map = F.avg_pool2d(sim_map[None, None], kernel_size=3, stride=1, padding=1).squeeze(0).squeeze(0)
    if SIMILAR_PEAK_NMS_KERNEL > 1:
        kernel_size = SIMILAR_PEAK_NMS_KERNEL + (SIMILAR_PEAK_NMS_KERNEL % 2 == 0)
        pooled = F.max_pool2d(sim_map[None, None], kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        peak_mask = sim_map >= pooled.squeeze(0).squeeze(0)
        sim_map_for_topk = sim_map.masked_fill(~peak_mask, float("-inf")) if bool(peak_mask.any().item()) else sim_map
    else:
        sim_map_for_topk = sim_map

    ref_w_ratio = max(reference_bnd_points[2] / max(ref_width, 1), 8.0 / max(query_width, 1))
    ref_h_ratio = max(reference_bnd_points[3] / max(ref_height, 1), 8.0 / max(query_height, 1))
    candidate_w = max(8.0, min(float(query_width), ref_w_ratio * query_width))
    candidate_h = max(8.0, min(float(query_height), ref_h_ratio * query_height))

    n_candidates = max(top_k * SIMILAR_CANDIDATE_PREFILTER_MULTIPLIER, top_k)
    n_candidates = min(n_candidates, feat_h * feat_w)
    top_values, top_indices = torch.topk(sim_map_for_topk.reshape(-1), k=n_candidates)

    candidates: List[Dict[str, Any]] = []
    for score_tensor, flat_idx_tensor in zip(top_values, top_indices):
        score = float(score_tensor.item())
        if not np.isfinite(score):
            continue
        flat_idx = int(flat_idx_tensor.item())
        grid_y = flat_idx // feat_w
        grid_x = flat_idx % feat_w
        center_x = (float(grid_x) + 0.5) / float(feat_w) * float(query_width)
        center_y = (float(grid_y) + 0.5) / float(feat_h) * float(query_height)
        clipped_box = clip_bnd_points_to_image(
            [center_x - candidate_w / 2.0, center_y - candidate_h / 2.0, candidate_w, candidate_h],
            query_width,
            query_height,
        )
        if any(bbox_iou_xywh(existing["bnd_points"], clipped_box) > SIMILAR_CANDIDATE_NMS_IOU for existing in candidates):
            continue
        candidates.append({"bnd_points": clipped_box, "coarse_similarity": score})
        if len(candidates) >= top_k:
            break
    return candidates


MODEL_LOCK = threading.Lock()
INFERENCE_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)


def run_detection_pipeline(
    image: Image.Image,
    prompt: str,
    confidence_threshold: float,
    polygon_simplify_epsilon: float,
    pic_id: Optional[str] = None,
) -> Dict[str, Any]:
    image = image.convert("RGB")
    start_time = time.perf_counter()
    normalized_pic_id = (pic_id or "").strip() or uuid.uuid4().hex[:16]

    original_classes = split_prompt_classes(prompt)
    if not original_classes:
        raise ValueError("Prompt is empty after parsing. Use ';' or ',' to separate classes.")

    classes_info: List[Dict[str, str]] = []
    translated_classes: List[str] = []
    for one_class in original_classes:
        translated = translate_to_english(one_class)
        translated = translated.strip() if translated else one_class
        if not translated:
            translated = one_class

        translated_classes.append(translated)
        classes_info.append(
            {
                "class_name": translated,
                "original_class_name": one_class,
            }
        )

    translated_prompt = "; ".join(translated_classes)
    original_prompt_joined = "; ".join(original_classes)
    was_translated = translated_prompt != original_prompt_joined

    for index, class_info in enumerate(classes_info):
        class_info["color"] = build_class_color(index)

    pic_labels: List[Dict[str, Any]] = []

    lock_ctx = MODEL_LOCK if SERIALIZE_MODEL_ACCESS else contextlib.nullcontext()
    autocast_ctx = _inference_autocast_context()
    with torch.inference_mode(), lock_ctx, autocast_ctx:
        result = run_ultralytics_prediction(
            image,
            text=translated_classes,
            confidence_threshold=confidence_threshold,
        )
        arrays = extract_ultralytics_arrays(result)

    masks_np = arrays["masks"]
    boxes_np = arrays["boxes"]
    scores_np = arrays["scores"]
    classes_np = arrays["classes"]

    for mask, box, score, class_index in zip(masks_np, boxes_np, scores_np, classes_np):
        if float(score) < confidence_threshold:
            continue

        class_index = int(class_index)
        if class_index < 0 or class_index >= len(classes_info):
            continue

        class_info = classes_info[class_index]
        bnd_points = bbox_to_xywh(box)
        polygons = mask_to_polygons(mask, epsilon=polygon_simplify_epsilon)
        polygon_points = polygons[0]["points"] if polygons else bbox_xywh_to_polygon_points(bnd_points)

        pic_labels.append(
            {
                "category": class_info["original_class_name"],
                "translated_category": class_info["class_name"],
                "score": round(float(score), 6),
                "bnd_points": bnd_points,
                "polygon_points": polygon_points,
                "mask_area": int(np.count_nonzero(np.asarray(mask) > 0.5)),
            }
        )

    if EMPTY_CUDA_CACHE_EACH_REQUEST and torch.cuda.is_available():
        torch.cuda.empty_cache()

    detection_details: Dict[str, int] = {}
    for one_label in pic_labels:
        category = one_label["category"]
        detection_details[category] = detection_details.get(category, 0) + 1

    processing_time_ms = int((time.perf_counter() - start_time) * 1000)

    response: Dict[str, Any] = {
        "model": MODEL_LABEL,
        "pic_id": normalized_pic_id,
        "success": True,
        "pic_labels": pic_labels,
        "num_detections": len(pic_labels),
        "classes_detected": len(detection_details),
        "detection_details": detection_details,
        "prompt": prompt,
        "translated_prompt": translated_prompt if was_translated else None,
        "was_translated": was_translated,
        "confidence_threshold": confidence_threshold,
        "created": int(time.time()),
        "processing_time_ms": processing_time_ms,
    }

    return response


def normalize_box_segmentation_inputs(raw_bnd_points: Any) -> List[List[float]]:
    if not isinstance(raw_bnd_points, list) or len(raw_bnd_points) == 0:
        raise ValueError(
            "bnd_points payload is required. Use bnd_points=[x,y,w,h] or bnd_points_list=[[x,y,w,h], ...]"
        )

    first_item = raw_bnd_points[0]
    if isinstance(first_item, (list, tuple, np.ndarray)):
        return [list(one_box) for one_box in raw_bnd_points]
    return [list(raw_bnd_points)]


def _build_box_segmentation_result(
    arrays: Dict[str, np.ndarray],
    index: int,
    requested_bnd_points: List[float],
    polygon_simplify_epsilon: float,
) -> Dict[str, Any]:
    masks_np = arrays["masks"]
    boxes_np = arrays["boxes"]
    scores_np = arrays["scores"]

    used_fallback = False
    score = 0.0
    mask_area = 0
    selected_bnd_points = requested_bnd_points
    polygon_points: List[List[float]] = bbox_xywh_to_polygon_points(requested_bnd_points)

    has_predictions = (
        isinstance(scores_np, np.ndarray)
        and scores_np.size > 0
        and isinstance(masks_np, np.ndarray)
        and masks_np.shape[0] > 0
    )
    if has_predictions:
        best_index = min(index, scores_np.shape[0] - 1)
        score = round(float(scores_np[best_index]), 6)

        best_mask = np.asarray(masks_np[best_index])
        if best_mask.ndim > 2:
            best_mask = np.squeeze(best_mask)
        mask_area = int(np.count_nonzero(best_mask > 0.5))

        if isinstance(boxes_np, np.ndarray) and boxes_np.shape[0] > best_index:
            selected_bnd_points = bbox_to_xywh(boxes_np[best_index])

        polygons = mask_to_polygons(best_mask, epsilon=polygon_simplify_epsilon)
        if polygons:
            polygon_points = polygons[0]["points"]
        else:
            used_fallback = True
    else:
        used_fallback = True

    return {
        "input_bnd_points": requested_bnd_points,
        "bnd_points": selected_bnd_points,
        "polygon_points": polygon_points,
        "score": score,
        "mask_area": mask_area,
        "used_fallback": used_fallback,
    }


def run_box_segmentation_pipeline(
    image: Image.Image,
    bnd_points: Any,
    polygon_simplify_epsilon: float,
    pic_id: Optional[str] = None,
) -> Dict[str, Any]:
    image = image.convert("RGB")
    start_time = time.perf_counter()
    normalized_pic_id = (pic_id or "").strip() or uuid.uuid4().hex[:16]

    raw_boxes = normalize_box_segmentation_inputs(bnd_points)

    clipped_boxes: List[List[float]] = []
    xyxy_boxes: List[List[float]] = []
    for index, one_bnd_points in enumerate(raw_boxes):
        try:
            clipped_bnd_points = clip_bnd_points_to_image(
                bnd_points=one_bnd_points,
                image_width=image.width,
                image_height=image.height,
            )
        except ValueError as exc:
            raise ValueError(f"Invalid bnd_points_list[{index}]: {exc}") from exc

        clipped_boxes.append(clipped_bnd_points)
        xyxy_boxes.append(bbox_xywh_to_xyxy(clipped_bnd_points))

    segmentations: List[Dict[str, Any]] = []
    lock_ctx = MODEL_LOCK if SERIALIZE_MODEL_ACCESS else contextlib.nullcontext()
    autocast_ctx = _inference_autocast_context()
    with torch.inference_mode(), lock_ctx, autocast_ctx:
        result = run_ultralytics_prediction(
            image,
            bboxes=xyxy_boxes,
            confidence_threshold=0.0,
        )
        arrays = extract_ultralytics_arrays(result)
        for index, clipped_box in enumerate(clipped_boxes):
            one_result = _build_box_segmentation_result(
                arrays=arrays,
                index=index,
                requested_bnd_points=clipped_box,
                polygon_simplify_epsilon=polygon_simplify_epsilon,
            )
            one_result["index"] = index
            segmentations.append(one_result)

    if EMPTY_CUDA_CACHE_EACH_REQUEST and torch.cuda.is_available():
        torch.cuda.empty_cache()

    processing_time_ms = int((time.perf_counter() - start_time) * 1000)
    response: Dict[str, Any] = {
        "model": MODEL_LABEL,
        "pic_id": normalized_pic_id,
        "success": True,
        "segmentations": segmentations,
        "num_segmentations": len(segmentations),
        "created": int(time.time()),
        "processing_time_ms": processing_time_ms,
    }

    # Backward compatibility for previous single-box clients.
    if len(segmentations) == 1:
        first = segmentations[0]
        response.update(
            {
                "bnd_points": first["bnd_points"],
                "polygon_points": first["polygon_points"],
                "score": first["score"],
                "mask_area": first["mask_area"],
                "used_fallback": first["used_fallback"],
            }
        )

    return response


def _prepare_similar_reference_context(
    reference_image: Image.Image,
    reference_bnd_points: Optional[List[float]],
    top_k: int,
) -> Dict[str, Any]:
    reference_image = reference_image.convert("RGB")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if reference_bnd_points is None:
        raise ValueError("reference_bnd_points is required, format: [x, y, w, h]")

    reference_bnd_points = clip_bnd_points_to_image(
        bnd_points=reference_bnd_points,
        image_width=reference_image.width,
        image_height=reference_image.height,
    )
    reference_xyxy = bbox_xywh_to_xyxy(reference_bnd_points)

    reference_feature_map = set_ultralytics_image_and_features(reference_image)
    result = run_ultralytics_prediction(
        reference_image,
        bboxes=[reference_xyxy],
        confidence_threshold=0.0,
        reset_cached_image=False,
    )
    arrays = extract_ultralytics_arrays(result)
    ref_masks_np = arrays["masks"]
    ref_boxes_np = arrays["boxes"]
    ref_scores_np = arrays["scores"]

    if ref_scores_np.size == 0 or ref_masks_np.shape[0] == 0:
        raise ValueError("reference image SAM+box did not produce valid masks")

    ref_best_idx = int(np.argmax(ref_scores_np))
    ref_best_mask = np.asarray(ref_masks_np[ref_best_idx])
    ref_sam_score = float(ref_scores_np[ref_best_idx])
    if isinstance(ref_boxes_np, np.ndarray) and ref_boxes_np.shape[0] > ref_best_idx:
        ref_best_xyxy = clip_xyxy_to_image([float(v) for v in ref_boxes_np[ref_best_idx]], reference_image.width, reference_image.height)
        ref_best_xywh = bbox_to_xywh(np.asarray(ref_best_xyxy))
    else:
        ref_best_xywh = reference_bnd_points
        ref_best_xyxy = reference_xyxy

    ref_primary_mask_u8 = _extract_primary_component_mask(ref_best_mask, ref_best_xywh)
    if ref_primary_mask_u8.max() == 0:
        raise ValueError("reference mask is empty after postprocess")

    ref_box_feature_vec = _extract_feature_vector_from_box(
        reference_feature_map,
        ref_best_xywh,
        reference_image.width,
        reference_image.height,
    )
    ref_mask_feature_vec = _extract_feature_vector_from_mask(reference_feature_map, ref_primary_mask_u8.astype(np.float32))
    reference_feature_vec = 0.15 * ref_box_feature_vec + 0.85 * ref_mask_feature_vec

    ref_detection_for_viz = [
        {
            "class_name": "reference_object",
            "original_class_name": "reference_object",
            "masks": np.asarray([ref_primary_mask_u8.astype(np.float32)]),
            "boxes": np.asarray([ref_best_xyxy]),
            "scores": np.asarray([max(0.0, min(1.0, ref_sam_score))]),
            "color": build_class_color(1),
        }
    ]
    reference_result_image = visualize_results(reference_image, ref_detection_for_viz)

    return {
        "reference_image": reference_image,
        "reference_bnd_points": reference_bnd_points,
        "reference_mask": ref_primary_mask_u8,
        "reference_feature_vec": reference_feature_vec,
        "ref_best_xywh": ref_best_xywh,
        "reference_result_image": reference_result_image,
    }


def _run_similar_query_with_reference_context(
    reference_ctx: Dict[str, Any],
    query_image: Image.Image,
    top_k: int,
    similarity_threshold: float,
    sam_threshold: float,
    polygon_simplify_epsilon: float,
    pic_id: str,
) -> Dict[str, Any]:
    query_image = query_image.convert("RGB")
    query_start_time = time.perf_counter()
    reference_image = reference_ctx["reference_image"]
    reference_bnd_points = reference_ctx["reference_bnd_points"]
    reference_feature_vec = reference_ctx["reference_feature_vec"]
    ref_best_xywh = reference_ctx["ref_best_xywh"]

    set_image_start = time.perf_counter()
    query_feature_map = set_ultralytics_image_and_features(query_image)
    set_image_ms = int((time.perf_counter() - set_image_start) * 1000)

    propose_start = time.perf_counter()
    prompt_candidate_count = min(max(top_k * SIMILAR_CANDIDATE_MULTIPLIER, top_k), MAX_SIMILAR_PROMPT_CANDIDATES)
    query_prompt_boxes = _propose_candidate_boxes_from_similarity(
        sample_feature_vec=reference_feature_vec,
        query_feature_map=query_feature_map,
        reference_bnd_points=ref_best_xywh,
        reference_size=(reference_image.width, reference_image.height),
        query_size=(query_image.width, query_image.height),
        top_k=prompt_candidate_count,
    )
    propose_ms = int((time.perf_counter() - propose_start) * 1000)

    matched_labels: List[Dict[str, Any]] = []
    matched_masks: List[np.ndarray] = []
    matched_boxes_xyxy: List[List[float]] = []
    matched_scores: List[float] = []
    prompt_forward_ms = 0
    candidate_loop_start = time.perf_counter()

    for one_candidate in query_prompt_boxes:
        candidate_bnd_points = one_candidate["bnd_points"]
        candidate_xyxy = bbox_xywh_to_xyxy(candidate_bnd_points)
        prompt_start = time.perf_counter()
        result = run_ultralytics_prediction(
            query_image,
            bboxes=[candidate_xyxy],
            confidence_threshold=0.0,
            reset_cached_image=False,
        )
        prompt_forward_ms += int((time.perf_counter() - prompt_start) * 1000)
        arrays = extract_ultralytics_arrays(result)
        masks_np = arrays["masks"]
        boxes_np = arrays["boxes"]
        scores_np = arrays["scores"]
        if scores_np.size == 0 or masks_np.shape[0] == 0:
            continue

        best_local = None
        for idx in range(masks_np.shape[0]):
            one_mask = np.asarray(masks_np[idx])
            one_box_xyxy = (
                [float(v) for v in boxes_np[idx]]
                if isinstance(boxes_np, np.ndarray) and boxes_np.shape[0] > idx
                else candidate_xyxy
            )
            one_box_xyxy = clip_xyxy_to_image(one_box_xyxy, query_image.width, query_image.height)
            one_box_xywh = bbox_to_xywh(np.asarray(one_box_xyxy))
            one_sam_score = float(scores_np[idx])
            one_primary_u8 = _extract_primary_component_mask(one_mask, candidate_bnd_points)
            if one_primary_u8.size <= 1 or one_primary_u8.max() == 0:
                continue
            one_primary = one_primary_u8.astype(np.float32)
            one_mask_feat = _extract_feature_vector_from_mask(query_feature_map, one_primary)
            one_similarity = _cosine_similarity(reference_feature_vec, one_mask_feat)
            one_combined = 0.85 * one_similarity + 0.15 * one_sam_score
            if best_local is None or one_combined > best_local["combined_score"]:
                best_local = {
                    "mask": one_primary,
                    "mask_u8": one_primary_u8,
                    "box_xyxy": one_box_xyxy,
                    "box_xywh": one_box_xywh,
                    "sam_score": one_sam_score,
                    "similarity_score": one_similarity,
                    "combined_score": one_combined,
                }

        if best_local is None:
            continue
        similarity_score = float(best_local["similarity_score"])
        sam_score = float(best_local["sam_score"])
        primary_area = int(np.count_nonzero(best_local["mask_u8"]))
        area_ratio = primary_area / max(1, query_image.width * query_image.height)
        if area_ratio <= 0.0002:
            continue
        if any(bbox_iou_xywh(existing["bnd_points"], best_local["box_xywh"]) > 0.75 for existing in matched_labels):
            continue

        coarse_similarity = _cosine_similarity(
            reference_feature_vec,
            _extract_feature_vector_from_box(query_feature_map, best_local["box_xywh"], query_image.width, query_image.height),
        )
        if similarity_score < similarity_threshold or sam_score < sam_threshold:
            continue
        polygons = mask_to_polygons(best_local["mask"], epsilon=polygon_simplify_epsilon)
        polygon_points = polygons[0]["points"] if polygons else bbox_xywh_to_polygon_points(best_local["box_xywh"])
        matched_labels.append(
            {
                "category": "similar_object",
                "score": round(sam_score, 6),
                "similarity_score": round(similarity_score, 6),
                "combined_score": round(float(best_local["combined_score"]), 6),
                "coarse_similarity": round(float(coarse_similarity), 6),
                "bnd_points": best_local["box_xywh"],
                "polygon_points": polygon_points,
                "mask_area": primary_area,
            }
        )
        matched_masks.append(np.asarray(best_local["mask"]))
        matched_boxes_xyxy.append(best_local["box_xyxy"])
        matched_scores.append(max(0.0, min(1.0, float(best_local["combined_score"]))))

    candidate_loop_ms = int((time.perf_counter() - candidate_loop_start) * 1000)
    order = sorted(range(len(matched_labels)), key=lambda i: matched_labels[i]["combined_score"], reverse=True)[:top_k]
    matched_labels = [matched_labels[i] for i in order]
    matched_masks = [matched_masks[i] for i in order]
    matched_boxes_xyxy = [matched_boxes_xyxy[i] for i in order]
    matched_scores = [matched_scores[i] for i in order]

    result_image = None
    if matched_masks and matched_boxes_xyxy and matched_scores:
        detection_for_viz = [
            {
                "class_name": "similar_object",
                "original_class_name": "similar_object",
                "masks": np.asarray(matched_masks),
                "boxes": np.asarray(matched_boxes_xyxy),
                "scores": np.asarray(matched_scores),
                "color": build_class_color(0),
            }
        ]
        result_image = visualize_results(query_image, detection_for_viz)

    processing_time_ms = int((time.perf_counter() - query_start_time) * 1000)
    return {
        "model": MODEL_LABEL,
        "pic_id": pic_id,
        "success": True,
        "similar_mode": "feature_match",
        "reference_bnd_points": [round(float(v), 3) for v in reference_bnd_points],
        "reference_box_auto_generated": False,
        "top_k": int(top_k),
        "similarity_threshold": float(similarity_threshold),
        "sam_threshold": float(sam_threshold),
        "num_candidates": len(query_prompt_boxes),
        "num_matches": len(matched_labels),
        "pic_labels": matched_labels,
        "reference_result_image": reference_ctx["reference_result_image"],
        "result_image": result_image,
        "created": int(time.time()),
        "processing_time_ms": processing_time_ms,
        "profile": {
            "set_image_ms": set_image_ms,
            "propose_candidates_ms": propose_ms,
            "prompt_forward_ms": prompt_forward_ms,
            "candidate_loop_ms": candidate_loop_ms,
            "prompt_candidates_requested": prompt_candidate_count,
            "prompt_candidates_used": len(query_prompt_boxes),
            "candidate_nms_iou": SIMILAR_CANDIDATE_NMS_IOU,
            "peak_nms_kernel": SIMILAR_PEAK_NMS_KERNEL,
        },
    }


def _crop_reference_patch(
    reference_image: Image.Image,
    reference_bnd_points: List[float],
    reference_mask: Optional[np.ndarray] = None,
    padding_ratio: float = 0.15,
) -> Tuple[Image.Image, List[float]]:
    x, y, w, h = [float(v) for v in reference_bnd_points]
    pad = max(2.0, max(w, h) * padding_ratio)
    x1 = max(0.0, x - pad)
    y1 = max(0.0, y - pad)
    x2 = min(float(reference_image.width), x + w + pad)
    y2 = min(float(reference_image.height), y + h + pad)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid reference crop after padding")
    crop_box = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
    patch = reference_image.crop(crop_box).convert("RGB")
    if reference_mask is not None:
        mask_np = np.asarray(reference_mask)
        if mask_np.ndim > 2:
            mask_np = np.squeeze(mask_np)
        if mask_np.ndim == 2 and mask_np.shape[0] == reference_image.height and mask_np.shape[1] == reference_image.width:
            mask_crop = mask_np[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]]
            mask_crop = (mask_crop > 0.5).astype(np.uint8)
            if mask_crop.max() > 0:
                patch_arr = np.asarray(patch).copy()
                neutral = np.full_like(patch_arr, 245)
                patch_arr = np.where(mask_crop[..., None].astype(bool), patch_arr, neutral)
                patch = Image.fromarray(patch_arr, mode="RGB")
    patch_box = [x - x1, y - y1, w, h]
    return patch, patch_box


def _build_concat_prompt_image(
    reference_patch: Image.Image,
    reference_patch_box: List[float],
    query_image: Image.Image,
    scale: float,
) -> Tuple[Image.Image, List[float], Dict[str, int]]:
    scale = max(0.1, float(scale))
    scaled_w = max(8, int(round(reference_patch.width * scale)))
    scaled_h = max(8, int(round(reference_patch.height * scale)))
    scaled_patch = reference_patch.resize((scaled_w, scaled_h), Image.Resampling.BICUBIC)

    left_width = max(scaled_w + CONCAT_PROMPT_PADDING * 2, 16)
    canvas_w = left_width + CONCAT_PROMPT_SEPARATOR + query_image.width
    canvas_h = max(query_image.height, scaled_h + CONCAT_PROMPT_PADDING * 2)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))

    patch_x = CONCAT_PROMPT_PADDING + max(0, (left_width - scaled_w - CONCAT_PROMPT_PADDING * 2) // 2)
    patch_y = max(CONCAT_PROMPT_PADDING, (canvas_h - scaled_h) // 2)
    query_x = left_width + CONCAT_PROMPT_SEPARATOR
    query_y = 0

    canvas.paste(scaled_patch, (patch_x, patch_y))
    canvas.paste(query_image, (query_x, query_y))

    ref_x, ref_y, ref_w, ref_h = reference_patch_box
    prompt_box = [
        patch_x + ref_x * scale,
        patch_y + ref_y * scale,
        patch_x + (ref_x + ref_w) * scale,
        patch_y + (ref_y + ref_h) * scale,
    ]
    regions = {
        "query_x": query_x,
        "query_y": query_y,
        "query_w": query_image.width,
        "query_h": query_image.height,
        "prompt_region_w": left_width + CONCAT_PROMPT_SEPARATOR,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
    }
    return canvas, prompt_box, regions


def _intersect_concat_box_to_query(box_xyxy: List[float], regions: Dict[str, int]) -> Optional[List[float]]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    query_x1 = float(regions["query_x"])
    query_y1 = float(regions["query_y"])
    query_x2 = query_x1 + float(regions["query_w"])
    query_y2 = query_y1 + float(regions["query_h"])

    inter_x1 = max(x1, query_x1)
    inter_y1 = max(y1, query_y1)
    inter_x2 = min(x2, query_x2)
    inter_y2 = min(y2, query_y2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return None

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    raw_area = max(1.0, (x2 - x1) * (y2 - y1))
    if inter_area < 4.0:
        return None

    return [
        inter_x1 - query_x1,
        inter_y1 - query_y1,
        inter_x2 - query_x1,
        inter_y2 - query_y1,
    ]


def _translate_concat_mask_to_query(mask: np.ndarray, regions: Dict[str, int]) -> np.ndarray:
    mask_arr = np.asarray(mask)
    if mask_arr.ndim > 2:
        mask_arr = np.squeeze(mask_arr)
    query_x = regions["query_x"]
    query_y = regions["query_y"]
    query_w = regions["query_w"]
    query_h = regions["query_h"]
    return mask_arr[query_y : query_y + query_h, query_x : query_x + query_w]


def mask_to_xyxy(mask_2d: np.ndarray) -> Optional[List[float]]:
    mask_np = np.asarray(mask_2d)
    if mask_np.ndim > 2:
        mask_np = np.squeeze(mask_np)
    if mask_np.ndim != 2:
        return None
    ys, xs = np.where(mask_np > 0.5)
    if xs.size == 0 or ys.size == 0:
        return None
    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)
    return [x1, y1, x2, y2]


def save_debug_image(image: Image.Image, prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_prefix = re.sub(r"[^a-zA-Z0-9._-]", "_", prefix or "debug")
    filename = f"{safe_prefix}_{timestamp}.jpg"
    image.convert("RGB").save(RESULT_DIR / filename, quality=92)
    return filename


def _run_concat_prompt_query(
    reference_ctx: Dict[str, Any],
    query_image: Image.Image,
    top_k: int,
    similarity_threshold: float,
    sam_threshold: float,
    polygon_simplify_epsilon: float,
    pic_id: str,
    prompt_text: Optional[str] = None,
) -> Dict[str, Any]:
    query_image = query_image.convert("RGB")
    query_start_time = time.perf_counter()
    reference_image = reference_ctx["reference_image"]
    reference_bnd_points = reference_ctx["reference_bnd_points"]
    reference_feature_vec = reference_ctx["reference_feature_vec"]
    reference_mask = reference_ctx.get("reference_mask")
    reference_patch, reference_patch_box = _crop_reference_patch(reference_image, reference_bnd_points, reference_mask)
    text_prompt, original_prompt, translated_prompt, was_translated = prepare_single_text_prompt(prompt_text)
    category_name = original_prompt or "similar_object"
    concat_debug_images: List[str] = []

    matched_labels: List[Dict[str, Any]] = []
    matched_masks: List[np.ndarray] = []
    matched_boxes_xyxy: List[List[float]] = []
    matched_scores: List[float] = []
    prompt_forward_ms = 0
    candidate_count = 0

    scales = CONCAT_PROMPT_SCALES or [1.0]
    for scale in scales:
        concat_image, prompt_box_xyxy, regions = _build_concat_prompt_image(
            reference_patch,
            reference_patch_box,
            query_image,
            scale,
        )
        concat_debug_images.append(save_debug_image(concat_image, f"concat_prompt_{pic_id}_{scale:g}"))
        prompt_start = time.perf_counter()
        result = run_ultralytics_prediction(
            concat_image,
            text=text_prompt,
            bboxes=[prompt_box_xyxy],
            confidence_threshold=0.0,
        )
        prompt_forward_ms += int((time.perf_counter() - prompt_start) * 1000)
        arrays = extract_ultralytics_arrays(result)
        masks_np = arrays["masks"]
        boxes_np = arrays["boxes"]
        scores_np = arrays["scores"]
        candidate_count += int(scores_np.size)
        if scores_np.size == 0 or masks_np.shape[0] == 0:
            continue

        query_feature_map = set_ultralytics_image_and_features(query_image)

        for idx in range(masks_np.shape[0]):
            raw_concat_box_xyxy = (
                [float(v) for v in boxes_np[idx]]
                if isinstance(boxes_np, np.ndarray) and boxes_np.shape[0] > idx
                else None
            )
            query_box_from_concat = (
                _intersect_concat_box_to_query(raw_concat_box_xyxy, regions)
                if raw_concat_box_xyxy is not None
                else None
            )
            if raw_concat_box_xyxy is not None and query_box_from_concat is None:
                continue

            query_mask = _translate_concat_mask_to_query(masks_np[idx], regions)
            mask_box_xyxy = mask_to_xyxy(query_mask)
            if mask_box_xyxy is None:
                continue
            query_box_xyxy = clip_xyxy_to_image(mask_box_xyxy, query_image.width, query_image.height)
            if query_box_from_concat is not None:
                query_box_xyxy = clip_xyxy_to_image(
                    [
                        min(query_box_xyxy[0], query_box_from_concat[0]),
                        min(query_box_xyxy[1], query_box_from_concat[1]),
                        max(query_box_xyxy[2], query_box_from_concat[2]),
                        max(query_box_xyxy[3], query_box_from_concat[3]),
                    ],
                    query_image.width,
                    query_image.height,
                )
            query_box_xywh = bbox_to_xywh(np.asarray(query_box_xyxy))
            primary_u8 = _extract_primary_component_mask(query_mask, query_box_xywh)
            if primary_u8.size <= 1 or primary_u8.max() == 0:
                continue
            primary_box_xyxy = mask_to_xyxy(primary_u8)
            if primary_box_xyxy is None:
                continue
            query_box_xyxy = clip_xyxy_to_image(primary_box_xyxy, query_image.width, query_image.height)
            query_box_xywh = bbox_to_xywh(np.asarray(query_box_xyxy))
            primary_area = int(np.count_nonzero(primary_u8))
            area_ratio = primary_area / max(1, query_image.width * query_image.height)
            if area_ratio <= 0.0002:
                continue

            mask_feat = _extract_feature_vector_from_mask(query_feature_map, primary_u8.astype(np.float32))
            similarity_score = _cosine_similarity(reference_feature_vec, mask_feat)
            sam_score = float(scores_np[idx])
            if similarity_score < similarity_threshold or sam_score < sam_threshold:
                continue
            combined_score = 0.5 * similarity_score + 0.5 * sam_score
            if any(bbox_iou_xywh(existing["bnd_points"], query_box_xywh) > 0.75 for existing in matched_labels):
                continue

            polygons = mask_to_polygons(primary_u8.astype(np.float32), epsilon=polygon_simplify_epsilon)
            polygon_points = polygons[0]["points"] if polygons else bbox_xywh_to_polygon_points(query_box_xywh)
            matched_labels.append(
                {
                    "category": category_name,
                    "translated_category": translated_prompt if original_prompt else None,
                    "score": round(sam_score, 6),
                    "similarity_score": round(similarity_score, 6),
                    "combined_score": round(float(combined_score), 6),
                    "coarse_similarity": round(similarity_score, 6),
                    "bnd_points": query_box_xywh,
                    "polygon_points": polygon_points,
                    "mask_area": primary_area,
                    "concat_scale": round(float(scale), 4),
                }
            )
            matched_masks.append(primary_u8.astype(np.float32))
            matched_boxes_xyxy.append(query_box_xyxy)
            matched_scores.append(max(0.0, min(1.0, float(combined_score))))

    order = sorted(range(len(matched_labels)), key=lambda i: matched_labels[i]["combined_score"], reverse=True)[:top_k]
    matched_labels = [matched_labels[i] for i in order]
    matched_masks = [matched_masks[i] for i in order]
    matched_boxes_xyxy = [matched_boxes_xyxy[i] for i in order]
    matched_scores = [matched_scores[i] for i in order]

    result_image = None
    if matched_masks and matched_boxes_xyxy and matched_scores:
        detection_for_viz = [
            {
                "class_name": "similar_object",
                "original_class_name": category_name,
                "masks": np.asarray(matched_masks),
                "boxes": np.asarray(matched_boxes_xyxy),
                "scores": np.asarray(matched_scores),
                "color": build_class_color(0),
            }
        ]
        result_image = visualize_results(query_image, detection_for_viz)

    processing_time_ms = int((time.perf_counter() - query_start_time) * 1000)
    return {
        "model": MODEL_LABEL,
        "pic_id": pic_id,
        "success": True,
        "similar_mode": "concat_prompt",
        "prompt": original_prompt,
        "translated_prompt": translated_prompt if was_translated else None,
        "was_translated": was_translated,
        "box_text_prompt_enabled": text_prompt is not None,
        "reference_bnd_points": [round(float(v), 3) for v in reference_bnd_points],
        "reference_box_auto_generated": False,
        "top_k": int(top_k),
        "similarity_threshold": float(similarity_threshold),
        "sam_threshold": float(sam_threshold),
        "num_candidates": candidate_count,
        "num_matches": len(matched_labels),
        "pic_labels": matched_labels,
        "reference_result_image": reference_ctx["reference_result_image"],
        "concat_prompt_images": concat_debug_images,
        "result_image": result_image,
        "created": int(time.time()),
        "processing_time_ms": processing_time_ms,
        "profile": {
            "prompt_forward_ms": prompt_forward_ms,
            "concat_scales": scales,
            "concat_padding": CONCAT_PROMPT_PADDING,
            "concat_separator": CONCAT_PROMPT_SEPARATOR,
        },
    }


def _run_same_image_prompt_query(
    image: Image.Image,
    reference_bnd_points: List[float],
    top_k: int,
    similarity_threshold: float,
    sam_threshold: float,
    polygon_simplify_epsilon: float,
    pic_id: str,
) -> Dict[str, Any]:
    image = image.convert("RGB")
    query_start_time = time.perf_counter()
    reference_bnd_points = clip_bnd_points_to_image(reference_bnd_points, image.width, image.height)
    reference_xyxy = bbox_xywh_to_xyxy(reference_bnd_points)

    set_image_start = time.perf_counter()
    image_feature_map = set_ultralytics_image_and_features(image)
    set_image_ms = int((time.perf_counter() - set_image_start) * 1000)
    reference_box_feature_vec = _extract_feature_vector_from_box(
        image_feature_map,
        reference_bnd_points,
        image.width,
        image.height,
    )

    prompt_start = time.perf_counter()
    result = run_ultralytics_prediction(
        image,
        bboxes=[reference_xyxy],
        confidence_threshold=0.0,
        reset_cached_image=False,
    )
    prompt_forward_ms = int((time.perf_counter() - prompt_start) * 1000)
    arrays = extract_ultralytics_arrays(result)
    masks_np = arrays["masks"]
    boxes_np = arrays["boxes"]
    scores_np = arrays["scores"]

    matched_labels: List[Dict[str, Any]] = []
    matched_masks: List[np.ndarray] = []
    matched_boxes_xyxy: List[List[float]] = []
    matched_scores: List[float] = []
    reference_result_image = None
    reference_feature_vec = reference_box_feature_vec

    if scores_np.size > 0 and masks_np.shape[0] > 0:
        best_ref_idx = 0
        best_ref_overlap = -1.0
        for idx in range(masks_np.shape[0]):
            one_box_xyxy = (
                [float(v) for v in boxes_np[idx]]
                if isinstance(boxes_np, np.ndarray) and boxes_np.shape[0] > idx
                else reference_xyxy
            )
            one_box_xywh = bbox_to_xywh(np.asarray(clip_xyxy_to_image(one_box_xyxy, image.width, image.height)))
            overlap = bbox_iou_xywh(reference_bnd_points, one_box_xywh)
            if overlap > best_ref_overlap:
                best_ref_overlap = overlap
                best_ref_idx = idx

        ref_mask = np.asarray(masks_np[best_ref_idx])
        ref_primary_u8 = _extract_primary_component_mask(ref_mask, reference_bnd_points)
        if ref_primary_u8.max() > 0:
            reference_mask_feature_vec = _extract_feature_vector_from_mask(image_feature_map, ref_primary_u8.astype(np.float32))
            reference_feature_vec = 0.15 * reference_box_feature_vec + 0.85 * reference_mask_feature_vec
            ref_box_xyxy = mask_to_xyxy(ref_primary_u8)
            if ref_box_xyxy is None:
                ref_box_xyxy = reference_xyxy
            ref_detection_for_viz = [
                {
                    "class_name": "reference_object",
                    "original_class_name": "reference_object",
                    "masks": np.asarray([ref_primary_u8.astype(np.float32)]),
                    "boxes": np.asarray([clip_xyxy_to_image(ref_box_xyxy, image.width, image.height)]),
                    "scores": np.asarray([max(0.0, min(1.0, float(scores_np[best_ref_idx])))]),
                    "color": build_class_color(1),
                }
            ]
            reference_result_image = visualize_results(image, ref_detection_for_viz)

    for idx in range(masks_np.shape[0]):
        one_mask = np.asarray(masks_np[idx])
        one_box_xyxy = (
            [float(v) for v in boxes_np[idx]]
            if isinstance(boxes_np, np.ndarray) and boxes_np.shape[0] > idx
            else reference_xyxy
        )
        one_box_xyxy = clip_xyxy_to_image(one_box_xyxy, image.width, image.height)
        one_box_xywh = bbox_to_xywh(np.asarray(one_box_xyxy))
        primary_u8 = _extract_primary_component_mask(one_mask, one_box_xywh)
        if primary_u8.size <= 1 or primary_u8.max() == 0:
            continue

        primary_box_xyxy = mask_to_xyxy(primary_u8)
        if primary_box_xyxy is not None:
            one_box_xyxy = clip_xyxy_to_image(primary_box_xyxy, image.width, image.height)
            one_box_xywh = bbox_to_xywh(np.asarray(one_box_xyxy))

        primary_area = int(np.count_nonzero(primary_u8))
        area_ratio = primary_area / max(1, image.width * image.height)
        if area_ratio <= 0.0002:
            continue

        mask_feature_vec = _extract_feature_vector_from_mask(image_feature_map, primary_u8.astype(np.float32))
        similarity_score = _cosine_similarity(reference_feature_vec, mask_feature_vec)
        sam_score = float(scores_np[idx]) if scores_np.size > idx else 0.0
        if similarity_score < similarity_threshold or sam_score < sam_threshold:
            continue
        if any(bbox_iou_xywh(existing["bnd_points"], one_box_xywh) > 0.75 for existing in matched_labels):
            continue

        combined_score = 0.5 * similarity_score + 0.5 * sam_score
        polygons = mask_to_polygons(primary_u8.astype(np.float32), epsilon=polygon_simplify_epsilon)
        polygon_points = polygons[0]["points"] if polygons else bbox_xywh_to_polygon_points(one_box_xywh)
        matched_labels.append(
            {
                "category": "similar_object",
                "score": round(sam_score, 6),
                "similarity_score": round(similarity_score, 6),
                "combined_score": round(float(combined_score), 6),
                "coarse_similarity": round(similarity_score, 6),
                "bnd_points": one_box_xywh,
                "polygon_points": polygon_points,
                "mask_area": primary_area,
                "is_reference_overlap": bbox_iou_xywh(reference_bnd_points, one_box_xywh) > 0.5,
            }
        )
        matched_masks.append(primary_u8.astype(np.float32))
        matched_boxes_xyxy.append(one_box_xyxy)
        matched_scores.append(max(0.0, min(1.0, float(combined_score))))

    order = sorted(range(len(matched_labels)), key=lambda i: matched_labels[i]["combined_score"], reverse=True)[:top_k]
    matched_labels = [matched_labels[i] for i in order]
    matched_masks = [matched_masks[i] for i in order]
    matched_boxes_xyxy = [matched_boxes_xyxy[i] for i in order]
    matched_scores = [matched_scores[i] for i in order]

    result_image = None
    if matched_masks and matched_boxes_xyxy and matched_scores:
        detection_for_viz = [
            {
                "class_name": "similar_object",
                "original_class_name": "similar_object",
                "masks": np.asarray(matched_masks),
                "boxes": np.asarray(matched_boxes_xyxy),
                "scores": np.asarray(matched_scores),
                "color": build_class_color(0),
            }
        ]
        result_image = visualize_results(image, detection_for_viz)

    processing_time_ms = int((time.perf_counter() - query_start_time) * 1000)
    return {
        "model": MODEL_LABEL,
        "pic_id": pic_id,
        "success": True,
        "similar_mode": "same_image_prompt",
        "reference_bnd_points": [round(float(v), 3) for v in reference_bnd_points],
        "reference_box_auto_generated": False,
        "top_k": int(top_k),
        "similarity_threshold": float(similarity_threshold),
        "sam_threshold": float(sam_threshold),
        "num_candidates": int(scores_np.size),
        "num_matches": len(matched_labels),
        "pic_labels": matched_labels,
        "reference_result_image": reference_result_image,
        "result_image": result_image,
        "created": int(time.time()),
        "processing_time_ms": processing_time_ms,
        "profile": {
            "set_image_ms": set_image_ms,
            "prompt_forward_ms": prompt_forward_ms,
            "raw_visual_prompt_candidates": int(scores_np.size),
            "native_ultralytics_sam3_visual_prompt": True,
        },
    }


def run_similar_object_pipeline(
    reference_image: Image.Image,
    query_image: Image.Image,
    reference_bnd_points: Optional[List[float]],
    top_k: int,
    similarity_threshold: float,
    sam_threshold: float,
    polygon_simplify_epsilon: float,
    pic_id: Optional[str] = None,
    similar_mode: str = "concat_prompt",
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    start_time = time.perf_counter()
    normalized_pic_id = (pic_id or "").strip() or uuid.uuid4().hex[:16]
    lock_ctx = MODEL_LOCK if SERIALIZE_MODEL_ACCESS else contextlib.nullcontext()
    autocast_ctx = _inference_autocast_context()
    with torch.inference_mode(), lock_ctx, autocast_ctx:
        if similar_mode == "same_image_prompt":
            result = _run_same_image_prompt_query(
                reference_image,
                reference_bnd_points,
                top_k,
                similarity_threshold,
                sam_threshold,
                polygon_simplify_epsilon,
                normalized_pic_id,
            )
        else:
            reference_ctx = _prepare_similar_reference_context(reference_image, reference_bnd_points, top_k)
            if similar_mode == "concat_prompt":
                result = _run_concat_prompt_query(
                    reference_ctx,
                    query_image,
                    top_k,
                    similarity_threshold,
                    sam_threshold,
                    polygon_simplify_epsilon,
                    normalized_pic_id,
                    prompt,
                )
            else:
                result = _run_similar_query_with_reference_context(
                    reference_ctx,
                    query_image,
                    top_k,
                    similarity_threshold,
                    sam_threshold,
                    polygon_simplify_epsilon,
                    normalized_pic_id,
                )

    if EMPTY_CUDA_CACHE_EACH_REQUEST and torch.cuda.is_available():
        torch.cuda.empty_cache()
    result["processing_time_ms"] = int((time.perf_counter() - start_time) * 1000)
    return result


def run_similar_object_batch_pipeline(
    reference_image: Image.Image,
    query_images: List[Image.Image],
    reference_bnd_points: Optional[List[float]],
    top_k: int,
    similarity_threshold: float,
    sam_threshold: float,
    polygon_simplify_epsilon: float,
    pic_id: Optional[str] = None,
    query_names: Optional[List[str]] = None,
    similar_mode: str = "concat_prompt",
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    if similar_mode not in SIMILAR_MODES:
        raise ValueError(f"similar_mode must be one of: {', '.join(sorted(SIMILAR_MODES))}")

    if similar_mode == "same_image_prompt":
        start_time = time.perf_counter()
        normalized_pic_id = (pic_id or "").strip() or uuid.uuid4().hex[:16]
        lock_ctx = MODEL_LOCK if SERIALIZE_MODEL_ACCESS else contextlib.nullcontext()
        autocast_ctx = _inference_autocast_context()
        with torch.inference_mode(), lock_ctx, autocast_ctx:
            result = _run_same_image_prompt_query(
                reference_image,
                reference_bnd_points,
                top_k,
                similarity_threshold,
                sam_threshold,
                polygon_simplify_epsilon,
                normalized_pic_id,
            )
        if query_names:
            result["query_name"] = query_names[0]
        if EMPTY_CUDA_CACHE_EACH_REQUEST and torch.cuda.is_available():
            torch.cuda.empty_cache()
        total_processing_ms = int((time.perf_counter() - start_time) * 1000)
        query_result = dict(result)
        result["batch_size"] = 1
        result["query_results"] = [query_result]
        result["processing_time_ms"] = total_processing_ms
        return result

    if not query_images:
        raise ValueError("At least one query image is required")

    start_time = time.perf_counter()
    normalized_pic_id = (pic_id or "").strip() or uuid.uuid4().hex[:16]
    query_results: List[Dict[str, Any]] = []
    lock_ctx = MODEL_LOCK if SERIALIZE_MODEL_ACCESS else contextlib.nullcontext()
    autocast_ctx = _inference_autocast_context()
    with torch.inference_mode(), lock_ctx, autocast_ctx:
        reference_ctx = _prepare_similar_reference_context(reference_image, reference_bnd_points, top_k)
        for index, query_image in enumerate(query_images):
            query_pic_id = normalized_pic_id if len(query_images) == 1 else f"{normalized_pic_id}_{index + 1}"
            if similar_mode == "concat_prompt":
                one_result = _run_concat_prompt_query(
                    reference_ctx,
                    query_image,
                    top_k,
                    similarity_threshold,
                        sam_threshold,
                        polygon_simplify_epsilon,
                        query_pic_id,
                        prompt,
                    )
            else:
                one_result = _run_similar_query_with_reference_context(
                    reference_ctx,
                    query_image,
                    top_k,
                    similarity_threshold,
                    sam_threshold,
                    polygon_simplify_epsilon,
                    query_pic_id,
                )
            if query_names and index < len(query_names):
                one_result["query_name"] = query_names[index]
            query_results.append(one_result)

    if EMPTY_CUDA_CACHE_EACH_REQUEST and torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_processing_ms = int((time.perf_counter() - start_time) * 1000)
    if len(query_results) == 1:
        single = dict(query_results[0])
        single["batch_size"] = 1
        single["query_results"] = query_results
        single["processing_time_ms"] = total_processing_ms
        return single

    return {
        "model": MODEL_LABEL,
        "pic_id": normalized_pic_id,
        "success": True,
        "batch_size": len(query_results),
        "query_results": query_results,
        "reference_bnd_points": query_results[0].get("reference_bnd_points") if query_results else None,
        "reference_box_auto_generated": False,
        "top_k": int(top_k),
        "similarity_threshold": float(similarity_threshold),
        "sam_threshold": float(sam_threshold),
        "similar_mode": similar_mode,
        "total_num_matches": sum(int(item.get("num_matches", 0)) for item in query_results),
        "total_num_candidates": sum(int(item.get("num_candidates", 0)) for item in query_results),
        "reference_result_image": query_results[0].get("reference_result_image") if query_results else None,
        "created": int(time.time()),
        "processing_time_ms": total_processing_ms,
    }


def extract_api_key(authorization: Optional[str], x_api_key: Optional[str]) -> Optional[str]:
    if x_api_key:
        return x_api_key.strip()

    if authorization:
        auth_value = authorization.strip()
        if auth_value.lower().startswith("bearer "):
            return auth_value[7:].strip()

    return None


async def require_api_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Dict[str, Any]:
    raw_key = extract_api_key(authorization, x_api_key)
    if not raw_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Use Authorization: Bearer <api_key> or X-API-Key header.",
        )

    metadata = api_key_manager.validate_key(raw_key)
    if not metadata:
        raise HTTPException(status_code=401, detail="Invalid or expired API key")

    return metadata


async def require_admin_key(key_metadata: Dict[str, Any] = Depends(require_api_key)) -> Dict[str, Any]:
    if key_metadata.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin API key required")
    return key_metadata


class SegmentationRequest(BaseModel):
    pic_id: str = Field(..., min_length=1, max_length=128, description="Client image ID")
    image_base64: str = Field(..., description="Base64 image string or data URL")
    prompt: str = Field(..., description="Classes separated by ';' or ','")
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    polygon_simplify_epsilon: float = Field(default=2.0, ge=0.0, le=50.0)


class BoxSegmentationRequest(BaseModel):
    pic_id: str = Field(..., min_length=1, max_length=128, description="Client image ID")
    image_base64: str = Field(..., description="Base64 image string or data URL")
    bnd_points: Optional[Union[List[float], List[List[float]]]] = Field(
        default=None,
        description="[x, y, w, h] or [[x, y, w, h], ...]",
    )
    bnd_points_list: Optional[List[List[float]]] = Field(
        default=None,
        min_length=1,
        description="Multiple boxes: [[x, y, w, h], ...]",
    )
    polygon_simplify_epsilon: float = Field(default=2.0, ge=0.0, le=50.0)

    @model_validator(mode="after")
    def normalize_bnd_points(self) -> "BoxSegmentationRequest":
        payload = self.bnd_points_list if self.bnd_points_list is not None else self.bnd_points
        if payload is None:
            raise ValueError("Either bnd_points or bnd_points_list is required")

        normalized = normalize_box_segmentation_inputs(payload)
        self.bnd_points = normalized
        self.bnd_points_list = None
        return self


class SimilarObjectRequest(BaseModel):
    pic_id: str = Field(..., min_length=1, max_length=128, description="Client image ID")
    reference_image_base64: str = Field(..., description="Base64 sample image string or data URL")
    query_image_base64: Optional[str] = Field(
        default=None,
        description="Base64 query image string or data URL. Kept for single-image compatibility.",
    )
    query_image_base64_list: Optional[List[str]] = Field(
        default=None,
        description="Multiple base64 query image strings or data URLs.",
    )
    reference_bnd_points: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Sample object box [x, y, w, h] in reference image",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Optional text prompt describing the similar target. Used with the reference box in concat_prompt mode.",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Candidate boxes to verify in query image")
    similarity_threshold: float = Field(default=0.6, ge=-1.0, le=1.0)
    sam_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    polygon_simplify_epsilon: float = Field(default=2.0, ge=0.0, le=50.0)
    similar_mode: Literal["feature_match", "concat_prompt", "same_image_prompt"] = Field(default="concat_prompt")

    @model_validator(mode="after")
    def normalize_query_images(self) -> "SimilarObjectRequest":
        if self.query_image_base64_list is None:
            if self.query_image_base64 is None:
                if self.similar_mode == "same_image_prompt":
                    self.query_image_base64_list = []
                    return self
                raise ValueError("Either query_image_base64 or query_image_base64_list is required")
            self.query_image_base64_list = [self.query_image_base64]
        elif len(self.query_image_base64_list) == 0 and self.similar_mode != "same_image_prompt":
            raise ValueError("query_image_base64_list must contain at least one image")
        return self


class ConcatPromptSegmentationRequest(SimilarObjectRequest):
    prompt: str = Field(..., min_length=1, description="Text prompt used with the reference box.")
    similar_mode: Literal["concat_prompt"] = Field(default="concat_prompt")


class CreateApiKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    role: Literal["client", "admin"] = Field(default="client")
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=3650)


def infer_auth_scope(route: APIRoute) -> str:
    dependant = getattr(route, "dependant", None)
    dependencies = getattr(dependant, "dependencies", []) if dependant is not None else []
    dependency_calls = {
        getattr(dep, "call", None).__name__
        for dep in dependencies
        if getattr(dep, "call", None) is not None
    }

    if "require_admin_key" in dependency_calls:
        return "admin_api_key"
    if "require_api_key" in dependency_calls:
        return "api_key"
    return "public"


def get_available_api_routes() -> List[Dict[str, Any]]:
    routes: List[Dict[str, Any]] = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue

        methods = sorted(method for method in route.methods if method not in {"HEAD", "OPTIONS"})
        if not methods:
            continue

        routes.append(
            {
                "name": route.name,
                "path": route.path,
                "methods": methods,
                "auth": infer_auth_scope(route),
                "summary": route.summary or "",
            }
        )

    routes.sort(key=lambda item: (item["path"], ",".join(item["methods"])))
    return routes


@app.get("/api-list.json")
async def available_api_list_json() -> Dict[str, Any]:
    routes = get_available_api_routes()
    return {
        "success": True,
        "count": len(routes),
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
        "data": routes,
    }


@app.get("/api-list", response_class=HTMLResponse)
@app.get("/apis", response_class=HTMLResponse)
async def available_api_list_page() -> HTMLResponse:
    routes = get_available_api_routes()
    rows = []
    for route in routes:
        methods = html.escape(", ".join(route["methods"]))
        path = html.escape(route["path"])
        auth = html.escape(route["auth"])
        summary = html.escape(route["summary"])
        rows.append(
            "<tr>"
            f"<td>{methods}</td>"
            f"<td><code>{path}</code></td>"
            f"<td>{auth}</td>"
            f"<td>{summary}</td>"
            "</tr>"
        )

    content = f"""
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>SAM3 可用 API 列表</title>
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ border: 1px solid #ddd; text-align: left; padding: 8px; vertical-align: top; }}
        th {{ background: #f7f7f7; }}
        code {{ background: #f3f3f3; padding: 2px 6px; border-radius: 4px; }}
      </style>
    </head>
    <body>
      <h2>SAM3 可用 API 列表</h2>
      <p>
        当前共 <strong>{len(routes)}</strong> 个接口。
        <a href="/docs">Swagger 文档</a> |
        <a href="/openapi.json">OpenAPI JSON</a> |
        <a href="/api-list.json">API 列表 JSON</a>
      </p>
      <table>
        <thead>
          <tr>
            <th>Methods</th>
            <th>Path</th>
            <th>Auth</th>
            <th>Summary</th>
          </tr>
        </thead>
        <tbody>
          {"".join(rows)}
        </tbody>
      </table>
    </body>
    </html>
    """
    return HTMLResponse(content)


@app.post("/v1/segmentations")
async def create_segmentation(
    payload: SegmentationRequest,
    _: Dict[str, Any] = Depends(require_api_key),
) -> Dict[str, Any]:
    try:
        image = decode_base64_image(payload.image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if SAVE_UPLOADS:
        save_upload_image(image, "base64_input.jpg")

    try:
        async with INFERENCE_SEMAPHORE:
            result = await asyncio.to_thread(
                run_detection_pipeline,
                image,
                payload.prompt,
                payload.confidence_threshold,
                payload.polygon_simplify_epsilon,
                payload.pic_id,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return result


@app.post("/v1/box-segmentations")
async def create_box_segmentation(
    payload: BoxSegmentationRequest,
    _: Dict[str, Any] = Depends(require_api_key),
) -> Dict[str, Any]:
    try:
        image = decode_base64_image(payload.image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if SAVE_UPLOADS:
        save_upload_image(image, "box_seg_input.jpg")

    try:
        async with INFERENCE_SEMAPHORE:
            result = await asyncio.to_thread(
                run_box_segmentation_pipeline,
                image,
                payload.bnd_points,
                payload.polygon_simplify_epsilon,
                payload.pic_id,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Box segmentation failed: {exc}") from exc

    return result


@app.post("/v1/similar-object-segmentations")
async def create_similar_object_segmentation(
    payload: SimilarObjectRequest,
    _: Dict[str, Any] = Depends(require_api_key),
) -> Dict[str, Any]:
    try:
        reference_image = decode_base64_image(payload.reference_image_base64)
        query_images = [decode_base64_image(item) for item in payload.query_image_base64_list or []]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if SAVE_UPLOADS:
        save_upload_image(reference_image, "similar_reference_input.jpg")
        for index, query_image in enumerate(query_images, start=1):
            save_upload_image(query_image, f"similar_query_input_{index}.jpg")

    try:
        async with INFERENCE_SEMAPHORE:
            result = await asyncio.to_thread(
                run_similar_object_batch_pipeline,
                reference_image,
                query_images,
                payload.reference_bnd_points,
                payload.top_k,
                payload.similarity_threshold,
                payload.sam_threshold,
                payload.polygon_simplify_epsilon,
                payload.pic_id,
                None,
                payload.similar_mode,
                payload.prompt,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Similar object segmentation failed: {exc}") from exc

    return result


@app.post("/v1/concat-prompt-segmentations")
async def create_concat_prompt_segmentation(
    payload: ConcatPromptSegmentationRequest,
    _: Dict[str, Any] = Depends(require_api_key),
) -> Dict[str, Any]:
    payload.similar_mode = "concat_prompt"
    return await create_similar_object_segmentation(payload, _)


@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(default=None),
    confidence: float = Form(0.3),
    polygon_simplify_epsilon: float = Form(2.0),
    pic_id: Optional[str] = Form(default=None),
) -> Dict[str, Any]:
    try:
        contents = await file.read()
        if len(contents) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=400, detail=f"Image payload too large. Max bytes: {MAX_IMAGE_BYTES}")

        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid upload image: {exc}") from exc

    if SAVE_UPLOADS:
        save_upload_image(image, file.filename or "upload.jpg")

    try:
        async with INFERENCE_SEMAPHORE:
            result = await asyncio.to_thread(
                run_detection_pipeline,
                image,
                prompt,
                confidence,
                polygon_simplify_epsilon,
                pic_id,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return result


@app.post("/similar-detect")
async def detect_similar_objects(
    request: Request,
    reference_file: UploadFile = File(...),
    reference_bnd_points: str = Form(...),
    prompt: str = Form(...),
    top_k: int = Form(5),
    similarity_threshold: float = Form(0.6),
    sam_threshold: float = Form(0.2),
    polygon_simplify_epsilon: float = Form(2.0),
    similar_mode: str = Form("concat_prompt"),
    pic_id: Optional[str] = Form(default=None),
) -> Dict[str, Any]:
    try:
        reference_contents = await reference_file.read()
        if len(reference_contents) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=400, detail=f"Reference image too large. Max bytes: {MAX_IMAGE_BYTES}")
        is_same_image_mode = similar_mode == "same_image_prompt"
        form = await request.form()
        query_files = [
            item
            for item in form.getlist("query_file")
            if getattr(item, "filename", None) and hasattr(item, "read")
        ]
        if not query_files and not is_same_image_mode:
            raise HTTPException(status_code=400, detail="At least one query image is required")

        reference_image = Image.open(io.BytesIO(reference_contents)).convert("RGB")
        query_images: List[Image.Image] = []
        query_names: List[str] = []
        for index, one_file in enumerate(query_files, start=1):
            query_contents = await one_file.read()
            if len(query_contents) > MAX_IMAGE_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Query image #{index} too large. Max bytes: {MAX_IMAGE_BYTES}",
                )
            query_images.append(Image.open(io.BytesIO(query_contents)).convert("RGB"))
            query_names.append(one_file.filename or f"query_upload_{index}.jpg")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid upload image: {exc}") from exc

    try:
        parsed_reference_bnd_points = parse_optional_bnd_points_text(reference_bnd_points)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if parsed_reference_bnd_points is None:
        raise HTTPException(status_code=400, detail="reference_bnd_points is required, format: x,y,w,h")

    if SAVE_UPLOADS:
        save_upload_image(reference_image, reference_file.filename or "reference_upload.jpg")
        for index, query_image in enumerate(query_images):
            query_name = query_names[index] if index < len(query_names) else f"query_upload_{index + 1}.jpg"
            save_upload_image(query_image, query_name)

    try:
        async with INFERENCE_SEMAPHORE:
            result = await asyncio.to_thread(
                run_similar_object_batch_pipeline,
                reference_image,
                query_images,
                parsed_reference_bnd_points,
                top_k,
                similarity_threshold,
                sam_threshold,
                polygon_simplify_epsilon,
                pic_id,
                query_names,
                similar_mode,
                prompt,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Similar object detection failed: {exc}") from exc

    return result


@app.post("/ui/api-keys")
async def ui_create_api_key(payload: CreateApiKeyRequest) -> Dict[str, Any]:
    created = api_key_manager.create_key(
        name=payload.name,
        role=payload.role,
        expires_in_days=payload.expires_in_days,
    )
    return {
        "success": True,
        "message": "API key created. Save api_key now; it will not be shown again.",
        "data": created,
    }


@app.get("/ui/api-keys")
async def ui_list_api_keys() -> Dict[str, Any]:
    return {
        "success": True,
        "data": api_key_manager.list_keys(),
    }


@app.delete("/ui/api-keys/{key_id}")
async def ui_delete_api_key(key_id: str) -> Dict[str, Any]:
    try:
        deleted = api_key_manager.delete_key(key_id, protect_last_admin=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not deleted:
        raise HTTPException(status_code=404, detail="API key not found")

    return {
        "success": True,
        "message": f"API key {key_id} deleted",
    }


@app.post("/v1/api-keys")
async def create_api_key(
    payload: CreateApiKeyRequest,
    _: Dict[str, Any] = Depends(require_admin_key),
) -> Dict[str, Any]:
    created = api_key_manager.create_key(
        name=payload.name,
        role=payload.role,
        expires_in_days=payload.expires_in_days,
    )
    return {
        "success": True,
        "message": "API key created. Save api_key now; it will not be shown again.",
        "data": created,
    }


@app.get("/v1/api-keys")
async def list_api_keys(_: Dict[str, Any] = Depends(require_admin_key)) -> Dict[str, Any]:
    return {
        "success": True,
        "data": api_key_manager.list_keys(),
    }


@app.delete("/v1/api-keys/{key_id}")
async def delete_api_key(
    key_id: str,
    _: Dict[str, Any] = Depends(require_admin_key),
) -> Dict[str, Any]:
    try:
        deleted = api_key_manager.delete_key(key_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not deleted:
        raise HTTPException(status_code=404, detail="API key not found")

    return {
        "success": True,
        "message": f"API key {key_id} deleted",
    }


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(index_file.read_text(encoding="utf-8"))

    return HTMLResponse("<h3>SAM3 API is running</h3><p>Use /docs for OpenAPI documentation.</p>")


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "model": MODEL_LABEL,
        "device": device,
        "max_concurrent_inferences": MAX_CONCURRENT_INFERENCES,
        "model_access_mode": "serialized" if SERIALIZE_MODEL_ACCESS else "parallel",
        "backend": "ultralytics",
        "ultralytics_imgsz": ULTRALYTICS_IMGSZ,
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("SAM3_HOST", "0.0.0.0")
    port = int(os.getenv("SAM3_PORT", "8006"))

    print("Starting SAM3 Detection Server...")
    print(f"Server URL: http://{host}:{port}")
    print(f"OpenAPI docs: http://{host}:{port}/docs")

    uvicorn.run(app, host=host, port=port)
