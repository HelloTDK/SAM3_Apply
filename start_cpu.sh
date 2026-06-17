#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p uploads results static
export PYTHONPATH="${SCRIPT_DIR}/_vendor${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRIPT_DIR}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

export SAM3_HOST="${SAM3_HOST:-0.0.0.0}"
export SAM3_PORT="${SAM3_PORT:-8006}"

if [ -z "${SAM3_CHECKPOINT_PATH:-}" ]; then
  if [ -f "${SCRIPT_DIR}/weights/sam3.pt" ]; then
    export SAM3_CHECKPOINT_PATH="${SCRIPT_DIR}/weights/sam3.pt"
  elif [ -f "${SCRIPT_DIR}/weights/sam3.1_multiplex.pt" ]; then
    export SAM3_CHECKPOINT_PATH="${SCRIPT_DIR}/weights/sam3.1_multiplex.pt"
  else
    echo "ERROR: missing model file. expected one of:"
    echo "  - weights/sam3.pt (Ultralytics SAM3)"
    echo "  - weights/sam3.1_multiplex.pt"
    exit 1
  fi
fi

if [ ! -f "${SAM3_CHECKPOINT_PATH}" ]; then
  echo "ERROR: SAM3_CHECKPOINT_PATH does not exist: ${SAM3_CHECKPOINT_PATH}"
  exit 1
fi

python - <<'PY'
try:
    import ultralytics  # noqa: F401
    from ultralytics.models.sam import SAM3SemanticPredictor  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "ERROR: ultralytics with SAM3 support is required. "
        "Run: python -m pip install -U 'ultralytics>=8.4.51'\n"
        f"Import error: {exc}"
    )
PY

# CPU-only runtime.
export SAM3_DEVICE="cpu"
export ARGOS_DEVICE_TYPE="cpu"
export SAM3_INFER_DTYPE="float32"
export SAM3_ENABLE_AUTOCAST="0"
export SAM3_CUDA_CLEANUP_AFTER_REQUEST="0"
export SAM3_EMPTY_CUDA_CACHE_EACH_REQUEST="0"
export SAM3_CUDA_CLEANUP_LOG="${SAM3_CUDA_CLEANUP_LOG:-0}"
export SAM3_IDLE_MODEL_UNLOAD_SECONDS="${SAM3_IDLE_MODEL_UNLOAD_SECONDS:-0}"

# CPU inference is slow for SAM3; keep the default conservative.
export SAM3_MAX_CONCURRENT_INFERENCES="${SAM3_MAX_CONCURRENT_INFERENCES:-1}"
export SAM3_SERIALIZE_MODEL_ACCESS="${SAM3_SERIALIZE_MODEL_ACCESS:-1}"
export SAM3_ULTRALYTICS_IMGSZ="${SAM3_ULTRALYTICS_IMGSZ:-1036}"
export SAM3_ULTRALYTICS_IOU="${SAM3_ULTRALYTICS_IOU:-0.7}"

# Keep translation offline. Missing Argos resources should not block CPU startup,
# but Chinese prompts need these files to translate correctly.
export ARGOS_PACKAGES_DIR="${ARGOS_PACKAGES_DIR:-${SCRIPT_DIR}/argos-packages}"
export SAM3_ARGOS_MODEL_DIR="${SAM3_ARGOS_MODEL_DIR:-${SCRIPT_DIR}/argos_models}"
export SAM3_ARGOS_AUTO_INSTALL="${SAM3_ARGOS_AUTO_INSTALL:-0}"
export SAM3_ARGOS_FORCE_OFFLINE="${SAM3_ARGOS_FORCE_OFFLINE:-1}"
export SAM3_ARGOS_SOURCE_CODE="${SAM3_ARGOS_SOURCE_CODE:-zh}"
export SAM3_ARGOS_TARGET_CODE="${SAM3_ARGOS_TARGET_CODE:-en}"

if [ ! -f "${SAM3_ARGOS_MODEL_DIR}/translate-zh_en-1_9.argosmodel" ]; then
  echo "WARN: missing ${SAM3_ARGOS_MODEL_DIR}/translate-zh_en-1_9.argosmodel"
  echo "WARN: CPU service will still start, but Chinese prompt translation may be unavailable."
fi

export SAM3_API_KEYS_FILE="${SAM3_API_KEYS_FILE:-${SCRIPT_DIR}/api_keys.json}"

echo "Starting Ultralytics SAM3 CPU service..."
echo "HOST=${SAM3_HOST} PORT=${SAM3_PORT}"
echo "CHECKPOINT=${SAM3_CHECKPOINT_PATH}"
echo "DEVICE=${SAM3_DEVICE}"
echo "INFER_DTYPE=${SAM3_INFER_DTYPE}"
echo "ENABLE_AUTOCAST=${SAM3_ENABLE_AUTOCAST}"
echo "MAX_CONCURRENT_INFERENCES=${SAM3_MAX_CONCURRENT_INFERENCES}"
echo "SERIALIZE_MODEL_ACCESS=${SAM3_SERIALIZE_MODEL_ACCESS}"
echo "ARGOS_PACKAGES_DIR=${ARGOS_PACKAGES_DIR}"
echo "ARGOS_MODEL_DIR=${SAM3_ARGOS_MODEL_DIR}"
echo "ARGOS_FORCE_OFFLINE=${SAM3_ARGOS_FORCE_OFFLINE}"

exec python sam_app.py
