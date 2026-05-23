#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p uploads results static
export PYTHONPATH="${SCRIPT_DIR}/_vendor${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRIPT_DIR}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

if [ ! -f "argos_models/translate-zh_en-1_9.argosmodel" ]; then
  echo "ERROR: missing translation model argos_models/translate-zh_en-1_9.argosmodel"
  exit 1
fi

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
        "ERROR: ultralytics with SAM3 support is required in conda env 'sam3'. "
        "Run: python -m pip install -U 'ultralytics>=8.4.51'\n"
        f"Import error: {exc}"
    )
PY

# Offline translation: only load/install local Argos model, no network access.
export ARGOS_PACKAGES_DIR="${ARGOS_PACKAGES_DIR:-${SCRIPT_DIR}/argos-packages}"
export SAM3_ARGOS_MODEL_DIR="${SAM3_ARGOS_MODEL_DIR:-${SCRIPT_DIR}/argos_models}"
export SAM3_ARGOS_AUTO_INSTALL="${SAM3_ARGOS_AUTO_INSTALL:-0}"
export SAM3_ARGOS_FORCE_OFFLINE="${SAM3_ARGOS_FORCE_OFFLINE:-1}"
export SAM3_ARGOS_SOURCE_CODE="${SAM3_ARGOS_SOURCE_CODE:-zh}"
export SAM3_ARGOS_TARGET_CODE="${SAM3_ARGOS_TARGET_CODE:-en}"

# Ensure API key storage stays inside deploy3.3 directory.
export SAM3_API_KEYS_FILE="${SAM3_API_KEYS_FILE:-${SCRIPT_DIR}/api_keys.json}"

# Optional runtime tuning.
export SAM3_MAX_CONCURRENT_INFERENCES="${SAM3_MAX_CONCURRENT_INFERENCES:-4}"
export SAM3_SERIALIZE_MODEL_ACCESS="${SAM3_SERIALIZE_MODEL_ACCESS:-1}"
export SAM3_INFER_DTYPE="${SAM3_INFER_DTYPE:-bfloat16}"
export SAM3_ENABLE_AUTOCAST="${SAM3_ENABLE_AUTOCAST:-1}"
export SAM3_ULTRALYTICS_IMGSZ="${SAM3_ULTRALYTICS_IMGSZ:-1036}"
export SAM3_ULTRALYTICS_IOU="${SAM3_ULTRALYTICS_IOU:-0.7}"

echo "Starting Ultralytics SAM3 offline service..."
echo "HOST=${SAM3_HOST} PORT=${SAM3_PORT}"
echo "CHECKPOINT=${SAM3_CHECKPOINT_PATH}"
echo "ARGOS_PACKAGES_DIR=${ARGOS_PACKAGES_DIR}"
echo "ARGOS_MODEL_DIR=${SAM3_ARGOS_MODEL_DIR}"
echo "ARGOS_FORCE_OFFLINE=${SAM3_ARGOS_FORCE_OFFLINE}"
echo "INFER_DTYPE=${SAM3_INFER_DTYPE}"
echo "ENABLE_AUTOCAST=${SAM3_ENABLE_AUTOCAST}"
echo "ULTRALYTICS_IMGSZ=${SAM3_ULTRALYTICS_IMGSZ}"
echo "ULTRALYTICS_IOU=${SAM3_ULTRALYTICS_IOU}"

exec python sam_app.py
