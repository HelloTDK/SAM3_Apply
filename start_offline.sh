#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

activate_sam3_env() {
  local env_name="${SAM3_CONDA_ENV:-sam3}"
  local conda_sh=""

  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [ -n "${conda_base}" ] && [ -f "${conda_base}/etc/profile.d/conda.sh" ]; then
      conda_sh="${conda_base}/etc/profile.d/conda.sh"
    fi
  fi

  if [ -z "${conda_sh}" ] && [ -f "/expdata/miniconda/etc/profile.d/conda.sh" ]; then
    conda_sh="/expdata/miniconda/etc/profile.d/conda.sh"
  elif [ -z "${conda_sh}" ] && [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    conda_sh="/root/miniconda3/etc/profile.d/conda.sh"
  elif [ -z "${conda_sh}" ] && [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    conda_sh="/opt/conda/etc/profile.d/conda.sh"
  fi

  if [ -z "${conda_sh}" ]; then
    echo "ERROR: conda.sh not found. Set SAM3_CONDA_ENV or install conda."
    exit 1
  fi

  # shellcheck source=/dev/null
  source "${conda_sh}"
  echo "Activating conda env: ${env_name}"
  conda activate "${env_name}"
}

activate_sam3_env

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
export SAM3_CUDA_CLEANUP_AFTER_REQUEST="${SAM3_CUDA_CLEANUP_AFTER_REQUEST:-1}"
export SAM3_IDLE_MODEL_UNLOAD_SECONDS="${SAM3_IDLE_MODEL_UNLOAD_SECONDS:-0}"
export SAM3_CUDA_CLEANUP_LOG="${SAM3_CUDA_CLEANUP_LOG:-0}"

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
echo "CUDA_CLEANUP_AFTER_REQUEST=${SAM3_CUDA_CLEANUP_AFTER_REQUEST}"
echo "IDLE_MODEL_UNLOAD_SECONDS=${SAM3_IDLE_MODEL_UNLOAD_SECONDS}"
echo "CUDA_CLEANUP_LOG=${SAM3_CUDA_CLEANUP_LOG}"

export SAM3_PID_FILE="${SAM3_PID_FILE:-${SCRIPT_DIR}/sam3_offline.pid}"
export SAM3_LOG_FILE="${SAM3_LOG_FILE:-${SCRIPT_DIR}/start_offline.log}"
export SAM3_FOREGROUND="${SAM3_FOREGROUND:-0}"

if [ "${SAM3_FOREGROUND}" = "1" ]; then
  echo "Running in foreground mode."
  exec python sam_app.py
fi

if [ -f "${SAM3_PID_FILE}" ]; then
  old_pid="$(cat "${SAM3_PID_FILE}" 2>/dev/null || true)"
  if [ -n "${old_pid}" ] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "SAM3 offline service is already running."
    echo "PID=${old_pid}"
    echo "PID_FILE=${SAM3_PID_FILE}"
    echo "LOG_FILE=${SAM3_LOG_FILE}"
    exit 0
  fi
  rm -f "${SAM3_PID_FILE}"
fi

echo "Running in background mode."
echo "PID_FILE=${SAM3_PID_FILE}"
echo "LOG_FILE=${SAM3_LOG_FILE}"
printf '\n[%s] Starting SAM3 offline service\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "${SAM3_LOG_FILE}"
nohup python sam_app.py >> "${SAM3_LOG_FILE}" 2>&1 &
new_pid="$!"
echo "${new_pid}" > "${SAM3_PID_FILE}"

sleep 1
if ! kill -0 "${new_pid}" 2>/dev/null; then
  echo "ERROR: SAM3 offline service failed to start. Last log lines:"
  tail -n 80 "${SAM3_LOG_FILE}" || true
  rm -f "${SAM3_PID_FILE}"
  exit 1
fi

echo "SAM3 offline service started."
echo "PID=${new_pid}"
echo "URL=http://${SAM3_HOST}:${SAM3_PORT}"
