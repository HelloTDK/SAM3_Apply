#!/bin/bash
set -euo pipefail

# SAM3 Web Service Startup Script (Ascend OM / Ultralytics)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "Starting SAM3 Detection Web Service..."
echo "======================================="

export SAM3_BACKEND="${SAM3_BACKEND:-om}"
if [ "${SAM3_BACKEND}" = "om" ]; then
  export ARGOS_DEVICE_TYPE="cpu"
  if [ -z "${SAM3_PYTHON:-}" ] && [ -x /opt/python310_venv/bin/python ]; then
    SAM3_PYTHON=/opt/python310_venv/bin/python
  fi
else
  # Initialize conda and activate the target environment.
  if command -v conda >/dev/null 2>&1; then
      CONDA_BASE="$(conda info --base 2>/dev/null)"
      # shellcheck source=/dev/null
      source "${CONDA_BASE}/etc/profile.d/conda.sh"
  elif [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
      # shellcheck source=/dev/null
      source "/root/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "/expdata/miniconda/etc/profile.d/conda.sh" ]; then
      # shellcheck source=/dev/null
      source "/expdata/miniconda/etc/profile.d/conda.sh"
  else
      echo "ERROR: conda.sh not found. Please install conda or adjust this script."
      exit 1
  fi

  echo "Activating conda env: sam3"
  conda activate sam3

fi
SAM3_PYTHON="${SAM3_PYTHON:-python}"

# Create necessary directories
mkdir -p uploads results static
export PYTHONPATH="${SCRIPT_DIR}/_vendor${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRIPT_DIR}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

"${SAM3_PYTHON}" - <<'PY'
import os
from importlib import import_module
modules = ["fastapi", "uvicorn", "multipart", "cv2", "matplotlib", "requests"]
modules += ["acl", "tokenizers"] if os.getenv("SAM3_BACKEND") == "om" else ["ultralytics.models.sam"]
for name in modules:
    try:
        import_module(name)
    except ImportError as exc:
        raise SystemExit(f"ERROR: missing service dependency {name}: {exc}") from exc
PY

export SAM3_HOST="${SAM3_HOST:-0.0.0.0}"
export SAM3_PORT="${SAM3_PORT:-8006}"

# Force Argos translation to run fully offline in Docker.
# These defaults can still be overridden from the shell before running this script.
export ARGOS_PACKAGES_DIR="${ARGOS_PACKAGES_DIR:-${SCRIPT_DIR}/argos-packages}"
export SAM3_ARGOS_MODEL_DIR="${SAM3_ARGOS_MODEL_DIR:-${SCRIPT_DIR}/argos_models}"
export SAM3_ARGOS_AUTO_INSTALL="${SAM3_ARGOS_AUTO_INSTALL:-0}"
export SAM3_ARGOS_FORCE_OFFLINE="${SAM3_ARGOS_FORCE_OFFLINE:-1}"
export SAM3_CUDA_CLEANUP_AFTER_REQUEST="${SAM3_CUDA_CLEANUP_AFTER_REQUEST:-1}"
export SAM3_IDLE_MODEL_UNLOAD_SECONDS="${SAM3_IDLE_MODEL_UNLOAD_SECONDS:-0}"
export SAM3_CUDA_CLEANUP_LOG="${SAM3_CUDA_CLEANUP_LOG:-0}"

# Optional: set local stanza resources if the zh->en package exists.
if [ -d "${ARGOS_PACKAGES_DIR}" ]; then
    STANZA_LOCAL_DIR="$(find "${ARGOS_PACKAGES_DIR}" -maxdepth 2 -type d -name "stanza" | head -n 1 || true)"
    if [ -n "${STANZA_LOCAL_DIR}" ]; then
        export STANZA_RESOURCES_DIR="${STANZA_RESOURCES_DIR:-${STANZA_LOCAL_DIR}}"
    fi
fi

echo "======================================="
echo "Server URL: http://localhost:${SAM3_PORT}"
echo "OpenAPI docs: http://localhost:${SAM3_PORT}/docs"
echo "BACKEND=${SAM3_BACKEND} PYTHON=${SAM3_PYTHON}"
echo "ARGOS_DEVICE_TYPE=${ARGOS_DEVICE_TYPE:-auto}"
echo "ARGOS_PACKAGES_DIR=${ARGOS_PACKAGES_DIR}"
echo "SAM3_ARGOS_MODEL_DIR=${SAM3_ARGOS_MODEL_DIR}"
echo "SAM3_ARGOS_AUTO_INSTALL=${SAM3_ARGOS_AUTO_INSTALL}"
echo "SAM3_ARGOS_FORCE_OFFLINE=${SAM3_ARGOS_FORCE_OFFLINE}"
echo "SAM3_CUDA_CLEANUP_AFTER_REQUEST=${SAM3_CUDA_CLEANUP_AFTER_REQUEST}"
echo "SAM3_IDLE_MODEL_UNLOAD_SECONDS=${SAM3_IDLE_MODEL_UNLOAD_SECONDS}"
echo "SAM3_CUDA_CLEANUP_LOG=${SAM3_CUDA_CLEANUP_LOG}"
if [ -n "${STANZA_RESOURCES_DIR:-}" ]; then
    echo "STANZA_RESOURCES_DIR=${STANZA_RESOURCES_DIR}"
fi
echo "======================================="

exec "${SAM3_PYTHON}" sam_app.py
