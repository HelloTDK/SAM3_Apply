#!/bin/bash
set -euo pipefail

# SAM3 Web Service Startup Script (Conda)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "Starting SAM3 Detection Web Service..."
echo "======================================="

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

SAM3_CONDA_ENV="${SAM3_CONDA_ENV:-base}"
echo "Activating conda env: ${SAM3_CONDA_ENV}"
conda activate "${SAM3_CONDA_ENV}"

# Docker 非交互 shell 不读取 ~/.bashrc，需要在启动脚本中显式暴露 CUDA/TRT 动态库。
export SAM3_CUDA_HOME="${SAM3_CUDA_HOME:-/usr/local/cuda-12.8}"
export SAM3_TRT_ROOT="${SAM3_TRT_ROOT:-/usr/local/TensorRT-10.8.0.43}"
export PATH="${SAM3_CUDA_HOME}/bin:${SAM3_TRT_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${SAM3_TRT_ROOT}/lib:${SAM3_TRT_ROOT}/targets/x86_64-linux-gnu/lib:${SAM3_CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Create necessary directories
mkdir -p uploads results static
export PYTHONPATH="${SCRIPT_DIR}/_vendor${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRIPT_DIR}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

if [ -z "${SAM3_TRT_ENGINE_DIR:-}" ]; then
    # 默认只使用当前程序内置的 TensorRT engine。
    export SAM3_TRT_ENGINE_DIR="${SCRIPT_DIR}/engines"
fi

python - <<'PY'
try:
    import cupy  # noqa: F401
    import tensorrt  # noqa: F401
    import tokenizers  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "ERROR: TensorRT/CuPy/tokenizers are required in the current environment.\n"
        f"Import error: {exc}"
    )
PY

export SAM3_HOST="0.0.0.0"
export SAM3_PORT="8006"

# Force Argos translation to run fully offline in Docker.
# These defaults can still be overridden from the shell before running this script.
export ARGOS_PACKAGES_DIR="${ARGOS_PACKAGES_DIR:-${SCRIPT_DIR}/argos-packages}"
export SAM3_ARGOS_MODEL_DIR="${SAM3_ARGOS_MODEL_DIR:-${SCRIPT_DIR}/argos_models}"
export SAM3_ARGOS_AUTO_INSTALL="${SAM3_ARGOS_AUTO_INSTALL:-0}"
export SAM3_ARGOS_FORCE_OFFLINE="${SAM3_ARGOS_FORCE_OFFLINE:-1}"
export SAM3_CUDA_CLEANUP_AFTER_REQUEST="${SAM3_CUDA_CLEANUP_AFTER_REQUEST:-1}"

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
echo "SAM3_TRT_ENGINE_DIR=${SAM3_TRT_ENGINE_DIR}"
echo "SAM3_TRT_ROOT=${SAM3_TRT_ROOT}"
echo "SAM3_CUDA_HOME=${SAM3_CUDA_HOME}"
echo "ARGOS_PACKAGES_DIR=${ARGOS_PACKAGES_DIR}"
echo "SAM3_ARGOS_MODEL_DIR=${SAM3_ARGOS_MODEL_DIR}"
echo "SAM3_ARGOS_AUTO_INSTALL=${SAM3_ARGOS_AUTO_INSTALL}"
echo "SAM3_ARGOS_FORCE_OFFLINE=${SAM3_ARGOS_FORCE_OFFLINE}"
echo "SAM3_CUDA_CLEANUP_AFTER_REQUEST=${SAM3_CUDA_CLEANUP_AFTER_REQUEST}"
if [ -n "${STANZA_RESOURCES_DIR:-}" ]; then
    echo "STANZA_RESOURCES_DIR=${STANZA_RESOURCES_DIR}"
fi
echo "======================================="

exec python sam_app.py
