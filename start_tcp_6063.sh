#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Match the existing HTTP launcher's CUDA/TRT paths without changing the active environment.
export SAM3_CUDA_HOME="${SAM3_CUDA_HOME:-/usr/local/cuda-12.8}"
export SAM3_TRT_ROOT="${SAM3_TRT_ROOT:-/usr/local/TensorRT-10.8.0.43}"
export PATH="${SAM3_CUDA_HOME}/bin:${SAM3_TRT_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${SAM3_TRT_ROOT}/lib:${SAM3_TRT_ROOT}/targets/x86_64-linux-gnu/lib:${SAM3_CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${SCRIPT_DIR}/_vendor${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRIPT_DIR}/.matplotlib}"
cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" ann_tcp_6063.py
