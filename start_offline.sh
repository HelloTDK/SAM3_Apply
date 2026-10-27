#!/usr/bin/env bash
set -euo pipefail

case "$#:${1:-}" in
  1:--background) SAM3_FOREGROUND=0 ;;
  0:) ;;
  *) echo "Usage: $0 [--background]" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

activate_sam3_env() {
  local env_name="${SAM3_CONDA_ENV:-base}"
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

# Docker 非交互 shell 不读取 ~/.bashrc，需要在启动脚本中显式暴露 CUDA/TRT 动态库。
export SAM3_CUDA_HOME="${SAM3_CUDA_HOME:-/usr/local/cuda-12.8}"
export SAM3_TRT_ROOT="${SAM3_TRT_ROOT:-/usr/local/TensorRT-10.8.0.43}"
export PATH="${SAM3_CUDA_HOME}/bin:${SAM3_TRT_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${SAM3_TRT_ROOT}/lib:${SAM3_TRT_ROOT}/targets/x86_64-linux-gnu/lib:${SAM3_CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

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
# TCP and HTTP are started by the same Python process; override these ports here or via env.
export SAM3_TCP_HOST="${SAM3_TCP_HOST:-0.0.0.0}"
export SAM3_TCP_PORT="${SAM3_TCP_PORT:-6063}"
export SAM3_TCP_POLYGON_EPSILON="${SAM3_TCP_POLYGON_EPSILON:-0.5}"
if [ -z "${SAM3_TRT_ENGINE_DIR:-}" ]; then
  # 默认只使用当前程序内置的 TensorRT engine。
  export SAM3_TRT_ENGINE_DIR="${SCRIPT_DIR}/engines"
fi
export SAM3_NATIVE_ENGINE_DIR="${SAM3_NATIVE_ENGINE_DIR:-${SAM3_TRT_ENGINE_DIR}/native_interactive}"

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

for engine_name in shared-image-encoder text-encoder geometry-encoder decoder; do
  if ! find "${SAM3_TRT_ENGINE_DIR}" -maxdepth 1 -type f -name "${engine_name}*.engine" -print -quit | grep -q .; then
    echo "ERROR: missing ${engine_name} engine in ${SAM3_TRT_ENGINE_DIR}"
    exit 1
  fi
done
for engine_name in interactive-decoder-first interactive-decoder-refine; do
  if [ ! -f "${SAM3_NATIVE_ENGINE_DIR}/${engine_name}.engine" ]; then
    echo "ERROR: missing ${engine_name}.engine in ${SAM3_NATIVE_ENGINE_DIR}"
    exit 1
  fi
done
if [ ! -f "${SAM3_TRT_ENGINE_DIR}/tokenizer.json" ]; then
  echo "ERROR: missing tokenizer.json in ${SAM3_TRT_ENGINE_DIR}"
  exit 1
fi

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
export SAM3_TRT_NMS_IOU="${SAM3_TRT_NMS_IOU:-0.7}"
export SAM3_CUDA_CLEANUP_AFTER_REQUEST="${SAM3_CUDA_CLEANUP_AFTER_REQUEST:-1}"

echo "Starting SAM3 TensorRT HTTP + TCP offline service..."
echo "HOST=${SAM3_HOST} PORT=${SAM3_PORT}"
echo "TCP_HOST=${SAM3_TCP_HOST} TCP_PORT=${SAM3_TCP_PORT}"
echo "TCP_POLYGON_EPSILON=${SAM3_TCP_POLYGON_EPSILON}"
echo "TRT_ENGINE_DIR=${SAM3_TRT_ENGINE_DIR}"
echo "NATIVE_ENGINE_DIR=${SAM3_NATIVE_ENGINE_DIR}"
echo "TRT_ROOT=${SAM3_TRT_ROOT}"
echo "CUDA_HOME=${SAM3_CUDA_HOME}"
echo "ARGOS_PACKAGES_DIR=${ARGOS_PACKAGES_DIR}"
echo "ARGOS_MODEL_DIR=${SAM3_ARGOS_MODEL_DIR}"
echo "ARGOS_FORCE_OFFLINE=${SAM3_ARGOS_FORCE_OFFLINE}"
echo "TRT_NMS_IOU=${SAM3_TRT_NMS_IOU}"
echo "CUDA_CLEANUP_AFTER_REQUEST=${SAM3_CUDA_CLEANUP_AFTER_REQUEST}"

export SAM3_PID_FILE="${SAM3_PID_FILE:-${SCRIPT_DIR}/sam3_offline.pid}"
export SAM3_LOG_FILE="${SAM3_LOG_FILE:-${SCRIPT_DIR}/start_offline.log}"
# --background overrides Docker's inherited foreground setting for interactive launches.
export SAM3_FOREGROUND="${SAM3_FOREGROUND:-0}"
export SAM3_STARTUP_TIMEOUT_SECONDS="${SAM3_STARTUP_TIMEOUT_SECONDS:-180}"

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
nohup python sam_app.py >> "${SAM3_LOG_FILE}" 2>&1 </dev/null &
new_pid="$!"
echo "${new_pid}" > "${SAM3_PID_FILE}"

if ! SAM3_SERVICE_PID="${new_pid}" python - <<'PY'
import http.client
import os
import socket
import time


def loopback_for_wildcard(host):
    # Probe wildcard listeners through loopback, while preserving explicit bind addresses.
    return "127.0.0.1" if host in {"", "0.0.0.0", "::"} else host


pid = int(os.environ["SAM3_SERVICE_PID"])
http_host = loopback_for_wildcard(os.environ["SAM3_HOST"])
tcp_host = loopback_for_wildcard(os.environ["SAM3_TCP_HOST"])
http_port = int(os.environ["SAM3_PORT"])
tcp_port = int(os.environ["SAM3_TCP_PORT"])
deadline = time.monotonic() + int(os.environ["SAM3_STARTUP_TIMEOUT_SECONDS"])
last_error = "waiting for listeners"

while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit("SAM3 process exited during startup")
    # A failed child can remain as a zombie until this launcher reaps it.
    try:
        with open(f"/proc/{pid}/stat") as process_stat:
            if process_stat.read().rsplit(")", 1)[1].split()[0] == "Z":
                raise SystemExit("SAM3 process exited during startup")
    except FileNotFoundError:
        raise SystemExit("SAM3 process exited during startup")

    try:
        connection = http.client.HTTPConnection(http_host, http_port, timeout=2)
        connection.request("GET", "/health")
        response = connection.getresponse()
        response.read()
        connection.close()
        if response.status != 200:
            raise RuntimeError(f"HTTP health check returned {response.status}")

        with socket.create_connection((tcp_host, tcp_port), timeout=2):
            pass
        print(f"HTTP and TCP are ready on {http_port} and {tcp_port}.")
        break
    except (OSError, TimeoutError, RuntimeError) as exc:
        last_error = str(exc)
        time.sleep(1)
else:
    raise SystemExit(f"Timed out waiting for HTTP {http_port} and TCP {tcp_port}: {last_error}")
PY
then
  echo "ERROR: SAM3 HTTP + TCP service failed to start. Last log lines:"
  kill "${new_pid}" 2>/dev/null || true
  wait "${new_pid}" 2>/dev/null || true
  tail -n 80 "${SAM3_LOG_FILE}" || true
  rm -f "${SAM3_PID_FILE}"
  exit 1
fi

echo "SAM3 offline service started."
echo "PID=${new_pid}"
echo "URL=http://${SAM3_HOST}:${SAM3_PORT}"
