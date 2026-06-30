#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR_REAL="$(readlink -f "${SCRIPT_DIR}" 2>/dev/null || printf '%s' "${SCRIPT_DIR}")"
cd "${SCRIPT_DIR}"

PID_FILE="${SAM3_PID_FILE:-${SCRIPT_DIR}/sam3_offline.pid}"
STOP_TIMEOUT_SECONDS="${SAM3_STOP_TIMEOUT_SECONDS:-20}"
SERVICE_PORT="${SAM3_PORT:-8006}"

stop_pid() {
  local pid="$1"
  local reason="${2:-matched process}"

  if [ -z "${pid}" ] || [ "${pid}" = "$$" ]; then
    return 0
  fi

  if ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi

  echo "Stopping SAM3 offline service (${reason}), PID=${pid}"
  kill "${pid}" 2>/dev/null || true

  local waited=0
  while kill -0 "${pid}" 2>/dev/null; do
    if [ "${waited}" -ge "${STOP_TIMEOUT_SECONDS}" ]; then
      echo "PID=${pid} did not stop after ${STOP_TIMEOUT_SECONDS}s, forcing kill."
      kill -9 "${pid}" 2>/dev/null || true
      break
    fi
    sleep 1
    waited=$((waited + 1))
  done
}

find_service_pids_by_cwd() {
  for proc_dir in /proc/[0-9]*; do
    local pid cwd cwd_real cmdline
    pid="${proc_dir##*/}"
    [ "${pid}" = "$$" ] && continue
    cwd="$(readlink "${proc_dir}/cwd" 2>/dev/null || true)"
    cwd_real="$(readlink -f "${proc_dir}/cwd" 2>/dev/null || true)"
    if [ "${cwd}" != "${SCRIPT_DIR}" ] \
      && [ "${cwd}" != "${SCRIPT_DIR_REAL}" ] \
      && [ "${cwd_real}" != "${SCRIPT_DIR}" ] \
      && [ "${cwd_real}" != "${SCRIPT_DIR_REAL}" ]; then
      continue
    fi
    cmdline="$(tr '\0' ' ' < "${proc_dir}/cmdline" 2>/dev/null || true)"
    case "${cmdline}" in
      *"python sam_app.py"*|*"python3 sam_app.py"*)
        echo "${pid}"
        ;;
    esac
  done
}

find_service_pids_by_port() {
  ss -ltnp "sport = :${SERVICE_PORT}" 2>/dev/null \
    | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
    | sort -u
}

stop_existing_service() {
  if [ ! -f "${PID_FILE}" ]; then
    echo "No PID file found: ${PID_FILE}"
    return 0
  fi

  local pid
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [ -z "${pid}" ]; then
    echo "Empty PID file, removing: ${PID_FILE}"
    rm -f "${PID_FILE}"
    return 0
  fi

  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "Stale PID file found, removing: ${PID_FILE}"
    rm -f "${PID_FILE}"
    return 0
  fi

  stop_pid "${pid}" "pid file"
  rm -f "${PID_FILE}"
}

stop_existing_service

for pid in $(find_service_pids_by_cwd); do
  stop_pid "${pid}" "cwd=${SCRIPT_DIR}"
done

for pid in $(find_service_pids_by_port); do
  stop_pid "${pid}" "port ${SERVICE_PORT}"
done

rm -f "${PID_FILE}"

echo "Starting SAM3 offline service again..."
exec "${SCRIPT_DIR}/start_offline.sh"
