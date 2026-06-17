#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PID_FILE="${SAM3_PID_FILE:-${SCRIPT_DIR}/sam3_offline.pid}"
STOP_TIMEOUT_SECONDS="${SAM3_STOP_TIMEOUT_SECONDS:-20}"

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

  echo "Stopping SAM3 offline service, PID=${pid}"
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

  rm -f "${PID_FILE}"
}

stop_existing_service

echo "Starting SAM3 offline service again..."
exec "${SCRIPT_DIR}/start_offline.sh"
