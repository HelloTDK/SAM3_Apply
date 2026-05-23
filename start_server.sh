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

echo "Activating conda env: sam3"
conda activate sam3

# Create necessary directories
mkdir -p uploads results static
export PYTHONPATH="${SCRIPT_DIR}/_vendor${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRIPT_DIR}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

# Check if model exists
if [ ! -f "weights/sam3.pt" ]; then
    echo "ERROR: Model file 'weights/sam3.pt' not found."
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

export SAM3_HOST="0.0.0.0"
export SAM3_PORT="8006"

# Force Argos translation to run fully offline in Docker.
# These defaults can still be overridden from the shell before running this script.
export ARGOS_PACKAGES_DIR="${ARGOS_PACKAGES_DIR:-${SCRIPT_DIR}/argos-packages}"
export SAM3_ARGOS_MODEL_DIR="${SAM3_ARGOS_MODEL_DIR:-${SCRIPT_DIR}/argos_models}"
export SAM3_ARGOS_AUTO_INSTALL="${SAM3_ARGOS_AUTO_INSTALL:-0}"
export SAM3_ARGOS_FORCE_OFFLINE="${SAM3_ARGOS_FORCE_OFFLINE:-1}"

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
echo "ARGOS_PACKAGES_DIR=${ARGOS_PACKAGES_DIR}"
echo "SAM3_ARGOS_MODEL_DIR=${SAM3_ARGOS_MODEL_DIR}"
echo "SAM3_ARGOS_AUTO_INSTALL=${SAM3_ARGOS_AUTO_INSTALL}"
echo "SAM3_ARGOS_FORCE_OFFLINE=${SAM3_ARGOS_FORCE_OFFLINE}"
if [ -n "${STANZA_RESOURCES_DIR:-}" ]; then
    echo "STANZA_RESOURCES_DIR=${STANZA_RESOURCES_DIR}"
fi
echo "======================================="

exec python sam_app.py
