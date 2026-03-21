#!/usr/bin/env bash
#
# auto_install.sh
#
# Safer, logged bootstrap wrapper for install.sh.
# - strict mode
# - robust Python discovery
# - dual logging (console + logs/)
# - optional virtualenv creation
# - Docker preflight check
#
set -euo pipefail
echo "[Info] If you are a STUDENT running this script, please make sure you have read:"
echo "       STUDENT_START_HERE.md  (at the repo root)"
echo "       docs/ops/Student_Quickstart_v3.4.73.md"

# Resolve project root to this script's directory and cd into it
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# --- Python discovery ----------------------------------------------------
PYTHON_CMD="$(command -v python3 || true)"
if [ -z "$PYTHON_CMD" ]; then
  PYTHON_CMD="$(command -v python || true)"
fi

if [ -z "$PYTHON_CMD" ]; then
  echo "❌ Critical: No Python interpreter found (python3 or python)."
  exit 1
fi

# --- Dual logging: stdout/stderr to console and file --------------------
mkdir -p logs
LOG_FILE="logs/install_$(date +%Y%m%d_%H%M%S).log"
echo "📓 Logging install to ${LOG_FILE}"
exec > >(tee -i "${LOG_FILE}") 2>&1

# --- Virtualenv enforcement ---------------------------------------------
if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "⚠️  No virtualenv detected. Creating .venv..."
  "${PYTHON_CMD}" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "✅ Activated virtualenv at .venv"
else
  echo "✅ Using existing virtualenv at ${VIRTUAL_ENV}"
fi

# --- Install host-side dependencies for guard scripts -------------------
echo "📦 Installing host-side dependencies..."
pip install --quiet -r requirements-install.txt || {
  echo "❌ Failed to install host-side dependencies from requirements-install.txt"
  exit 1
}

# --- Docker preflight ----------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  echo "❌ Fatal: Docker is required but was not found in PATH."
  echo "    Please install Docker Desktop or a compatible Docker engine."
  exit 1
fi

# Optional: warn if docker-compose is missing (install.sh expects it)
if ! command -v docker-compose >/dev/null 2>&1; then
  echo "⚠️  docker-compose not found on host. install.sh will call 'docker-compose';"
  echo "    ensure it is installed or aliased appropriately."
fi

# --- Delegate to main installer -----------------------------------------
echo "🚀 Delegating to install.sh..."
bash install.sh
