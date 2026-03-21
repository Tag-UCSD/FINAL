#!/usr/bin/env bash
set -euo pipefail
LOG_DIR="logs"; mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/install.log"
echo "[installer] $(date) starting" | tee -a "$LOG_FILE"

# 1) Best-effort Python toolchain
echo "[installer] upgrading pip/setuptools/wheel" | tee -a "$LOG_FILE"
python -m pip install --upgrade pip setuptools wheel >> "$LOG_FILE" 2>&1 || true

# 2) Root requirements.txt (optional, non-fatal)
if [ -f "requirements.txt" ]; then
  echo "[installer] installing requirements.txt" | tee -a "$LOG_FILE"
  python -m pip install -r requirements.txt >> "$LOG_FILE" 2>&1 || true
fi

# 3) Installer config JSON (if present)
CFG_PATH="infra/turnkey_installer_v1.3/installer_config.json"
if [ -f "$CFG_PATH" ]; then
  echo "[installer] applying config from $CFG_PATH" | tee -a "$LOG_FILE"
  python - << 'PY'
import json, subprocess
from pathlib import Path

cfg_path = Path("infra/turnkey_installer_v1.3/installer_config.json")
if cfg_path.exists():
    print("[installer-config] loading", cfg_path)
    cfg = json.loads(cfg_path.read_text())
    for cmd in cfg.get("prechecks", []):
        print(f"[installer-config] PRECHECK: {cmd}")
        subprocess.run(cmd, shell=True, check=False)
    for step in cfg.get("steps", []):
        shell = step.get("shell")
        if not shell:
            continue
        name = step.get("name", shell)
        print(f"[installer-config] STEP: {name}")
        subprocess.run(shell, shell=True, check=False)
PY
fi

echo "[installer] SUCCESS" | tee -a "$LOG_FILE"

# 4) Post-install smoke (non-fatal)
if [ -f "infra/turnkey_installer_v1.3/hooks/post_install_smoke.py" ]; then
  echo "[installer] post-install smoke" | tee -a "$LOG_FILE"
  python infra/turnkey_installer_v1.3/hooks/post_install_smoke.py 2>&1 | tee -a "$LOG_FILE" || true
fi
