#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv/phyprobe"
# perception_models requires Python >=3.11, unlike the main Conda environment.
if [[ ! -f "$VENV_DIR/pyvenv.cfg" ]]; then
    PYTHON="${PHYPROBE_PYTHON:-python}"
    "$PYTHON" -c 'import sys; assert sys.version_info >= (3, 11), "Set PHYPROBE_PYTHON to a Python >=3.11 interpreter"'
    "$PYTHON" -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/python" - "$VENV_DIR" <<'CHECK'
from pathlib import Path
import sys
expected = Path(sys.argv[1]).resolve()
assert sys.version_info >= (3, 11), 'PhyProbe requires Python >=3.11'
assert Path(sys.prefix).resolve() == expected and sys.prefix != sys.base_prefix
assert 'include-system-site-packages = false' in (expected / 'pyvenv.cfg').read_text().lower()
CHECK
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT_DIR/cache/pip}"
"$VENV_DIR/bin/python" -m pip install --no-compile -r "$ROOT_DIR/scripts/phyprobe/requirements.txt" -e "$ROOT_DIR/third_party/PhyProbe"
"$VENV_DIR/bin/python" "$ROOT_DIR/scripts/phyprobe/check_environment.py"
printf 'Run with: %s\n' "$VENV_DIR/bin/python"
