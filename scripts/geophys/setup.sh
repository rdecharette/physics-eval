#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv/geophys"
# Use the active Conda Python to create an isolated environment.
if [[ ! -f "$VENV_DIR/pyvenv.cfg" ]]; then
    python -m venv "$VENV_DIR"
fi
# Refuse an existing environment that exposes the parent environment's packages.
"$VENV_DIR/bin/python" - "$VENV_DIR" <<'CHECK'
from pathlib import Path
import sys
expected = Path(sys.argv[1]).resolve()
config = (expected / "pyvenv.cfg").read_text().lower()
if Path(sys.prefix).resolve() != expected or sys.prefix == sys.base_prefix:
    raise SystemExit(f"Error: {expected} is not a valid GeoPhys virtualenv.")
if "include-system-site-packages = false" not in config:
    raise SystemExit(f"Error: recreate {expected} without --system-site-packages.")
CHECK
# DINOv3 requires a newer Transformers than upstream's generic minimum.
"$VENV_DIR/bin/python" -m pip install -e "$ROOT_DIR/third_party/GeoPhys" 'transformers>=4.56,<5'
printf 'GeoPhys installed. Run with: %s\n' "$VENV_DIR/bin/python"
