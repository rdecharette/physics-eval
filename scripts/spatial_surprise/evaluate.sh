#!/usr/bin/env bash
# Submit one GPU job for all variants, loading V-JEPA once.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAG="${TAG:-s4-v0.5}"
JOB_PROFILE="${JOB_PROFILE:-salsa}"
VIDEO="${VIDEO:-$ROOT_DIR/cache/datasets_variants/masked/$TAG/videos.txt}"
MAXFRAMES="${MAXFRAMES:-150}"
WINDOW_SIZE="${WINDOW_SIZE:-16}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-8}"
STRIDE="${STRIDE:-8}"
MODE="${MODE:-mean}"
VJEPA_HUB_DIR="${VJEPA_HUB_DIR:-$ROOT_DIR/cache/vjepa2}"
OUTPUT_PATH="${OUTPUT_PATH:-$ROOT_DIR/output/spatial-surprise/$TAG/scores/vith/$MODE/mf-${MAXFRAMES}_w-${WINDOW_SIZE}_c-${CONTEXT_FRAMES}_s-${STRIDE}/surprises.csv}"
[[ -f "$VIDEO" && -f "$VJEPA_HUB_DIR/hubconf.py" ]] || {
    echo 'Missing video list or V-JEPA source; see scripts/spatial_surprise/README.md' >&2
    exit 1
}
# Validate input list before creating or submitting a job.
python - "$VIDEO" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); lines=p.read_text().splitlines()
assert lines and len(lines)==len(set(lines)), 'Empty or duplicate video list'
assert all(Path(x).is_absolute() and Path(x).is_file() for x in lines), 'Missing input video'
print(f'Validated {len(lines)} input videos')
PY
source "$ROOT_DIR/scripts/job_profiles.sh"
job_profile_config "$JOB_PROFILE"
# Salsa's conda info command can hang in plugin discovery. Use its installed
# environment directly; the evaluator needs no conda CLI during a batch job.
if [[ "$JOB_PROFILE" == salsa ]]; then
    PYTHON_ENV="${PYTHON_ENV:-$HOME/miniconda3/envs/physics-eval}"
    [[ -x "$PYTHON_ENV/bin/python" ]] || { echo "Missing Python environment: $PYTHON_ENV" >&2; exit 1; }
    printf -v q_env '%q' "$PYTHON_ENV"
    JOB_HEADER=("export PATH=$q_env/bin:\$PATH" "export CONDA_PREFIX=$q_env")
fi
[[ "$JOB_LAUNCHER" == sbatch ]] || { echo 'This launcher requires sbatch' >&2; exit 1; }
LOG_DIR="$ROOT_DIR/logs/spatial-surprise/$TAG"
mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_PATH")"
RUNNER="$(mktemp "$LOG_DIR/run_XXXXXXXX.sh")"
{
    printf '#!/usr/bin/env bash\nset -euo pipefail\necho "Starting spatial-surprise evaluation"\n'
    printf '%s\n' "${JOB_HEADER[@]}"
    printf 'cd %q\n' "$ROOT_DIR"
    for key in VIDEO MAXFRAMES WINDOW_SIZE CONTEXT_FRAMES STRIDE MODE VJEPA_HUB_DIR OUTPUT_PATH; do
        printf 'export %s=%q\n' "$key" "${!key}"
    done
    printf 'export PYTHONPATH=%q${PYTHONPATH:+:$PYTHONPATH}\n' "$VJEPA_HUB_DIR"
    printf 'export DIRECT_PATHS=1 EVAL_MAX=-1\n'
    printf 'exec 9>%q\nflock -n 9 || { echo "Scoring already running for this output" >&2; exit 1; }\n' "$OUTPUT_PATH.lock"
    cat <<'BODY'
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'GPU allocation is required'
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY
bash third_party/WMReward/test_vith.sh
python scripts/spatial_surprise/verify_scores.py --video-list "$VIDEO" --scores "$OUTPUT_PATH"
BODY
} > "$RUNNER"
ARGS=(--parsable --job-name="spatial_$TAG" --output="$LOG_DIR/%j.log" --error="$LOG_DIR/%j.log" "${JOB_PROFILE_ARGS[@]}" "$RUNNER")
if [[ "${1:-}" == --dry-run ]]; then
    printf 'Prepared runner: %s\n' "$RUNNER"
    printf 'sbatch '; printf '%q ' "${ARGS[@]}"; printf '\n'
    exit 0
fi
[[ $# == 0 ]] || { echo 'Usage: evaluate.sh [--dry-run]' >&2; exit 1; }
JOB_ID="$(sbatch "${ARGS[@]}")"
printf 'Submitted job %s\nScores: %s\nLog: %s/%s.log\n' "$JOB_ID" "$OUTPUT_PATH" "$LOG_DIR" "${JOB_ID%%;*}"
printf '%s\n' "$JOB_ID" > "$RUNNER.jobid"
