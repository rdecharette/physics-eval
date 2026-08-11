#!/usr/bin/env bash
set -euo pipefail
set -x

JZ_WORK="/lustre/fswork/projects/rech/pab/ual15br"
JZ_SCRATCH="/lustre/fsn1/projects/rech/pab/ual15br"

SRC_DIR="."

DST_SERVER="ual15br@localhost"
DST_DIR="${DST_SERVER}:${JZ_WORK}/physics-eval"

CACHE_TRANSFERS=(
  "${TORCH_HOME:-$HOME/.cache/torch}|${JZ_SCRATCH}/.cache/torch"
  "$HOME/.cache/clip|${JZ_SCRATCH}/.cache/clip"
  "$HOME/.cache/vbench|${JZ_SCRATCH}/.cache/vbench"
)

echo "Transfer main files.."
rsync -avP \
  -e "ssh -p 2222" \
  --exclude='/cache/' \
  --exclude='/tmp/' \
  --exclude='/.git/' \
  --exclude='/logs/' \
  --exclude='/.vscode/' \
  --exclude='/datasets/' \
  --exclude='/output/' \
  --exclude='__pycache__' \
  "$SRC_DIR/" "$DST_DIR"

echo "Transfer datasets tars.."
rsync -avP \
  -e "ssh -p 2222" \
  --include='pisabench.tar' \
  --exclude='*/' \
  --exclude='*' \
  "$SRC_DIR/datasets/" "$DST_DIR/datasets/"

for transfer in "${CACHE_TRANSFERS[@]}"; do
  IFS='|' read -r local_dir remote_dir <<< "$transfer"

  echo -e "\nTransfer ${local_dir} -> ${remote_dir}"
  ssh -p 2222 "$DST_SERVER" "mkdir -p \"$remote_dir\""
  rsync -avP \
    -e "ssh -p 2222" \
    --out-format='%n -> %o %f' \
    "$local_dir/" "$DST_SERVER:${remote_dir}/"
done

