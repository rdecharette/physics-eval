#!/usr/bin/env bash
set -euo pipefail
# set -x

echo "Currently only doing NewtPhys dataset. YOU NEED TO REWRITE THE SCRIPT."
exit 0

SOURCE_DIR="${1:-./../physics-sim/output/sims/v4_bis}"
LINK_ROOT="${2:-./datasets/NewtPhys}"
OUTPUT_TAR="${3:-./datasets/NewtPhys.tar}"

SOURCE_DIR_ABS="$(realpath -m "$SOURCE_DIR")"
LINK_ROOT_ABS="$(realpath -m "$LINK_ROOT")"
OUTPUT_TAR_ABS="$(realpath -m "$OUTPUT_TAR")"

mkdir -p "$LINK_ROOT_ABS"
mkdir -p "$(dirname "$OUTPUT_TAR_ABS")"

# Rebuild the symlink tree to avoid stale links from previous runs.
rm -rf "$LINK_ROOT_ABS"
mkdir -p "$LINK_ROOT_ABS"

while IFS= read -r -d '' source_file; do
  relative_path="${source_file#"$SOURCE_DIR_ABS"/}"
  link_path="$LINK_ROOT_ABS/$relative_path"

  echo "Creating symlink: $link_path -> $source_file"
  mkdir -p "$(dirname "$link_path")"
  ln -sfn "$source_file" "$link_path"
done < <(find "$SOURCE_DIR_ABS" \( -type d -name '_invalid' -prune \) -o \( -type f -name '*_fps-25_render.mp4' -print0 \))
echo "Created symlink tree at: $LINK_ROOT_ABS"

# Archive dereferenced files so tar contains actual mp4 data, not symlink entries.
(
  cd "$(dirname "$LINK_ROOT_ABS")"
  tar -h -cvf "$OUTPUT_TAR_ABS" \
    "$(basename "$LINK_ROOT_ABS")"
)

echo "Created uncompressed tar archive with real files at: $OUTPUT_TAR_ABS"
