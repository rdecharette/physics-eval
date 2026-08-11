#!/usr/bin/env bash
set -euo pipefail
# set -x





DATASETS_DIR="./datasets"
LIST_FILE="./pisabench.txt"
OUTPUT_TAR="${1:-./datasets/pisabench.tar}"

DATASETS_DIR_ABS="$(realpath -m "$DATASETS_DIR")"
OUTPUT_TAR_ABS="$(realpath -m "$OUTPUT_TAR")"

mkdir -p "$(dirname "$OUTPUT_TAR_ABS")"

tmp_list="$(mktemp)"
while IFS= read -r entry; do
  [ -n "$entry" ] || continue
  case "$entry" in
    \#*) continue ;;
  esac

  if [[ "$entry" == /* ]]; then
    rel_path="${entry#"$DATASETS_DIR_ABS/"}"
  else
    rel_path="$entry"
    rel_path="${rel_path#./}"
    rel_path="${rel_path#datasets/}"
  fi

  echo "$rel_path" >> "$tmp_list"
done < "$LIST_FILE"

tar -C "$DATASETS_DIR_ABS" -cvf "$OUTPUT_TAR_ABS" -T "$tmp_list"
rm -f "$tmp_list"

echo "Created tar from $LIST_FILE at: $OUTPUT_TAR_ABS"

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
