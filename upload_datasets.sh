#!/usr/bin/env bash
set -euo pipefail

# Upload top-level datasets/*.tar files, excluding names starting with _old.
# Requires the Hugging Face CLI (hf) and write access to the dataset repo.
# Authenticate first with `hf auth login` or set HF_TOKEN.
# Usage: ./upload_datasets.sh [--dry-run]

dry_run=false
case "${1:-}" in
  --dry-run) dry_run=true ;;
  -h|--help)
    echo "Usage: $0 [--dry-run]"
    echo "Upload datasets/*.tar (excluding _old*) to astra-vision/physics-datasets."
    exit 0
    ;;
  "") ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
datasets_dir="$script_dir/datasets"
repo_id="astra-vision/physics-datasets"

if [[ ! -d "$datasets_dir" ]]; then
  echo "Dataset directory does not exist: $datasets_dir" >&2
  exit 1
fi
if ! "$dry_run" && ! command -v hf >/dev/null 2>&1; then
  echo "Missing Hugging Face CLI. Install with: pip install -U huggingface_hub" >&2
  exit 1
fi

shopt -s nullglob dotglob
count=0
for archive in "$datasets_dir"/*.tar; do
  name="${archive##*/}"
  [[ -f "$archive" && "$name" != _old* ]] || continue
  if "$dry_run"; then
    printf 'Would upload %s -> %s/%s\n' "$archive" "$repo_id" "$name"
  else
    printf 'Uploading %s to %s\n' "$name" "$repo_id"
    hf upload "$repo_id" "$archive" "$name" --repo-type dataset
  fi
  count=$((count + 1))
done

if (( count == 0 )); then
  echo "No eligible .tar files found in $datasets_dir."
elif "$dry_run"; then
  printf 'Dry run: %d archive(s) selected.\n' "$count"
else
  printf 'Uploaded %d archive(s).\n' "$count"
fi
