#!/usr/bin/env bash
set -euo pipefail
# set -x


# Tar the files listed in a LIST_FILE (paths relative to the physics-eval root,
# e.g. datasets/mydata/my/file/video.mp4) into an uncompressed archive with paths
# relative to datasets/ (e.g. mydata/my/file/video.mp4). Symlinked videos are
# dereferenced (-h) so the tar contains the actual file content, not the link.
# The output tar is named after LIST_FILE and written to datasets/, e.g.
# NewtPhys.txt -> datasets/NewtPhys.tar
tar_from_list() {
  local list_file="$1"
  local out_tar="datasets/$(basename "$list_file" .txt).tar"
  local rel_list
  rel_list="$(mktemp)"
  sed -e 's#^datasets/##' "$list_file" > "$rel_list"
  tar -chf "$out_tar" -C datasets -T "$rel_list"
  rm -f "$rel_list"
}

# Example:
tar_from_list NewtPhys.txt