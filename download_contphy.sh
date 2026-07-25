#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Default target directory if none is provided
TARGET_DIR="./datasets/ContPhy"

# Display usage instructions
usage() {
    echo "Usage: $0 [-d target_directory]"
    echo "  -d : Specify the target directory where files will be downloaded and unzipped (Default: $TARGET_DIR)"
    exit 1
}

# Parse command-line flags
while getopts "d:h" opt; do
    case "$opt" in
        d) TARGET_DIR="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

echo "========================================================"
echo "Target Directory: $TARGET_DIR"
echo "========================================================"

# 1. Validate unzip utility exists
if ! command -v unzip > /dev/null 2>&1; then
    echo "[-] Error: 'unzip' utility is not found in PATH."
    exit 1
fi

# 2. Ensure safe huggingface_hub version
pip install "huggingface_hub<1.0" --quiet

# 3. Setup absolute pathways
mkdir -p "$TARGET_DIR"
TARGET_DIR_ABS=$(cd "$TARGET_DIR" && pwd)

echo "[+] Beginning download pipeline with fixed skipping and auto-cleanup..."

# 4. Run inline Python to track extraction states cleanly via markers
python -c "
import os
import subprocess
from huggingface_hub import hf_hub_download, list_repo_files

repo_id = 'zzcnewly/ContPhy_Dataset'
target_dir = '$TARGET_DIR_ABS'

print('[*] Fetching file list from repository...')
all_files = list_repo_files(repo_id=repo_id, repo_type='dataset')
zip_files = sorted([f for f in all_files if f.endswith('.zip')])

for file_name in zip_files:
    zip_basename = os.path.basename(file_name)
    # We write a tiny hidden .done file when extraction successfully finishes 
    marker_file = os.path.join(target_dir, f'.{zip_basename}.done')
    
    # If the marker exists, we know this zip was completely downloaded and extracted already
    if os.path.exists(marker_file):
        print(f'[-] Dataset from \"{zip_basename}\" is already extracted. Skipping download.')
        continue

    print(f'\n[*] Downloading: {file_name}')
    
    # Download file sequentially
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=file_name,
        repo_type='dataset',
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )
    
    print(f'[+] Download complete. Extracting and cleaning up {zip_basename} in background...')
    
    # Unzip -> Touch the hidden verification marker -> Remove the raw zip file safely
    cmd = f'cd {target_dir} && unzip -oq {zip_basename} && touch .{zip_basename}.done && rm {zip_basename}'
    
    # Dispatched to background execution
    subprocess.Popen(cmd, shell=True)

print('\n[+] Processing loop complete!')
print('[*] Note: Active zip extractions are wrapping up tasks in the background.')
"

echo "========================================================"
echo "[+] Pipeline complete. Target folder: $TARGET_DIR_ABS"
echo "========================================================"