# Spatial surprise — masks and masked videos

Run from the repository root with the existing `physics-eval` environment
(Python with Pillow) and `ffmpeg` providing the `libx264` encoder.

## Step 1: masks

```bash
conda run -n physics-eval python scripts/spatial_surprise/generate_masks.py --scale 4 --overlap 0.5
```

The experiment uses `s4-v0.5`: 256×256 images, 64×64 valid windows, 32-pixel
strides, and 49 masks. The generator's historical default remains scale 8, so
pass `--scale 4` explicitly. Output paths default to this repository regardless
of the working directory. `--output-root` selects a different parent directory.

Each PNG is an 8-bit grayscale image with 255 meaning valid and 0 meaning masked.
Names encode the exact normalized top-left coordinates, `y/height_x/width.png`, with the same fixed decimal width throughout each mask set.
For example, `0.125_0.250.png` starts at pixel `(32,64)` in a 256×256 image.
`data/mask/s4-v0.5/manifest.json` records each mask's half-open pixel bounds and
exact normalized coordinates. Coordinates use at least three decimal places;
scale 4 uses three (e.g. `0.000_0.125.png`), while scale 8 needs four to
represent `0.0625` exactly. The manifest records `coordinate_decimals`.
Step 2 uses the same fixed-width coordinates in `masked_<y>_<x>.mp4`; configuration
tags such as `s4-v0.5` remain unchanged. `preview/contact_sheet.png` shows all masks with
pixel-coordinate labels. Only masks are root-level PNG files.

Parameters must give integer window sizes, positive integer strides, and exact
edge coverage. Normalized coordinates must have terminating decimal expansions.
Nonempty output directories are rejected; use another output root to rerun.
The scale-4 windows and strides align with the 16-pixel V-JEPA patch grid.
Other accepted parameter choices are not automatically patch-aligned.

## Step 2: masked videos

The source dataset must be accessible at `datasets/intphys` (a symlink is fine),
or use `--source-root` pointing at a root containing `datasets/intphys`.
With the masks already generated:

```bash
conda run -n physics-eval python scripts/spatial_surprise/generate_masked_videos.py
```

The default video is `datasets/intphys/dev/O1/02/1`; the default mask directory is
`data/mask/s4-v0.5`. For several videos, provide `--video-list path/to/videos.txt`,
with one repository-relative `datasets/...` video directory per line. Blank lines
and lines beginning with `#` are ignored. Supply `--mask-dir` to select another
mask set; the manifest dimensions and masks determine the output geometry.

Frames in each `scene/*.png` sequence are sorted numerically by filename,
converted to RGB, and resized with Pillow bilinear interpolation before masking.
For this example the resize is 288×288 to 256×256. Pixels outside the retained
rectangle become constant RGB `(128,128,128)`; there is no preserved border.

Each mask yields one video directly in the source-video output directory:

```text
cache/datasets_variants/masked/s4-v0.5/datasets/intphys/dev/O1/02/1/masked_0.000_0.000.mp4
```

Masked RGB frames are piped directly to ffmpeg at the original **25 FPS** timing.
No intermediate PNGs, per-mask subdirectories, or cached contact sheets are written.
MP4s use standard H.264 High profile (`libx264`, CRF 12, `yuv420p`), with explicit
BT.709 color conversion and limited-range metadata for browser playback.
MP4s introduce small conversion/compression differences
and chroma subsampling. RGB H.264 previously decoded correctly in decord but
displayed distorted colors in the browser player. The FPS filter converts
25 FPS to **30 FPS** by duplicating frames. A 100-frame, four-second input therefore
produces 120 encoded frames and remains four seconds long. Override `--source-fps`
if the source sequence has a different acquisition rate; `--target-fps` defaults
to 30 for the evaluator. Fractional rates are accepted. General durations are
preserved to target-frame precision. The original source PNGs and step-1 mask PNGs remain untouched.

The per-source-video `manifest.json` records source filenames, counts, dimensions,
rates, mask metadata, fill and encoding parameters, and all output paths. The
mask-set-level `videos.txt` lists absolute encoded video paths for subsequent
evaluation. It is written only after the full generation succeeds. The script
validates input masks and source frames before generating outputs and refuses
a nonempty mask-set destination. After an interrupted run, use a fresh output
root (for example `--output-root /tmp/masked-review`) to avoid mixing artifacts.

Run the small end-to-end tests (requires the environment's NumPy and decord,
plus ffmpeg):

```bash
conda run -n physics-eval python -m unittest discover -s tests -p test_spatial_masked_videos.py -v
```

## Manual checkpoint

The generation scripts stop for review before step 3 and never submit jobs.
Use the separate step-3 launcher below after approval. Later stages retain unchanged temporal-surprise
scoring, use a heatmap normalized by the sum of covering masks, and use the
`s4-v0.5` directory tag.

## Step 3: Slurm scoring

Prepare the V-JEPA source dependency locally (the inherited `.gitmodules` entry
is not a tracked gitlink in the source repository):

```bash
git clone --no-hardlinks /nfs/data/workspaces/rdechare/codes/physics-eval/third_party/WMReward/vjepa2 cache/vjepa2
```

The current dependency revision is `c2963a47433ecca0ad4f06ec28bcfa8cb5b5cefb`.
The launcher exports this source as both `VJEPA_HUB_DIR` and `PYTHONPATH`, avoiding
network code lookup; TorchHub still uses its cached `checkpoints/vith.pt` weights.
The existing `physics-eval` conda environment and Salsa resource profile are reused.
On Salsa, its `bin` directory is used directly to avoid stalled Conda plugin
discovery during batch startup; `PYTHON_ENV` can override the environment path.

```bash
bash scripts/spatial_surprise/evaluate.sh --dry-run
bash scripts/spatial_surprise/evaluate.sh
```

One GPU job runs `third_party/WMReward/test_vith.sh` against all 49 paths, loading
the model once. Defaults match that script: ViT-H, max frames 150, temporal
window 16, context 8, temporal stride 8, mean reduction, seed 42. Each current
video has 120 frames, so the frame cap does not truncate it. Spatial overlap
and temporal stride are separate parameters. Environment variables `TAG`,
`JOB_PROFILE`, `VIDEO`, `MAXFRAMES`, `WINDOW_SIZE`, `CONTEXT_FRAMES`, `STRIDE`,
`MODE`, `VJEPA_HUB_DIR`, and `OUTPUT_PATH` can override defaults.

Scores are CSV rows `video,surprise` using the actual masked-video paths:

```text
output/spatial-surprise/s4-v0.5/scores/vith/mean/mf-150_w-16_c-8_s-8/surprises.csv
```

Logs and generated job scripts are under `logs/spatial-surprise/s4-v0.5/`.
The direct-path evaluator option bypasses the legacy original-video cache rewrite.
Finite completed scores can be resumed; a file lock prevents concurrent writers.
The job fails if any requested score is missing or nonfinite. A final independent
check requires exactly one finite row per input video and writes
`surprises.summary.json`. It can also be run manually:

```bash
python scripts/spatial_surprise/verify_scores.py \
  --video-list cache/datasets_variants/masked/s4-v0.5/videos.txt \
  --scores output/spatial-surprise/s4-v0.5/scores/vith/mean/mf-150_w-16_c-8_s-8/surprises.csv
```

Stop after score review; heatmaps and overlay videos require the next approvals.
