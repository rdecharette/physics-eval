# Spatial surprise — step 1

Generate masks from the repository root using the existing `physics-eval`
environment (Python with Pillow):

```bash
conda run -n physics-eval python scripts/spatial_surprise/generate_masks.py
```

Defaults are `--height 256 --width 256 --scale 8 --overlap 0.5`.
The script derives 32×32 windows with 16-pixel strides and writes 225 masks to
`data/mask/s8-v0.5/`. Output paths default to this repository regardless of the
working directory. `--output-root` selects a different parent directory.

Each PNG is an 8-bit grayscale image with 255 meaning valid and 0 meaning masked.
Names encode the exact normalized top-left coordinates, `y/height_x/width.png`.
For example, `0.0625_0.125.png` starts at pixel `(16,32)` in the default image.
`manifest.json` records every mask's half-open pixel bounds and exact normalized
coordinates. `preview/contact_sheet.png` shows all masks with pixel-coordinate
labels. The 225 root-level PNG files are exclusively masks.

Parameters must give integer window sizes, integer positive strides, and exact
edge coverage. Normalized coordinates must have terminating decimal expansions
so filenames remain exact. Nonempty output directories are rejected to prevent
stale or accidentally overwritten artifacts; use another output root to rerun.

Default rectangles and strides align with the 16-pixel V-JEPA spatial patch grid.
Other accepted parameter choices are not automatically patch-aligned.

## Manual checkpoint

Inspect the masks and contact sheet before starting step 2. No masked videos or
Slurm jobs are generated in this stage. Subsequent stages will use constant gray
outside the retained rectangle, no extra border, unchanged temporal-surprise
scoring, and a heatmap normalized by the sum of covering masks. Their directory
tag will also be `s8-v0.5`.
