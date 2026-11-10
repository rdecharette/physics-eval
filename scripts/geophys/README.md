# GeoPhys

Editable upstream source is duplicated in `third_party/GeoPhys`, with no clone or
submodule. See `third_party/README.md` for its pinned revision and original license.
This integration follows the upstream [per-video scoring instructions](https://github.com/ChristianInterno/GeoPhys#use-geophys-on-your-own-videos)
and the [paper](https://arxiv.org/abs/2606.20707). It uses `geophys.score`'s
single-encoder rating, rather than the paper's multi-encoder pairwise detection
or best-of-N ensemble. The original implementations of all modes are vendored.

## Setup

Activate your physics-eval environment (Python 3.10+ recommended), then:

```bash
bash scripts/geophys/setup.sh
hf auth login
```

Setup creates (or reuses) `.venv/geophys` using the active Python and installs
GeoPhys and its dependencies there. The Conda environment's packages, including
VBench's Transformers 4.33.2, are unchanged. No venv activation is necessary.
Run setup separately on each machine; virtualenvs should not be copied between hosts.
The scoring CLI checks `sys.prefix` before importing metric dependencies and
prints setup/run instructions if the wrong Python is used. `--help` works with
any Python.

Accept access to [DINOv3 ViT-L/16](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
with the account used by `hf auth login` (or supply `HF_TOKEN`). Weights download
on first use; pre-download them on a network-enabled node if necessary:

```bash
hf download facebook/dinov3-vitl16-pretrain-lvd1689m
```

Default scoring needs only DINOv3. Optional `cornet` and `vonenet` backbones also
need the packages in upstream's installation instructions. Setup pins Transformers
to `>=4.56,<5` for DINOv3 support; upstream's generic lower bound is insufficient.

## Run

From the repository root:

```bash
.venv/geophys/bin/python scripts/geophys/compute_geophys.py intphys2_possible --eval-max 500
.venv/geophys/bin/python scripts/geophys/compute_geophys.py intphys_possible intphys_impossible --variant original
.venv/geophys/bin/python scripts/geophys/compute_geophys.py PhysBench --reference /path/to/real_videos --eval-max 500
```

Inputs are the existing `<dataset>.txt` files. Relative original paths resolve
against the list directory. The default variant is `512p_30fps`, read from
`cache/datasets_variants/<variant>/`; use `--variant original` for source videos.
Paths are shuffled with seed 42 before applying `--eval-max`, matching VBench
and surprise. Zero, negative, or omitted limits mean all videos.

By default results are written to:

```text
output/geophys/<mode>/<variant>_n<limit|All>/dinov3_l18_direct/<dataset>_geophys.score
```

`<mode>` is `independent` for a single dataset, `combined` for multiple datasets,
or `ref-<reference name>` when using a reference. `--output-root` overrides the
root above the variant and encoder subdirectories.

This is a tab-separated table: `video`, `geophys.score`, the five raw signals
(`curv`, `angcons`, `speedcv`, `accel`, `prederr`), and `error`. Scores range
from 0 to 1; **higher means less plausible**. Failed videos have an empty score
and an error; the command exits nonzero, with valid results still saved.
An invalid reference video aborts scoring instead of changing the population.
`run.json` records settings, selected paths, reference paths and failures.
Use `--output-dir` to preserve separate experiments; reruns replace result files.

Upstream defaults are preserved: DINOv3 block 18, mean patch-token trajectories,
direct 224x224 resize and ImageNet normalization, at most 60 frames using
upstream's frame-stride rule. No additional FPS conversion occurs in this script.
The five signals are converted to percentiles and averaged **across the complete
selected population**, never separately per video or extraction batch.
Without a reference, a dataset's mean score is approximately 0.5 by construction:
separately normalized dataset means are not a meaningful comparison. Pass multiple
dataset names in one invocation or use the same `--reference` for every run.
A reference can be a recursive video directory or a text list of original paths;
`--eval-max` and `--variant` do not alter the reference. One target video requires
a reference. Missing/invalid target videos are excluded from normalization and
reported, so fix failures before comparing completed runs.

Signals are cached atomically in `cache/geophys/signals`, keyed by resolved video
path, size/mtime, settings and a digest of the vendored Python source. Interrupted
runs reuse successful extraction; failed videos are retried. Scores are always
recomputed against the current population. `--force` recomputes signals.
`--cache-dir`, `--backbone`, `--layer`, `--prep`, and `--device` are configurable.
A GPU is recommended for extraction.

## Cluster jobs

```bash
JOB_PROFILE=salsa EVAL_MAX=500 bash scripts/geophys/job_intphys2_geophys.sh
JOB_PROFILE=salsa EVAL_MAX=500 bash scripts/geophys/job_all_combined.sh
JOB_PROFILE=salsa EVAL_MAX=500 bash scripts/geophys/job_all_ref-pisabench.sh
JOB_PROFILE=jeanzay-a100-t3 REFERENCE=/path/to/real_videos bash scripts/geophys/job_all.sh
```

Dataset launchers match the VBench dataset mappings and use `scripts/job_hub.sh`.
They accept `EVAL_MAX`, `VARIANT`, `REFERENCE`, `BACKBONE`, `LAYER`, `PREP`, and
`DEVICE`, `REFERENCE_DATASET`, `OUTPUT_ROOT`, and `OUTPUT_DIR`. Run setup using the Conda environment selected by the job profile.
Jobs explicitly invoke `.venv/geophys/bin/python` and reject a missing venv
before submission.
`job_all_combined.sh` submits one job with all eight datasets, defaulting to 500
videos per dataset and `512p_30fps`. All selected videos share percentile
normalization (or the supplied `REFERENCE`). It writes separate `.score` files
per dataset and one `run.json`, replacing existing files at the same output path.
Cached signals are reused when available.

`job_all_ref-pisabench.sh` scores all eight datasets in one job against the full
`pisabench.txt` list, using the same `VARIANT` as the targets. `EVAL_MAX` limits
targets per dataset, not the reference. Outputs go under
`output/geophys/ref-pisabench/<variant>_n<limit>/<encoder>/`.
`job_all_combined.sh` defaults to `output/geophys/combined/`, and independent
jobs default to `output/geophys/independent/`, with the same subdirectories.
Existing results in the old layout are not moved.

`job_all.sh` submits separate jobs, so set a common `REFERENCE` when comparing
dataset means. IntPhys launchers select the possible split, matching VBench;
the impossible split can be run directly with the Python script.

## Validation

```bash
.venv/geophys/bin/python -m unittest discover -s scripts/geophys -p 'test_*.py'
```

Tests exercise integration with synthetic signals and the actual upstream
percentile function; they do not download model weights.
