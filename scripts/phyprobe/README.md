# PhyProbe

Editable upstream source is in `third_party/PhyProbe` (source archive, no nested
Git repository), revision `f2457b78e5169d695ec2d6c3acc9f4d7ef37cd3a`.
The original MIT license, examples and ~4 MB trained head are retained.

Sources: [upstream instructions](https://github.com/vinesmsuic/PhyProbe),
[paper and implementation details](https://arxiv.org/html/2609.38377v1).
The metric uses the frozen PE-Core-G14-448 backbone and pretrained `pecore_paper`
head. `phyprobe.score` is the upstream output clamped to [0, 1]: **higher means
more physical violations**, not greater plausibility. No prompt, reference
population, training, sigmoid or dataset normalization is applied.

## Setup

The main Conda environment uses Python 3.10, Torch 2.4 and Transformers 4.33.
The pinned `perception_models` source requires Python >=3.11, NumPy 2.1.2 and
Transformers >=4.48. Setup therefore creates an isolated `.venv/phyprobe` and
installs PhyProbe editable from this repository. The parent environment is untouched. Setup verifies PE imports, the bundled
checkpoint, and real video decoding. It accepts only Decord 0.6.0's known stale
CPython-3.6 wheel-tag warning from `pip check`, after successful decoding; all
other dependency-check failures remain fatal.

```bash
# Use an available Python 3.11 or 3.12; a Python 3.10 venv cannot upgrade Python.
PHYPROBE_PYTHON=python3.12 bash scripts/phyprobe/setup.sh
```

The backbone weights (~9 GB) download automatically on first inference. Run once
on a machine with network access before using an offline compute node. Default
caches live under this repository's `cache/`; explicit `HF_HOME`, `TORCH_HOME`
and `PHYPROBE_CACHE` settings are respected. PE's backbone retains its upstream
license; see the installed perception_models package/source. Use a GPU with
BF16 support and sufficient memory (e.g. A100). Start with batch size 1.

## Run

```bash
.venv/phyprobe/bin/python scripts/phyprobe/compute_phyprobe.py intphys2_possible --eval-max 500
.venv/phyprobe/bin/python scripts/phyprobe/compute_phyprobe.py NewtPhys PhysBench --batch-size 2
.venv/phyprobe/bin/python scripts/phyprobe/compute_phyprobe.py --video /path/to/video.mp4 --variant original
# Score original dataset videos, optionally restricting to the training window:
.venv/phyprobe/bin/python scripts/phyprobe/compute_phyprobe.py contphy --variant original --clip-sec 3
```

Dataset inputs are the root `<dataset>.txt` files. As in GeoPhys, paths default to
`cache/datasets_variants/512p_30fps/`; `--variant original` uses the original paths.
`--video` is repeatable and uses its paths directly. List inputs use the existing
seed-42 shuffle before `--eval-max`; zero/negative or omitted means all videos.

Preprocessing is delegated unchanged to upstream: resample to the checkpoint's
16 fps, sample 48 frames uniformly, repeat the last frame for short videos, and
letterbox with gray padding to the checkpoint's resolution. The public API
samples across the full duration by default. `--clip-sec 3` restricts sampling to
the first three seconds, matching the training window. Dataset cache FPS and
PhyProbe sampling FPS are distinct. We intentionally do not override checkpoint
resolution or add CLIP normalization.

Outputs default to:
`output/phyprobe/<variant>_n<limit>/pecore_paper_<full|clip3s>/<dataset>_phyprobe.score`.
Each TSV contains `video`, `phyprobe.score`, `error`; direct video inputs use
`videos_phyprobe.score`. `<dataset>_run.json` records settings and source/checkpoint
hashes. Override the destination with `--output-dir`.

Scores are saved incrementally under `cache/phyprobe/scores`, keyed by video
path/size/mtime, source/checkpoint hashes and inference settings. Successful videos
are reused after interruption; `--force` recomputes. Failed batches retry each
video separately. Failed videos have an empty score and an error, and the command
exits nonzero. Dependency/model initialization failures abort immediately.
`--checkpoint` accepts an alternative local head checkpoint.

## Cluster launchers

```bash
JOB_PROFILE=salsa EVAL_MAX=500 bash scripts/phyprobe/job_all.sh
JOB_PROFILE=salsa BATCH_SIZE=1 bash scripts/phyprobe/job_newtphys_phyprobe.sh
```

These use the existing `scripts/job_hub.sh` and profiles with the dedicated venv.
Supported environment options: `EVAL_MAX`, `VARIANT`, `BATCH_SIZE`, `CLIP_SEC`,
`CHECKPOINT`, `DEVICE`, `CACHE_DIR`, `OUTPUT_DIR`, and `FORCE=1`. Logs are under
`logs/phyprobe`. Like GeoPhys, IntPhys launchers default to the possible split;
uncomment the impossible split in those scripts to submit it too. `job_all.sh`
defaults to 500 videos per dataset and `512p_30fps`.

## Checks

```bash
python -m unittest discover -s scripts/phyprobe -p 'test_*.py'
```

These test dataset selection, batching failures, output, resume/invalidation and
argument validation without downloading a model. A real inference smoke test is:

```bash
.venv/phyprobe/bin/python scripts/phyprobe/compute_phyprobe.py \
  --video third_party/PhyProbe/assets/task383-ref.mp4 --variant original
```
