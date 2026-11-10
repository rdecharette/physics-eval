"""Score dataset lists or individual videos with the editable PhyProbe source."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = REPO_ROOT / 'third_party/PhyProbe'
sys.path.insert(0, str(VENDOR_ROOT))


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_video_list(path, variant='original', eval_max=None):
    videos = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        video = Path(line.strip()).expanduser()
        if variant != 'original':
            if video.is_absolute():
                video = video.relative_to(REPO_ROOT)
            if '..' in video.parts:
                raise ValueError(f'Variant path escapes dataset root: {video}')
            video = REPO_ROOT / 'cache/datasets_variants' / variant / video
        elif not video.is_absolute():
            video = path.parent / video
        videos.append(video.resolve())
    random.Random(42).shuffle(videos)
    if eval_max and eval_max > 0:
        videos = videos[:eval_max]
    if not videos:
        raise ValueError(f'No videos in {path}')
    return videos


def file_digest(paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def valid_score(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1


def score_chunk(model, videos, clip_sec):
    """Retry failed batches individually so one corrupt video cannot hide good ones."""
    try:
        scores = model.score_batch(videos, batch_size=len(videos), clip_sec=clip_sec)
        if len(scores) != len(videos) or not all(valid_score(s) for s in scores):
            raise ValueError('PhyProbe returned invalid scores')
        return dict(zip(videos, scores)), {}
    except Exception as exc:
        if len(videos) == 1:
            return {}, {videos[0]: f'{type(exc).__name__}: {exc}'}
        print(f'Batch failed ({exc}); retrying individually.', file=sys.stderr, flush=True)
    # Release unused CUDA allocations after leaving the exception traceback.
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    values, errors = {}, {}
    for video in videos:
        good, bad = score_chunk(model, [video], clip_sec)
        values.update(good)
        errors.update(bad)
    return values, errors


def run(args):
    jobs = [(name, video) for name in args.dataset
            for video in read_video_list(REPO_ROOT / f'{name}.txt', args.variant, args.eval_max)]
    jobs.extend(('videos', p.expanduser().resolve()) for p in args.video)
    checkpoint = (args.checkpoint or VENDOR_ROOT / 'phyprobe/weights/pecore_paper.pt').expanduser().resolve()
    settings = dict(model='pecore_paper', clip_sec=args.clip_sec, device=args.device,
                    batch_size=args.batch_size, checkpoint_sha256=file_digest([checkpoint]),
                    source_sha256=file_digest(list((VENDOR_ROOT / 'phyprobe').glob('*.py')) +
                                              [Path(__file__), Path(__file__).with_name('requirements.txt')]))
    limit = args.eval_max if args.eval_max and args.eval_max > 0 else 'All'
    window = 'full' if args.clip_sec is None else f'clip{args.clip_sec:g}s'
    output = (args.output_dir or REPO_ROOT / 'output/phyprobe' /
              f'{args.variant}_n{limit}' / f'pecore_paper_{window}').expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    values, errors, pending = {}, {}, []
    for video in dict.fromkeys(v for _, v in jobs):
        try:
            stat = video.stat()
            identity = dict(settings, path=str(video), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            cached = cache_dir / f'{key}.json'
            if cached.exists() and not args.force:
                try:
                    score = json.loads(cached.read_text())['score']
                    if valid_score(score):
                        values[video] = score
                        continue
                except (ValueError, KeyError, TypeError):
                    pass
            pending.append((video, cached, identity))
        except OSError as exc:
            errors[video] = str(exc)
    if pending:
        # Import and model initialization failures are fatal, never cached as video errors.
        from phyprobe import PhyProbe
        model = PhyProbe(checkpoint=checkpoint, device=args.device)
        for start in range(0, len(pending), args.batch_size):
            chunk = pending[start:start + args.batch_size]
            print(f'Scoring {start + 1}-{start + len(chunk)} / {len(pending)}', flush=True)
            good, bad = score_chunk(model, [v for v, _, _ in chunk], args.clip_sec)
            values.update(good)
            errors.update(bad)
            for video, cached, identity in chunk:
                if video in good:
                    atomic_write(cached, json.dumps(dict(identity, score=good[video]), allow_nan=False))
    for name in dict.fromkeys(name for name, _ in jobs):
        buffer = io.StringIO()
        writer = csv.writer(buffer, delimiter='\t')
        writer.writerow(['video', 'phyprobe.score', 'error'])
        for dataset, video in jobs:
            if dataset == name:
                writer.writerow([str(video), format(values[video], '.10g') if video in values else '',
                                 errors.get(video, '')])
        destination = output / f'{name}_phyprobe.score'
        atomic_write(destination, buffer.getvalue())
        print(f'Wrote {destination}', flush=True)
    # Dataset-specific metadata avoids collisions when separate dataset jobs share an output directory.
    for name in dict.fromkeys(name for name, _ in jobs):
        atomic_write(output / f'{name}_run.json', json.dumps(dict(settings, variant=args.variant,
                     eval_max=args.eval_max, checkpoint=str(checkpoint),
                     videos=[str(v) for n, v in jobs if n == name],
                     errors={str(v): errors[v] for n, v in jobs if n == name and v in errors}), indent=2))
    if errors:
        print(f'{len(errors)} video(s) failed; see error columns. Rerun to retry.', file=sys.stderr)
    return 1 if errors else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', nargs='*', help='Names of repository <dataset>.txt lists')
    parser.add_argument('--video', type=Path, action='append', default=[], help='Single video; repeatable, no variant remapping')
    parser.add_argument('--variant', default='512p_30fps', help='Dataset cache variant or original')
    parser.add_argument('--eval-max', type=int, help='Maximum videos per dataset; <=0 means all')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--clip-sec', type=float, help='Restrict to first N seconds; default is full duration as upstream')
    parser.add_argument('--checkpoint', type=Path, help='Custom head checkpoint; default is bundled pecore_paper')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cache-dir', type=Path, default=REPO_ROOT / 'cache/phyprobe/scores')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    if not args.dataset and not args.video:
        args.dataset = ['intphys2_possible']
    for name in [*args.dataset, args.variant]:
        if Path(name).name != name or name in ('.', '..'):
            parser.error('dataset and variant must be names, not paths')
    if args.video and 'videos' in args.dataset:
        parser.error('dataset name videos is reserved when using --video')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.clip_sec is not None and (not math.isfinite(args.clip_sec) or args.clip_sec <= 0):
        parser.error('--clip-sec must be finite and positive')
    return args


def main():
    args = parse_args()
    expected = (REPO_ROOT / '.venv/phyprobe').resolve()
    if Path(sys.prefix).resolve() != expected or sys.prefix == sys.base_prefix:
        raise SystemExit(f'Use {expected}/bin/python. Set up with: bash scripts/phyprobe/setup.sh')
    # Keep default model caches inside this repository; respect explicit cluster cache settings.
    os.environ.setdefault('HF_HOME', str(REPO_ROOT / 'cache/huggingface'))
    os.environ.setdefault('TORCH_HOME', str(REPO_ROOT / 'cache/torch'))
    os.environ.setdefault('PHYPROBE_CACHE', str(REPO_ROOT / 'cache/phyprobe/weights'))
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
