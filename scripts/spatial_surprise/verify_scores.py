#!/usr/bin/env python3
"""Require exactly one finite surprise score per input video."""
import argparse
import csv
import json
import math
import os
import hashlib
import subprocess
from pathlib import Path


def verify(video_list, scores):
    expected = video_list.read_text().splitlines()
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('Empty or duplicate input video list')
    with scores.open(newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['video', 'surprise']:
            raise ValueError('Expected CSV columns video,surprise')
        rows = list(reader)
    values = {}
    for row in rows:
        path, value = row['video'], float(row['surprise'])
        if path in values or not math.isfinite(value):
            raise ValueError(f'Duplicate or nonfinite score: {path}')
        values[path] = value
    missing, extra = set(expected) - values.keys(), values.keys() - set(expected)
    if missing or extra:
        raise ValueError(f'Incomplete score set: {len(missing)} missing, {len(extra)} unexpected')
    result = {'video_count': len(values), 'minimum': min(values.values()),
              'maximum': max(values.values()), 'mean': sum(values.values()) / len(values),
              'video_list': str(video_list.resolve()), 'scores': str(scores.resolve())}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video-list', required=True, type=Path)
    p.add_argument('--scores', required=True, type=Path)
    a = p.parse_args()
    result = verify(a.video_list, a.scores)
    # Record the submitted environment alongside the raw score table.
    if os.environ.get('SLURM_JOB_ID'):
        result['job_id'] = os.environ['SLURM_JOB_ID']
        result['configuration'] = {k: os.environ.get(k) for k in
            ['MAXFRAMES', 'WINDOW_SIZE', 'CONTEXT_FRAMES', 'STRIDE', 'MODE', 'VJEPA_HUB_DIR']}
        result['model'] = 'vith'
        result['seed'] = 42
        result['input_list_sha256'] = hashlib.sha256(a.video_list.read_bytes()).hexdigest()
        hub = os.environ.get('VJEPA_HUB_DIR')
        if hub:
            result['model_source_revision'] = subprocess.check_output(
                ['git', '-C', hub, 'rev-parse', 'HEAD'], text=True).strip()
    a.scores.with_suffix('.summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
