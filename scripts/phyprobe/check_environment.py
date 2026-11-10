"""Check the isolated install, including real decoding, without fetching weights."""
from pathlib import Path
import subprocess
import sys

import phyprobe
from phyprobe._backbone_pe import _import_pe
from phyprobe._hub import load_checkpoint
from phyprobe._preprocess import load_video

root = Path(__file__).resolve().parents[2]
assert Path(phyprobe.__file__).resolve().is_relative_to(root / 'third_party/PhyProbe')
_import_pe()
config, _ = load_checkpoint(device='cpu')
video = load_video(root / 'third_party/PhyProbe/assets/task383-ref.mp4',
                   num_frames=config['cur_frames'], height=config['cur_height'],
                   width=config['cur_width'], fps=config['cur_target_fps'])
assert tuple(video.shape) == (1, 3, 48, 480, 480)
assert video.isfinite().all() and video.min() >= 0 and video.max() <= 1
print('PhyProbe/PE imports, bundled checkpoint and real video decoding: OK', flush=True)

result = subprocess.run([sys.executable, '-m', 'pip', 'check'], text=True, capture_output=True)
print(result.stdout, end='')
print(result.stderr, end='', file=sys.stderr)
# PyPI names this ctypes wheel py3-none-manylinux2010_x86_64, but its internal
# WHEEL metadata still says cp36-cp36m. Modern pip check rejects that stale tag.
# Accept only this exact packaging warning, after exercising the installed decoder.
known = 'decord 0.6.0 is not supported on this platform'
if result.returncode and result.stdout.strip() == known and not result.stderr.strip():
    print('Decord has stale CPython 3.6 wheel metadata; actual decoding passed.')
else:
    raise SystemExit(result.returncode)
