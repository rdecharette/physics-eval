"""Small end-to-end checks for spatial variants, including evaluator decoding."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
from decord import VideoReader


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required")
class MaskedVideosTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="spatial-variants-test-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.relative = Path("datasets/intphys/dev/O1/02/1")
        self.scene = self.base / self.relative / "scene"
        self.scene.mkdir(parents=True)
        self.names = ["scene_1.png", "scene_2.png", "scene_3.png", "scene_10.png", "scene_11.png"]
        self.expected = []
        for i, name in enumerate(self.names):
            a = np.zeros((40, 40, 3), dtype=np.uint8)
            a[:, :, 0] = np.arange(40)[None, :] * 5
            a[:, :, 1] = np.arange(40)[:, None] * 5
            a[:, :, 2] = i * 40
            im = Image.fromarray(a)
            im.save(self.scene / name)
            self.expected.append(np.array(im.resize((32, 32), Image.Resampling.BILINEAR)))
        self.video_list = self.base / "videos.txt"
        self.video_list.write_text(str(self.relative) + "\n")
        self.mask_root = self.base / "masks"
        subprocess.run([
            sys.executable, str(ROOT / "scripts/spatial_surprise/generate_masks.py"),
            "--height", "32", "--width", "32", "--scale", "2", "--overlap", "0",
            "--output-root", str(self.mask_root),
        ], check=True, capture_output=True)
        self.mask_dir = self.mask_root / "s2-v0"
        self.output = self.base / "output"

    def run_generator(self):
        return subprocess.run([
            sys.executable, str(ROOT / "scripts/spatial_surprise/generate_masked_videos.py"),
            "--video-list", str(self.video_list), "--source-root", str(self.base),
            "--mask-dir", str(self.mask_dir), "--output-root", str(self.output),
        ], capture_output=True, text=True)

    def test_flat_videos_colors_timing_order_and_no_overwrite(self):
        result = self.run_generator()
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((self.mask_dir / "manifest.json").read_text())
        files = []
        for record in manifest["masks"]:
            mask = np.array(Image.open(self.mask_dir / record["filename"])) != 0
            video_dir = self.output / "s2-v0" / self.relative
            expected = []
            for frame in self.expected:
                a = np.full_like(frame, 128)
                a[mask] = frame[mask]
                expected.append(a)
            video = video_dir / f"masked_{Path(record['filename']).stem}.mp4"
            metadata = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=pix_fmt,profile,color_space,color_range,color_transfer,color_primaries",
                "-of", "json", str(video),
            ], text=True))["streams"][0]
            self.assertEqual(metadata["pix_fmt"], "yuv420p")
            self.assertEqual(metadata["profile"], "High")
            self.assertEqual(metadata["color_space"], "bt709")
            self.assertEqual(metadata["color_range"], "tv")
            self.assertEqual(metadata["color_transfer"], "bt709")
            self.assertEqual(metadata["color_primaries"], "bt709")
            vr = VideoReader(str(video), num_threads=1)
            self.assertEqual(len(vr), 6)
            self.assertAlmostEqual(vr.get_avg_fps(), 30)
            for j in range(6):
                source_index = ((2 * j + 1) * 25) // 60
                decoded = vr[j].asnumpy()
                error = np.abs(decoded.astype(float) - expected[source_index])
                self.assertLess(error.mean(), 3, f"Frame {j}: color conversion error")
                self.assertLess(error[mask].mean(), 6, f"Frame {j}: retained colors")
            files.append(video)
        self.assertEqual(list(self.output.rglob("*.png")), [])
        self.assertFalse(any(p.is_dir() for p in video_dir.iterdir()))
        output_manifest = json.loads((video_dir / "manifest.json").read_text())
        self.assertEqual(output_manifest["schema_version"], 2)
        self.assertEqual([r["video"] for r in output_manifest["variants"]], list(map(str, files)))
        self.assertTrue(all("frames" not in r for r in output_manifest["variants"]))
        self.assertEqual((self.output / "s2-v0/videos.txt").read_text().splitlines(), list(map(str, files)))
        before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
        result = self.run_generator()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(before, [hashlib.sha256(p.read_bytes()).hexdigest() for p in files])

    def test_invalid_mask_fails_without_producing_videos(self):
        mask = sorted(self.mask_dir.glob("*.png"))[0]
        Image.new("L", (32, 32), 127).save(mask)
        result = self.run_generator()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list(self.output.rglob("*.mp4")), [])

    def test_missing_source_fails_without_producing_videos(self):
        self.video_list.write_text("datasets/intphys/dev/missing\n")
        result = self.run_generator()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list(self.output.rglob("*.mp4")), [])

    def test_default_list_generates_both_source_videos(self):
        second = Path("datasets/intphys/dev/O1/02/2")
        shutil.copytree(self.scene, self.base / second / "scene")
        result = subprocess.run([
            sys.executable, str(ROOT / "scripts/spatial_surprise/generate_masked_videos.py"),
            "--source-root", str(self.base), "--mask-dir", str(self.mask_dir),
            "--output-root", str(self.output),
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        entries = (self.output / "s2-v0/videos.txt").read_text().splitlines()
        self.assertEqual(len(entries), 8)
        self.assertEqual(len(set(entries)), 8)
        for relative in (self.relative, second):
            directory = self.output / "s2-v0" / relative
            self.assertEqual(len(list(directory.glob("masked_*.mp4"))), 4)
            self.assertTrue((directory / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
