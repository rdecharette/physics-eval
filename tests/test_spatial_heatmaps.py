"""Analytical coverage checks and an end-to-end score/manifest fixture."""

import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.spatial_surprise.generate_heatmaps import aggregate, generate, SCORE_RELATIVE


class HeatmapsTest(unittest.TestCase):
    def test_overlap_average_and_constant_edges(self):
        masks = [np.array([[255, 255, 0], [255, 255, 0]], dtype=np.uint8),
                 np.array([[0, 255, 255], [0, 255, 255]], dtype=np.uint8)]
        result, coverage = aggregate(masks, [2., 6.])
        np.testing.assert_array_equal(result, [[2., 4., 6.], [2., 4., 6.]])
        np.testing.assert_array_equal(coverage, [[1, 2, 1], [1, 2, 1]])
        result, _ = aggregate(masks, [3., 3.])
        np.testing.assert_array_equal(result, np.full((2, 3), 3.))

    def test_reject_invalid_aggregation(self):
        with self.assertRaises(ValueError):
            aggregate([np.array([[1, 0]])], [2.])
        with self.assertRaises(ValueError):
            aggregate([np.ones((2, 2))], [float("nan")])
        with self.assertRaises(ValueError):
            aggregate([np.ones((2, 2))], [])

    def test_manifests_scores_shared_colors_and_fail_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tag = "s2-v0"
            masks = root / "masks" / tag
            masks.mkdir(parents=True)
            bounds = {"y0": 0, "x0": 0, "y1": 2, "x1": 2}
            record = {"filename": "0.000_0.000.png", "bounds_pixels": bounds}
            Image.new("L", (2, 2), 255).save(masks / record["filename"])
            (masks / "manifest.json").write_text(json.dumps({
                "width": 2, "height": 2, "mask_count": 1, "scale": 2,
                "overlap": 0, "masks": [record]}))
            variants = root / "variants" / tag
            paths = []
            for name in ("1", "2"):
                relative = Path("datasets/intphys/dev/O1/02") / name
                directory = variants / relative
                directory.mkdir(parents=True)
                video = directory / "masked_0.000_0.000.mp4"
                paths.append(str(video))
                (directory / "manifest.json").write_text(json.dumps({
                    "dataset_path": relative.as_posix(), "variants": [{
                        "video": str(video), "mask": record["filename"], "bounds_pixels": bounds}]}))
            (variants / "videos.txt").write_text("\n".join(paths) + "\n")
            scores = root / "output" / tag / SCORE_RELATIVE
            scores.parent.mkdir(parents=True)
            def write_scores(rows):
                with scores.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["video", "surprise"])
                    writer.writerows(rows)
            args = argparse.Namespace(tags=[tag], mask_root=root / "masks",
                                      variants_root=root / "variants", output_root=root / "output",
                                      score_relative=SCORE_RELATIVE, overwrite=False)
            write_scores([(paths[0], 2.)])
            with self.assertRaises(ValueError):
                generate(args)
            self.assertEqual(list((root / "output").rglob("map.png")), [])
            write_scores([(paths[1], 6.), (paths[0], 2.)])
            results = generate(args)
            colors = []
            for (directory, _, _, _), value in zip(results, [2., 6.]):
                np.testing.assert_array_equal(np.load(directory / "map.npy"), np.full((2, 2), value))
                np.testing.assert_array_equal(np.load(directory / "coverage.npy"), np.ones((2, 2)))
                metadata = json.loads((directory / "map.json").read_text())
                self.assertEqual(metadata["display"]["vmin"], 2.)
                self.assertEqual(metadata["display"]["vmax"], 6.)
                with Image.open(directory / "map.png") as image:
                    self.assertEqual(image.mode, "RGB")
                    colors.append(image.getpixel((0, 0)))
            self.assertNotEqual(colors[0], colors[1])
            with self.assertRaises(ValueError):
                generate(args)
            args.overwrite = True
            write_scores([(paths[0], 3.), (paths[1], 3.)])
            generate(args)
            for directory, _, _, _ in results:
                self.assertTrue(np.isfinite(np.load(directory / "map.npy")).all())


if __name__ == "__main__":
    unittest.main()
