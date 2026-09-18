"""Check exact, consistently padded mask coordinates and layout validation."""

from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_masks", ROOT / "scripts/spatial_surprise/generate_masks.py"
)
masks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(masks)


class SpatialMasksTest(unittest.TestCase):
    def test_coordinates_are_exact_fixed_width_and_sort_spatially(self):
        for scale, places, count, second, last in [
            (4, 3, 49, "0.000_0.125.png", "0.750_0.750.png"),
            (8, 4, 225, "0.0000_0.0625.png", "0.8750_0.8750.png"),
        ]:
            with self.subTest(scale=scale):
                window, stride, records = masks.mask_layout(256, 256, scale, Fraction("0.5"))
                self.assertEqual(window, (256 // scale,) * 2)
                self.assertEqual(stride, (128 // scale,) * 2)
                self.assertEqual(len(records), count)
                filenames = [record["filename"] for record in records]
                self.assertEqual(filenames, sorted(filenames))
                self.assertEqual(len(set(filenames)), count)
                self.assertEqual(filenames[1], second)
                self.assertEqual(filenames[-1], last)
                for record in records:
                    for axis in ("y", "x"):
                        coordinate = record["top_left_normalized"][axis]
                        self.assertEqual(len(coordinate.partition(".")[2]), places)
                        self.assertEqual(Fraction(coordinate), Fraction(record["bounds_pixels"][axis + "0"], 256))

    def test_manifest_precision_and_saved_mask_pixels(self):
        with tempfile.TemporaryDirectory(prefix="spatial-masks-test-") as temp:
            root = Path(temp)
            for scale, places in ((4, 3), (8, 4)):
                directory = masks.generate(256, 256, scale, Fraction("0.5"), root)
                self.assertEqual(directory.name, f"s{scale}-v0.5")
                manifest = json.loads((directory / "manifest.json").read_text())
                self.assertEqual(manifest["coordinate_decimals"], places)
                self.assertEqual(manifest["overlap_exact"], "0.5")
                self.assertTrue((directory / manifest["contact_sheet"]).is_file())
                for record in manifest["masks"]:
                    with Image.open(directory / record["filename"]) as mask:
                        self.assertEqual(mask.mode, "L")
                        self.assertEqual(mask.size, (256, 256))
                        bounds = record["bounds_pixels"]
                        self.assertEqual(mask.getbbox(), (bounds["x0"], bounds["y0"], bounds["x1"], bounds["y1"]))
                        self.assertEqual(dict((value, count) for count, value in mask.getcolors()),
                                         {0: 256**2 - (256 // scale)**2, 255: (256 // scale)**2})
                with self.assertRaises(ValueError):
                    masks.generate(256, 256, scale, Fraction("0.5"), root)

    def test_axes_share_precision(self):
        _, _, records = masks.mask_layout(16, 20, 2, Fraction("0.5"))
        self.assertTrue(all(len(v.partition(".")[2]) == 3
                            for r in records for v in r["top_left_normalized"].values()))

    def test_invalid_layouts_are_rejected(self):
        cases = [
            (0, 256, 4, Fraction("0.5")),
            (256, 256, 0, Fraction("0.5")),
            (256, 256, 4, Fraction(-1)),
            (256, 256, 4, Fraction(1)),
            (255, 256, 4, Fraction("0.5")),
            (256, 256, 4, Fraction("0.3")),
            (24, 24, 3, Fraction("0.25")),
            (12, 12, 3, Fraction("0.5")),  # repeating normalized coordinates
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                masks.mask_layout(*args)
        with self.assertRaises(ValueError):
            masks.decimal_string(Fraction("0.0625"), 3)


if __name__ == "__main__":
    unittest.main()
