"""Exercise evaluator orchestration without importing GPU/model dependencies."""

import argparse
import ast
from contextlib import nullcontext
import csv
import copy
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "third_party/WMReward/compute_wmreward.py"


class EvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        model = Mock()
        model.to.return_value.eval.return_value = model
        tensor = Mock(shape=(1, 3, 32, 256, 256))
        tensor.to.return_value = tensor
        self.env = dict(
            argparse=argparse, csv=csv, math=math, os=os, Path=Path, random=random, tempfile=tempfile,
            print=Mock(), torch=SimpleNamespace(device=lambda _: "cpu",
                cuda=SimpleNamespace(is_available=lambda: False), no_grad=nullcontext),
            load_vjepa_models=Mock(return_value=(model, model, model, 256)),
            assert_video_is_30_fps=Mock(), load_video_as_tensor=Mock(return_value=tensor),
            compute_vjepa_loss_sliding_window=Mock(return_value=SimpleNamespace(item=lambda: 0.25)),
        )
        tree = ast.parse(SOURCE.read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"compute_multi_vjepa_surprise", "main"}]
        exec(compile(tree, str(SOURCE), "exec"), self.env)

    def run_multi(self, paths, **kwargs):
        return self.env["compute_multi_vjepa_surprise"](paths, **kwargs)

    def test_direct_paths_output_and_single_model_load(self):
        paths = ["/cache/datasets_variants/masked/datasets/a.mp4", "/datasets/b.mp4"]
        output = self.root / "nested" / "scores.csv"
        scores = self.run_multi(paths.copy(), direct_paths=True, output_path=str(output))
        self.assertEqual(scores, dict.fromkeys(paths, 0.25))
        self.env["load_vjepa_models"].assert_called_once()
        self.assertEqual({call.args[0] for call in self.env["load_video_as_tensor"].call_args_list}, set(paths))
        with output.open() as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[0], ["video", "surprise"])
        self.assertEqual(len(rows), 3)

    def test_legacy_rewrite(self):
        scores = self.run_multi(["/root/datasets/a.mp4"])
        self.assertEqual(scores, {"/root/cache/datasets_variants/256p_30fps/datasets/a.mp4": 0.25})

    def test_resume_only_finite_rows(self):
        output = self.root / "scores.csv"
        output.write_text("video,surprise\n/a.mp4,0.1\n/a.mp4,0.1\n/b.mp4,nan\n/c.mp4,bad\n/d.mp4,inf\n")
        scores = self.run_multi(["/a.mp4", "/b.mp4", "/c.mp4", "/d.mp4"],
                                direct_paths=True, output_path=str(output))
        self.assertEqual(scores["/a.mp4"], 0.1)
        self.assertEqual(self.env["load_video_as_tensor"].call_count, 3)
        self.assertTrue(all(math.isfinite(score) for score in scores.values()))
        with output.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({row["video"] for row in rows}), 4)
        self.assertTrue(all(math.isfinite(float(row["surprise"])) for row in rows))

    def test_resume_rejects_conflicts_and_bad_headers_without_rewriting(self):
        output = self.root / "scores.csv"
        for content, error in [
            ("video,surprise\n/a.mp4,0.1\n/a.mp4,0.2\n", "Conflicting finite"),
            ("path,score\n/a.mp4,0.1\n", "Unexpected surprise CSV header"),
        ]:
            with self.subTest(content=content):
                output.write_text(content)
                with self.assertRaisesRegex(ValueError, error):
                    self.run_multi(["/a.mp4"], direct_paths=True, output_path=str(output))
                self.assertEqual(output.read_text(), content)

    def test_nonfinite_scores_fail_without_writing(self):
        self.env["compute_vjepa_loss_sliding_window"].return_value = SimpleNamespace(item=lambda: float("nan"))
        output = self.root / "scores.csv"
        with self.assertRaisesRegex(RuntimeError, "Missing finite"):
            self.run_multi(["/a.mp4"], direct_paths=True, output_path=str(output))
        self.assertEqual(output.read_text(), "video,surprise\n")

    def test_decode_failure_does_not_prevent_other_scores(self):
        self.env["assert_video_is_30_fps"].side_effect = lambda path: (_ for _ in ()).throw(ValueError("bad video")) if path == "/bad.mp4" else None
        output = self.root / "scores.csv"
        with self.assertRaisesRegex(RuntimeError, "/bad.mp4"):
            self.run_multi(["/bad.mp4", "/good.mp4"], direct_paths=True, output_path=str(output))
        self.assertIn("/good.mp4,0.25", output.read_text())

    def test_cli_default_eval_max_and_explicit_output(self):
        listing = self.root / "videos.txt"
        listing.write_text("datasets/a.mp4\n")
        evaluator = self.env["compute_multi_vjepa_surprise"] = Mock()
        output = self.root / "custom.csv"
        with patch.object(sys, "argv", ["compute_wmreward.py", "--video_path", str(listing),
                                       "--direct_paths", "--output_path", str(output)]):
            self.env["main"]()
        kwargs = evaluator.call_args.kwargs
        self.assertIsNone(kwargs["max_videos"])
        self.assertTrue(kwargs["direct_paths"])
        self.assertEqual(kwargs["output_path"], str(output))
        self.assertEqual(kwargs["videos_paths"], [str(self.root / "datasets/a.mp4")])

    def test_local_hub_override_and_legacy_default(self):
        tree = ast.parse(SOURCE.read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "load_vjepa_models"]
        hub = Mock(return_value=("encoder", "predictor"))
        env = dict(os=os, copy=copy, torch=SimpleNamespace(hub=SimpleNamespace(load=hub)))
        exec(compile(tree, str(SOURCE), "exec"), env)
        with patch.dict(os.environ, {"VJEPA_HUB_DIR": "/local/vjepa2"}):
            self.assertEqual(env["load_vjepa_models"]("vith")[-1], 256)
        hub.assert_called_with("/local/vjepa2", "vjepa2_vit_huge", source="local")
        with patch.dict(os.environ, {"VJEPA_HUB_DIR": ""}):
            env["load_vjepa_models"]("vith")
        hub.assert_called_with("facebookresearch/vjepa2", "vjepa2_vit_huge")


if __name__ == "__main__":
    unittest.main()
