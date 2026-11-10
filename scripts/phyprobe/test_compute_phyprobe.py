"""Runner regression tests; no model download or GPU needed."""
import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('compute_phyprobe', Path(__file__).with_name('compute_phyprobe.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_list_variants_and_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            listing = root / 'sample.txt'
            listing.write_text('# comment\n\ndatasets/a.mp4\ndatasets/b.mp4\ndatasets/c.mp4\n')
            with patch.object(runner, 'REPO_ROOT', root):
                videos = runner.read_video_list(listing, '512p_30fps', 2)
                self.assertEqual(len(videos), 2)
                self.assertEqual(videos, runner.read_video_list(listing, '512p_30fps', 2))
                self.assertTrue(all(v.is_relative_to(root / 'cache/datasets_variants/512p_30fps') for v in videos))
                self.assertTrue(all(v.is_relative_to(root / 'datasets') for v in runner.read_video_list(listing)))

    def test_batch_failure_preserves_good_video(self):
        class Model:
            def score_batch(self, videos, **kwargs):
                if 'bad' in videos:
                    raise ValueError('corrupt video')
                return [0.3 for _ in videos]
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, torch=torch):
            good, bad = runner.score_chunk(Model(), ['good', 'bad'], None)
        self.assertEqual(good, {'good': 0.3})
        self.assertIn('corrupt video', bad['bad'])

    def test_output_resume_and_invalidation(self):
        calls = []
        class Model:
            def __init__(self, **kwargs):
                calls.append('init')
            def score_batch(self, videos, **kwargs):
                calls.extend(videos)
                return [0.25] * len(videos)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / 'with spaces.mp4'
            video.write_bytes(b'video')
            args = runner.parse_args(['--video', str(video), '--cache-dir', str(root / 'cache'),
                                      '--output-dir', str(root / 'out'), '--device', 'cpu'])
            with patch.dict(sys.modules, phyprobe=types.SimpleNamespace(PhyProbe=Model)):
                self.assertEqual(runner.run(args), 0)
                self.assertEqual(runner.run(args), 0)
                self.assertEqual(calls.count('init'), 1)
                video.write_bytes(b'changed video')
                self.assertEqual(runner.run(args), 0)
                self.assertEqual(calls.count('init'), 2)
                args.clip_sec = 3.0
                self.assertEqual(runner.run(args), 0)
                self.assertEqual(calls.count('init'), 3)
                args.video.append(root / 'missing.mp4')
                self.assertEqual(runner.run(args), 1)
            with (root / 'out/videos_phyprobe.score').open() as handle:
                rows = list(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual(rows[0]['phyprobe.score'], '0.25')
            self.assertEqual(rows[1]['phyprobe.score'], '')
            self.assertTrue(rows[1]['error'])

    def test_arguments(self):
        self.assertEqual(runner.parse_args([]).dataset, ['intphys2_possible'])
        for flags in (['--batch-size', '0'], ['--clip-sec', 'nan'], ['--clip-sec', '-1'], ['../escape']):
            with self.assertRaises(SystemExit):
                runner.parse_args(flags)


if __name__ == '__main__':
    unittest.main()
