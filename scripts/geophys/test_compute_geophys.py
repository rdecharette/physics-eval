"""Integration checks using real upstream percentiles without model downloads."""
import argparse
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compute_geophys as runner
from geophys.score import SIGNALS


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.videos = [self.root / f'{i}.mp4' for i in range(3)]
        for video in self.videos:
            video.write_bytes(b'fixture')
        (self.root / 'sample.txt').write_text('\n'.join(str(p) for p in self.videos))
        self.args = argparse.Namespace(dataset=['sample'], variant='original', eval_max=0,
            reference=None, backbone='dinov3', layer=None, prep='direct', device='cpu',
            cache_dir=self.root / 'cache', output_dir=self.root / 'out', force=False)
        self.root_patch = patch.object(runner, 'REPO_ROOT', self.root)
        self.root_patch.start()
        self.model_patch = patch('geophys.backbones.get_backbone')
        self.model = self.model_patch.start()
        self.model.return_value.layers = ['18']
        self.signal_patch = patch('geophys.score.video_signals', side_effect=self.signals)
        self.extract = self.signal_patch.start()

    def tearDown(self):
        self.signal_patch.stop()
        self.model_patch.stop()
        self.root_patch.stop()
        self.temp.cleanup()

    @staticmethod
    def signals(ext, path, *args):
        return {k: float(Path(path).stem) for k in SIGNALS}

    def rows(self):
        with (self.args.output_dir / 'sample_geophys.score').open() as handle:
            return {Path(r['video']).stem: r for r in csv.DictReader(handle, delimiter='\t')}

    def test_percentiles_cache_and_invalidation(self):
        self.assertEqual(runner.run(self.args), 0)
        rows = self.rows()
        for i, expected in enumerate([1/6, 1/2, 5/6]):
            self.assertAlmostEqual(float(rows[str(i)]['geophys.score']), expected)
        self.assertEqual(self.extract.call_count, 3)
        self.assertEqual(runner.run(self.args), 0)
        self.assertEqual(self.extract.call_count, 3)
        self.videos[0].write_bytes(b'changed fixture')
        runner.run(self.args)
        self.assertEqual(self.extract.call_count, 4)

    def test_fixed_reference_and_empty_reference(self):
        reference = self.root / 'reference.txt'
        reference.write_text(str(self.videos[1]))
        self.args.reference = reference
        runner.run(self.args)
        self.assertEqual([float(self.rows()[str(i)]['geophys.score']) for i in range(3)], [0, .5, 1])
        reference.write_text('')
        with self.assertRaises(ValueError):
            runner.run(self.args)

    def test_failed_target_retained_and_retried(self):
        self.extract.side_effect = lambda ext, path, *a: None if Path(path).stem == '0' else self.signals(ext, path)
        self.assertEqual(runner.run(self.args), 1)
        self.assertEqual(self.rows()['0']['geophys.score'], '')
        self.assertIn('four frames', self.rows()['0']['error'])
        self.extract.side_effect = self.signals
        self.assertEqual(runner.run(self.args), 0)
        self.assertEqual(self.extract.call_count, 4)

    def test_singleton_requires_reference(self):
        self.args.eval_max = 1
        with self.assertRaises(ValueError):
            runner.run(self.args)
        self.model.assert_not_called()

    def test_invalid_reference_aborts(self):
        self.args.reference = self.root / 'reference.txt'
        self.args.reference.write_text(str(self.root / 'missing.mp4'))
        with self.assertRaisesRegex(FileNotFoundError, 'GeoPhys input video does not exist:') as caught:
            runner.run(self.args)
        self.assertIn(str(self.root / 'missing.mp4'), str(caught.exception))
        self.assertFalse(self.args.output_dir.exists())

    def test_missing_target_aborts(self):
        missing = runner.read_video_list(self.root / 'sample.txt')[0]
        missing.unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'GeoPhys input video does not exist:') as caught:
            runner.run(self.args)
        self.assertIn(str(missing), str(caught.exception))
        self.model.assert_not_called()
        self.extract.assert_not_called()
        self.assertFalse(self.args.output_dir.exists())

    def test_output_modes(self):
        self.args.output_dir = None
        suffix = Path('original_nAll/dinov3_l18_direct/sample_geophys.score')
        runner.run(self.args)
        self.assertTrue((self.root / 'output/geophys/independent' / suffix).exists())
        (self.root / 'second.txt').write_text(str(self.videos[0]))
        self.args.dataset.append('second')
        runner.run(self.args)
        self.assertTrue((self.root / 'output/geophys/combined' / suffix).exists())

    def test_reference_dataset_uses_full_matching_variant(self):
        self.args.variant = 'test_variant'
        self.args.eval_max = 1
        self.args.reference_dataset = 'pisabench'
        self.args.output_dir = None
        entries = [f'datasets/{i}.mp4' for i in range(3)]
        (self.root / 'sample.txt').write_text('\n'.join(entries))
        (self.root / 'pisabench.txt').write_text('\n'.join(entries))
        for entry in entries:
            video = self.root / 'cache/datasets_variants/test_variant' / entry
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b'fixture')
        runner.run(self.args)
        output = self.root / 'output/geophys/ref-pisabench/test_variant_n1/dinov3_l18_direct'
        import json
        metadata = json.loads((output / 'run.json').read_text())
        self.assertEqual(len(metadata['videos']), 1)
        self.assertEqual(len(metadata['reference']), 3)
        self.assertTrue(all('/test_variant/' in p for p in metadata['reference']))
        self.assertTrue((output / 'sample_geophys.score').exists())

    def test_variant_and_limit(self):
        path = self.root / 'relative.txt'
        path.write_text('# comment\n\ndatasets/a.mp4\ndatasets/b.mp4\n')
        videos = runner.read_video_list(path, '512p_30fps', 0)
        self.assertEqual(len(videos), 2)
        self.assertTrue(all(v.is_relative_to(self.root / 'cache/datasets_variants/512p_30fps') for v in videos))
        self.assertEqual(runner.read_video_list(path, '512p_30fps', 1), videos[:1])


if __name__ == '__main__':
    unittest.main()
