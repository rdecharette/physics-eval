import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

p = Path(__file__).resolve().parents[1] / 'scripts/spatial_surprise/verify_scores.py'
spec = importlib.util.spec_from_file_location('verify_scores', p)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ScoresTest(unittest.TestCase):
    def test_complete_and_invalid_results(self):
        with tempfile.TemporaryDirectory() as d:
            videos, scores = Path(d) / 'videos.txt', Path(d) / 'scores.csv'
            videos.write_text('/a.mp4\n/b.mp4\n')
            scores.write_text('video,surprise\n/b.mp4,0.4\n/a.mp4,0.2\n')
            result = module.verify(videos, scores)
            self.assertEqual(result['video_count'], 2)
            self.assertAlmostEqual(result['mean'], 0.3)
            for rows in ['/a.mp4,0.2\n', '/a.mp4,nan\n/b.mp4,0.4\n',
                         '/a.mp4,0.2\n/a.mp4,0.3\n/b.mp4,0.4\n',
                         '/a.mp4,0.2\n/b.mp4,0.4\n/c.mp4,0.5\n']:
                with self.subTest(rows=rows):
                    scores.write_text('video,surprise\n' + rows)
                    with self.assertRaises(ValueError):
                        module.verify(videos, scores)


if __name__ == '__main__':
    unittest.main()
