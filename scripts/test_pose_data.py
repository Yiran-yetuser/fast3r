#!/usr/bin/env python3
"""Offline synthetic preflight tests; not paper pose-evaluation results."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from check_pose_data import check_co3d, check_re10k, read_re10k_metadata, read_split


class PoseDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.split = self.root / 'split.txt'
        self.split.write_text('clip\n')
        self.video = self.root / 'video'
        self.metadata = self.root / 'metadata'
        (self.video / 'clip').mkdir(parents=True)
        self.metadata.mkdir()
        # w2c translation +2 -> c2w translation -2
        row = '1 0 0 2 0 1 0 0 0 0 1 0'
        self.txt = self.metadata / 'clip.txt'
        self.txt.write_text('https://example.com/video\n' + ''.join(
            f'{i} 1 1 .5 .5 0 0 {row}\n' for i in range(10)))
        for i in range(10):
            Image.new('RGB', (16, 16)).save(self.video / 'clip' / f'{i}.jpg')

    def check(self):
        return check_re10k(self.video, self.metadata, self.split)

    def test_pose_convention_and_finite(self):
        pose = read_re10k_metadata(self.txt)['0']['camera_pose']
        self.assertEqual(pose[0, 3], -2)
        self.assertTrue(np.isfinite(pose).all())

    def test_complete(self):
        self.assertEqual(self.check()['status'], 'ready')
        self.assertEqual(self.check()['ready_scene_count'], 1)

    def test_missing_scene_is_not_silently_dropped(self):
        self.split.write_text('clip\nmissing\n')
        report = self.check()
        self.assertEqual(report['status'], 'incomplete')
        self.assertEqual(report['expected_scene_count'], 2)
        self.assertEqual(report['issues'][0]['scene'], 'missing')

    def test_fewer_than_ten(self):
        (self.video / 'clip' / '9.jpg').unlink()
        self.assertEqual(self.check()['status'], 'incomplete')

    def test_unknown_timestamp(self):
        Image.new('RGB', (16, 16)).save(self.video / 'clip' / '100.jpg')
        self.assertIn('no GT', self.check()['issues'][0]['reason'])

    def test_corrupt_rgb(self):
        (self.video / 'clip' / '0.jpg').write_bytes(b'broken')
        self.assertEqual(self.check()['status'], 'incomplete')

    def test_duplicate_and_traversal_rejected(self):
        for text in ['clip\nclip\n', '../clip\n', '..\n', '\n']:
            self.split.write_text(text)
            with self.assertRaises(ValueError):
                read_split(self.split)

    def test_nonfinite_camera_rejected(self):
        self.txt.write_text(self.txt.read_text().replace('1 1 .5 .5', 'nan 1 .5 .5'))
        self.assertEqual(self.check()['status'], 'incomplete')

    def test_co3d_missing_split(self):
        self.assertEqual(check_co3d(self.root)['status'], 'incomplete')

    def test_co3d_complete_and_missing_mask(self):
        (self.root / 'selected_seqs_test.json').write_text(json.dumps({'apple': {'seq': list(range(10))}}))
        base = self.root / 'apple' / 'seq'
        for folder in ['images', 'depths', 'masks']:
            (base / folder).mkdir(parents=True)
        for i in range(10):
            stem = f'frame{i:06d}'
            Image.new('RGB', (16, 16)).save(base / 'images' / (stem + '.jpg'))
            Image.new('I;16', (16, 16), 100).save(base / 'depths' / (stem + '.jpg.geometric.png'))
            Image.new('L', (16, 16), 255).save(base / 'masks' / (stem + '.png'))
            np.savez(base / 'images' / (stem + '.npz'), camera_pose=np.eye(4),
                     camera_intrinsics=np.eye(3), maximum_depth=np.array(1))
        self.assertEqual(check_co3d(self.root)['status'], 'ready')
        (base / 'masks' / 'frame000000.png').unlink()
        self.assertEqual(check_co3d(self.root)['status'], 'incomplete')


if __name__ == '__main__':
    unittest.main()
