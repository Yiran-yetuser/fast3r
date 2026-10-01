#!/usr/bin/env python3
"""Offline synthetic tests: not real RE10K or Table1 scores."""
import argparse
import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import prepare_re10k_metadata as preparation
import fast3r_hf_re10k_pose_eval as evaluation
import download_re10k_rgb_archive as rgb_download
from audit_re10k_rgb_probe import first_chunk


class MetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive = self.root/'archive.tar.gz'
        self.camera = ('https://www.youtube.com/watch?v=example\n' + ''.join(
            f'{i} 1 1 .5 .5 0 0 1 0 0 0 0 1 0 0 0 0 1 0\n' for i in range(10))).encode()
        with tarfile.open(self.archive, 'w:gz') as tar:
            for path in ['RealEstate10K/test/clip.txt', 'RealEstate10K/train/clip.txt',
                         'RealEstate10K/test/unselected.txt']:
                entry = tarfile.TarInfo(path)
                entry.size = len(self.camera)
                tar.addfile(entry, io.BytesIO(self.camera))
        self.target = self.root/'test'

    def test_only_prescribed_test_files(self):
        preparation.extract_selected(self.archive, self.target, ['clip'])
        self.assertEqual([p.name for p in self.target.iterdir()], ['clip.txt'])
        self.assertEqual((self.target/'clip.txt').read_bytes(), self.camera)

    def test_reuse_identical_file(self):
        preparation.extract_selected(self.archive, self.target, ['clip'])
        preparation.extract_selected(self.archive, self.target, ['clip'])

    def test_existing_difference_not_overwritten(self):
        self.target.mkdir()
        path = self.target/'clip.txt'
        path.write_bytes(b'user data')
        with self.assertRaises(ValueError):
            preparation.extract_selected(self.archive, self.target, ['clip'])
        self.assertEqual(path.read_bytes(), b'user data')

    def test_missing_prescribed_id_not_dropped(self):
        with self.assertRaises(ValueError):
            preparation.extract_selected(self.archive, self.target, ['missing'])

    def test_space_reserve(self):
        with patch.object(preparation.shutil, 'disk_usage') as usage:
            usage.return_value.free = preparation.RESERVE-1
            with self.assertRaises(RuntimeError):
                preparation.budget(self.root)

    def test_prepared_manifest_skips_network(self):
        preparation.extract_selected(self.archive, self.target, ['clip'])
        split, manifest = self.root/'split.txt', self.root/'manifest.json'
        split.write_text('clip\n')
        args = argparse.Namespace(split_file=split, manifest=manifest,
                                  metadata_root=self.target, archive=self.archive)
        with patch.object(preparation, 'download'):
            first = preparation.prepare(args)
        with patch.object(preparation, 'download') as download:
            self.assertEqual(preparation.prepare(args), first)
            download.assert_not_called()


class PoseRunnerTests(unittest.TestCase):
    def test_sampling_stable_and_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'clip').mkdir()
            for i in range(30):
                (root/'clip'/f'{i}.jpg').touch()
            first = evaluation.frame_selection(root, 'clip', 10, 42)
            self.assertEqual(first, evaluation.frame_selection(root, 'clip', 10, 42))
            self.assertEqual(len(set(first)), 10)
            self.assertNotEqual(first, evaluation.frame_selection(root, 'clip', 10, 43))

    def test_crop_intrinsics(self):
        from PIL import Image
        camera = np.array([[600, 0, 320], [0, 600, 180], [0, 0, 1]], dtype=np.float32)
        image, cropped = evaluation.crop_resize(Image.new('RGB', (640, 360)), camera)
        self.assertEqual(image.size, (512, 288))
        self.assertTrue(np.allclose(cropped[:2, 2], [256, 144], atol=1))

    def test_official_metric_identity_and_world_gauge(self):
        gt = np.tile(np.eye(4), (10, 1, 1))
        gt[:, 0, 3] = np.arange(10)
        transform = np.eye(4)
        transform[:2, :2] = [[0, -1], [1, 0]]
        transform[:3, 3] = [3, 4, 5]
        predicted = transform@gt
        metrics, errors = evaluation.pose_metrics(predicted, gt)
        self.assertEqual(len(errors['rotation_deg']), 45)
        self.assertTrue(all(abs(x-1) < 1e-6 for x in metrics.values()))

    def test_aggregate_rejects_missing_or_duplicate_scenes(self):
        row = {'scene': 'a', 'metrics': dict.fromkeys(evaluation.METRICS, .5)}
        for rows in [[row], [row, row]]:
            with self.assertRaises(ValueError):
                evaluation.aggregate(rows, ['a', 'b'])

    def test_resume_signature_and_metrics_revalidated(self):
        gt = np.tile(np.eye(4), (2, 1, 1))
        gt[1, 0, 3] = 1
        metrics, errors = evaluation.pose_metrics(gt, gt)
        inputs = {'frames': [{}, {}]}
        row = {'scene': 'a', 'protocol_sha256': 'signature', 'inputs': inputs,
               'metrics': metrics, 'predicted_c2w': gt.tolist(), 'gt_c2w': gt.tolist(),
               'relative_errors': errors, 'pair_count': 1, 'pnp_failed_view_indices': []}
        evaluation.validate_row(row, 'a', 'signature', inputs)
        with self.assertRaises(ValueError):
            evaluation.validate_row(row, 'a', 'different', inputs)
        row['metrics']['mAA_30'] = .5
        with self.assertRaises(ValueError):
            evaluation.validate_row(row, 'a', 'signature', inputs)

    def test_pnp_failure_counted_not_dropped(self):
        preds = [{'pts3d_in_other_view': torch.ones(1, 8, 8, 3),
                  'conf': torch.full((1, 8, 8), 2.)} for _ in range(2)]
        with patch('fast3r.models.multiview_dust3r_module.estimate_focal', return_value=300), \
             patch('fast3r.dust3r.cloud_opt.init_im_poses.fast_pnp', return_value=(None, None)) as pnp:
            poses, focal, failures = evaluation.predict_poses(preds, 42)
        self.assertEqual(failures, [0, 1])
        self.assertEqual(poses.shape, (2, 4, 4))
        self.assertEqual(focal, 300)
        self.assertTrue(pnp.call_args.args[2].all())

    def test_existing_report_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'report.json'
            evaluation.save_new(target, {'history': True})
            with self.assertRaises(FileExistsError):
                evaluation.save_new(target, {'history': False})
            self.assertTrue(json.loads(target.read_text())['history'])


class RGBDownloadTests(unittest.TestCase):
    def test_prefix_resume_does_not_duplicate_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'partial.zip'
            path.write_bytes(b'original prefix')
            response = io.BytesIO(b'original prefix' + b'new suffix')
            rgb_download.consume_prefix(response, path.stat().st_size, path)
            self.assertEqual(response.read(), b'new suffix')
            self.assertEqual(path.read_bytes(), b'original prefix')

    def test_changed_resume_prefix_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'partial.zip'
            path.write_bytes(b'keep history')
            with self.assertRaises(ValueError):
                rgb_download.consume_prefix(io.BytesIO(b'changed data'), 12, path)
            self.assertEqual(path.read_bytes(), b'keep history')

    def test_source_identity_required(self):
        response = argparse.Namespace(headers={'Last-Modified': rgb_download.LAST_MODIFIED,
                                    'Content-Length': str(rgb_download.EXPECTED_SIZE)})
        rgb_download.check_source(response)
        response.headers['Content-Length'] = '1'
        with self.assertRaises(ValueError):
            rgb_download.check_source(response)


class RGBProbeTests(unittest.TestCase):
    def test_first_chunk_crc_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'archive.zip'
            with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('re10k/', b'')
                archive.writestr('re10k/test/000001.torch', b'safe payload')
            name, content = first_chunk(path)
            self.assertEqual(name, 're10k/test/000001.torch')
            self.assertEqual(content, b'safe payload')

    def test_partial_chunk_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'archive.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('clip.torch', b'payload'*20)
            path.write_bytes(path.read_bytes()[:45])
            with self.assertRaises(ValueError):
                first_chunk(path)

    def test_not_a_zip_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'archive.zip'
            path.write_bytes(b'not a zip')
            with self.assertRaises(ValueError):
                first_chunk(path)


if __name__ == '__main__':
    unittest.main()
