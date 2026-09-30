"""Offline checks for sparse storage, ZIP range download and depth registration."""
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from prepare_7scenes_sparse import D_TO_RGB, RemoteZipReader, download_member, prepare_sequence, register_depth, save_json
from fast3r.data.components.spann3r_datasets.sparse_eval_frames import sparse_frame_count


def reference_depth(depth):
    # Literal math / scalar z-buffer from the pinned SimpleRecon implementation.
    depth = depth.astype(np.float32) / 1000
    yy, xx = np.indices(depth.shape, dtype=np.float64)
    mask = (depth > 0) & (depth < 100)
    z = depth[mask]
    eye = np.stack([(xx[mask]+.5-320)/585*z, (yy[mask]+.5-240)/585*z, z, np.ones(len(z))])
    eye = D_TO_RGB @ eye
    output = np.full((480, 640), 2e3, dtype=np.float32)
    for column in range(eye.shape[1]):
        d = eye[2, column]
        x = round(eye[0, column] / d * 525 + 320)
        y = round(eye[1, column] / d * 525 + 240)
        if 0 <= x < 640 and 0 <= y < 480:
            output[y, x] = min(output[y, x], d)
    output[output > 1e3] = 0
    return (output * 1000).astype(np.uint16)


class SparsePreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_projection_matches_scalar_reference(self):
        rng = np.random.default_rng(42)
        depth = np.zeros((480, 640), dtype=np.uint16)
        depth[100:400:3, 100:500:3] = rng.integers(1, 65536, size=depth[100:400:3, 100:500:3].shape, dtype=np.uint16)
        np.testing.assert_array_equal(register_depth(depth), reference_depth(depth))

    def test_invalid_depth_dimensions(self):
        with self.assertRaises(ValueError): register_depth(np.zeros((2, 2)))

    def make_archive(self):
        archive = self.root / 'fixture.zip'
        _, color = cv2.imencode('.png', np.zeros((480, 640, 3), np.uint8))
        _, depth = cv2.imencode('.png', np.full((480, 640), 2000, np.uint16))
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for index in range(41):
                z.writestr(f'frame-{index:06d}.color.png', color.tobytes())
                if index % 20 == 0:
                    z.writestr(f'frame-{index:06d}.depth.png', depth.tobytes())
                    z.writestr(f'frame-{index:06d}.pose.txt', '1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n')
        return archive

    def test_exact_stride_preserved_and_reusable_without_zip(self):
        target = self.root / 'seq-01'
        manifest = prepare_sequence(self.make_archive(), target, 20, {'sequence_zip_crc_checked': True})
        self.assertEqual(manifest['stored_indices'], [0, 20, 40])
        self.assertEqual(sparse_frame_count(target, full_video=True, kf_every=20), 41)
        self.assertEqual(len(list(target.glob('*.color.png'))), 3)
        self.assertEqual(prepare_sequence(None, target, 20, {}), manifest)

    def test_wrong_stride_and_missing_frame_rejected(self):
        target = self.root / 'seq-01'
        prepare_sequence(self.make_archive(), target, 20, {})
        with self.assertRaisesRegex(ValueError, 'recorded'):
            sparse_frame_count(target, full_video=True, kf_every=5)
        with self.assertRaisesRegex(ValueError, 'recorded'):
            sparse_frame_count(target, full_video=False, kf_every=20)
        (target / 'frame-000020.pose.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            sparse_frame_count(target, full_video=True, kf_every=20)

    def test_non_sparse_existing_directory_not_overwritten(self):
        target = self.root / 'seq-01'
        target.mkdir()
        with self.assertRaises(FileExistsError): prepare_sequence(None, target, 20, {})

    def test_broken_inventory_rejected(self):
        target = self.root / 'seq-01'
        target.mkdir()
        save_json(target/'frame_inventory.json', {'schema': '7scenes_stride_storage_v1',
                  'stride': 20, 'original_frame_count': 41, 'stored_indices': [0, 1, 2]})
        with self.assertRaisesRegex(ValueError, 'indices'):
            sparse_frame_count(target, full_video=True, kf_every=20)

    def test_http_range_member_download_crc(self):
        contents = io.BytesIO()
        expected = b'inner zip fixture' * 100
        with zipfile.ZipFile(contents, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('heads/seq-01.zip', expected)
        data = contents.getvalue()

        class Response:
            def __init__(self, content, headers, code=206):
                self.content, self.headers, self.status_code = content, headers, code
            def raise_for_status(self): pass
            def close(self): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def iter_content(self, size): yield self.content

        class Session:
            def head(self, *args, **kwargs):
                return Response(b'', {'Content-Length': str(len(data)), 'ETag': 'fixture'})
            def get(self, url, *, headers, **kwargs):
                start, end = map(int, headers['Range'].removeprefix('bytes=').split('-'))
                return Response(data[start:end+1], {'Content-Range': f'bytes {start}-{end}/{len(data)}'})

        remote = RemoteZipReader(Session(), 'fixture-url')
        with zipfile.ZipFile(remote) as z:
            member = z.getinfo('heads/seq-01.zip')
            download_member(remote, member, self.root/'downloaded.zip')
            self.assertEqual((self.root/'downloaded.zip').read_bytes(), expected)
            member.CRC = 0
            with self.assertRaisesRegex(ValueError, 'CRC'):
                download_member(remote, member, self.root/'bad.zip')

    def test_full_zip_response_refused(self):
        class Response:
            status_code = 200
            headers = {}
            def raise_for_status(self): pass
            def close(self): pass
        remote = RemoteZipReader.__new__(RemoteZipReader)
        remote.session = type('Session', (), {'get': lambda *a, **k: Response()})()
        remote.url, remote.etag, remote.size = 'fixture', None, 100
        with self.assertRaisesRegex(RuntimeError, 'range'): remote.get_range(0, 10)


if __name__ == '__main__': unittest.main()
