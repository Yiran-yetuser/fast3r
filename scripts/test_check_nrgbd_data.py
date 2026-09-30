"""Small-fixture tests; no downloads, GPU, or real dataset writes."""
import tempfile
import unittest
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from check_nrgbd_data import validate_data


class CheckDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.counts = patch('check_nrgbd_data.EXPECTED_FRAMES', {'test_scene': 2})
        self.counts.start()
        self.addCleanup(self.counts.stop)
        scene = self.root / 'test_scene'
        for folder, prefix in [('images', 'img'), ('depth', 'depth')]:
            (scene / folder).mkdir(parents=True)
            for index in range(2):
                (scene / folder / f'{prefix}{index}.png').write_bytes(b'fixture')
        (scene / 'poses.txt').write_text('1 0 0 0\n' * 8)

    def test_ready_without_zip_or_manifest(self):
        self.assertEqual(validate_data(self.root), {'test_scene': 2})

    def test_symlink_root(self):
        link = self.root / 'alias'
        link.symlink_to(self.root, target_is_directory=True)
        self.assertEqual(validate_data(link), {'test_scene': 2})

    def test_missing_depth_despite_stale_manifest(self):
        (self.root / 'manifest.json').write_text('{"status":"prepared"}')
        (self.root / 'test_scene/depth/depth1.png').unlink()
        with self.assertRaisesRegex(ValueError, 'contiguous frames'):
            validate_data(self.root)

    def test_empty_frame(self):
        (self.root / 'test_scene/images/img0.png').write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'empty'):
            validate_data(self.root)

    def test_missing_directory(self):
        with self.assertRaisesRegex(ValueError, 'directory'):
            validate_data(self.root / 'missing')

    def test_truncated_poses(self):
        (self.root / 'test_scene/poses.txt').write_text('1 0 0 0\n')
        with self.assertRaisesRegex(ValueError, '4x4 poses'):
            validate_data(self.root)

    def test_invalid_pose_numbers(self):
        (self.root / 'test_scene/poses.txt').write_text('bad 0 0 0\n' * 8)
        with self.assertRaises(ValueError):
            validate_data(self.root)

    def run_queue_fixture(self, missing_data=False):
        # Exercise the real Bash branching in an isolated tiny repository.
        project = self.root / 'project'
        scripts = project / 'scripts'
        scripts.mkdir(parents=True)
        (project / 'results').mkdir()
        (project / 'data').mkdir()
        (project / 'data/neural_rgbd').symlink_to(self.root, target_is_directory=True)
        source = Path(__file__).resolve().parent
        shutil.copyfile(source / 'queue_nrgbd_reproduction.sh', scripts / 'queue_nrgbd_reproduction.sh')
        (scripts / 'check_nrgbd_data.py').write_text(
            f'import runpy\nm = runpy.run_path({str(source / "check_nrgbd_data.py")!r})\n'
            'm["validate_data"].__globals__["EXPECTED_FRAMES"] = {"test_scene": 2}\n'
            'm["main"]()\n')
        (scripts / 'fast3r_hf_dtu_eval.py').write_text(
            'from pathlib import Path\nPath("results/eval_called").touch()\n')
        (project / 'results/nrgbd_seed42_stride40.json').write_text('{}')
        if missing_data:
            (self.root / 'test_scene/depth/depth1.png').unlink()
            (project / 'results/nrgbd_data_manifest.json').write_text('{"status":"prepared"}')
        commands = project / 'mock_bin'
        commands.mkdir()
        for name, content in {
            'curl': '#!/bin/sh\ntouch results/download_called\nexit 42\n',
            'nvidia-smi': '#!/bin/sh\necho 9000\n',
        }.items():
            command = commands / name
            command.write_text(content)
            command.chmod(0o755)
        environment = dict(os.environ, PYTHON_BIN=sys.executable,
                           PATH=f'{commands}:{os.environ["PATH"]}')
        args = ['bash', str(scripts / 'queue_nrgbd_reproduction.sh')]
        if not missing_data:
            args.append(str(os.getpid()))  # Live PID must not delay ready data.
        result = subprocess.run(args, env=environment, timeout=10, capture_output=True)
        return project, result

    def test_queue_skips_download_and_wait_for_ready_data(self):
        project, result = self.run_queue_fixture()
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertFalse((project / 'results/download_called').exists())
        self.assertTrue((project / 'results/eval_called').exists())
        self.assertIn('skipping download and extraction',
                      (project / 'results/nrgbd_pipeline.log').read_text())

    def test_queue_downloads_when_stale_manifest_hides_missing_data(self):
        project, result = self.run_queue_fixture(missing_data=True)
        self.assertEqual(result.returncode, 42)
        self.assertTrue((project / 'results/download_called').exists())
        self.assertFalse((project / 'results/eval_called').exists())


if __name__ == '__main__':
    unittest.main()
