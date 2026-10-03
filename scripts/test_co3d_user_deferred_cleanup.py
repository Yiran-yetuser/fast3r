"""Offline check: a user-deferred CO3D queue must not recreate data or logs."""
import subprocess
import tempfile
import unittest
from pathlib import Path


class UserDeferredCo3dTests(unittest.TestCase):
    def test_both_queues_exit_before_any_work_when_deferred(self):
        for name in ('queue_co3d51_source_prepare_v1.sh','queue_co3d51_pose_eval_v1.sh'):
            with self.subTest(queue=name), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);(root/'results').mkdir()
                marker=root/'results/co3d_evaluation_deferred_cleanup_20261003.json'
                marker.write_text('{}')
                script=Path('scripts',name).read_text().replace(
                    'cd /home/yyz/fast3r',f'cd "{directory}"',1)
                result=subprocess.run(['/bin/bash'],input=script,text=True,capture_output=True,
                    timeout=3,env={'PATH':'/nonexistent'})
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('deferred by user',result.stdout)
                self.assertEqual(list(root.iterdir()),[root/'results'])
                self.assertEqual(list((root/'results').iterdir()),[marker])


if __name__=='__main__':unittest.main()
