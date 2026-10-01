"""Offline synthetic archive/GT tests, never reported as Table1 measurements."""
import io
import argparse
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import prepare_re10k_rgb_from_archive as prep
from verify_re10k_rgb_prepared import verify


class RGBPreparationTests(unittest.TestCase):
    def example(self):
        image = io.BytesIO()
        Image.new('RGB', (640,360)).save(image, format='JPEG')
        payload = torch.tensor(list(image.getvalue()), dtype=torch.uint8)
        camera = np.r_[1.,1.,.5,.5,0.,0.,np.eye(4)[:3].ravel()]
        example = {'timestamps': torch.arange(10), 'cameras': torch.tensor(np.tile(camera,(10,1))),
                   'images': [payload]*10}
        metadata = {str(i): {'intrinsics_normalized': np.array([1.,1.,.5,.5]),
                            'camera_pose': np.eye(4)} for i in range(10)}
        return example, metadata

    def test_full_candidate_inventory_and_camera_verified(self):
        example, metadata = self.example()
        frames, payloads, delta = prep.validate_example(example, metadata)
        self.assertEqual(len(frames), 10)
        self.assertEqual(len(payloads),10)
        self.assertEqual(delta,0)

    def test_missing_candidate_not_silently_accepted(self):
        example, metadata = self.example()
        metadata['10'] = metadata['0']
        with self.assertRaises(ValueError):
            prep.validate_example(example, metadata)

    def test_actual_nonstandard_dimensions_preserved_not_stretched(self):
        example, metadata = self.example()
        buffer = io.BytesIO()
        Image.new('RGB',(640,338)).save(buffer,format='JPEG')
        payload = torch.tensor(list(buffer.getvalue()),dtype=torch.uint8)
        example['images'] = [payload]*10
        frames,payloads,_ = prep.validate_example(example,metadata)
        self.assertEqual(frames[0]['shape_h_w'],[338,640])
        self.assertEqual(payloads[0],buffer.getvalue())

    def test_changed_camera_rejected_not_corrected(self):
        example, metadata = self.example()
        example['cameras'][0,0] += .1
        with self.assertRaises(ValueError):
            prep.validate_example(example, metadata)

    def test_write_resume_preserves_identical_and_rejects_user_change(self):
        with tempfile.TemporaryDirectory() as folder:
            frames,payloads,_ = prep.validate_example(*self.example())
            prep.write_scene(folder,'clip',frames,payloads)
            prep.write_scene(folder,'clip',frames,payloads)
            target = Path(folder)/'clip/0.jpg'
            target.write_bytes(b'user data')
            with self.assertRaises(ValueError):
                prep.write_scene(folder,'clip',frames,payloads)
            self.assertEqual(target.read_bytes(),b'user data')

    def test_extra_frame_changes_sampling_population_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            frames,payloads,_ = prep.validate_example(*self.example())
            prep.write_scene(folder,'clip',frames,payloads)
            (Path(folder)/'clip/extra.jpg').touch()
            with self.assertRaises(ValueError):
                prep.write_scene(folder,'clip',frames,payloads)

    def test_missing_index_id_explicit_and_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            present,missing = 'a'*16,'b'*16
            (root/(missing+'.txt')).write_text('https://example.test/video\n')
            for chunk in ['000000.torch','../../user.torch']:
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer,'w') as z:
                    z.writestr('re10k/test/index.json',json.dumps({present:chunk}))
                    z.writestr('re10k/test/'+chunk,b'synthetic')
                buffer.seek(0)
                with zipfile.ZipFile(buffer) as z:
                    if '..' in chunk:
                        with self.assertRaises(ValueError):
                            prep.audit_index(z,[present,missing],root)
                    else:
                        _,report = prep.audit_index(z,[present,missing],root)
                        self.assertEqual(report['missing_scene_ids'],[missing])
                        self.assertEqual(report['covered_scene_count'],1)
                        self.assertFalse(report['full_scene_coverage_verified'])

    def test_duplicate_index_key_rejected(self):
        with self.assertRaises(ValueError):
            json.loads('{"a":1,"a":2}',object_pairs_hook=prep.unique_object)

    def test_low_space_rejected(self):
        with patch.object(prep.shutil,'disk_usage') as usage:
            usage.return_value.free = prep.RESERVE-1
            with self.assertRaises(RuntimeError):
                prep.budget('.')

    def test_end_to_end_incomplete_split_and_resume_preserve_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            present,missing = 'a'*16,'b'*16
            split = root/'split.txt'
            split.write_text(present+'\n'+missing+'\n')
            metadata = root/'metadata'
            metadata.mkdir()
            camera = 'https://example.test/video\n'+''.join(
                f'{i} 1 1 .5 .5 0 0 1 0 0 0 0 1 0 0 0 0 1 0\n' for i in range(10))
            for name in [present,missing]:
                (metadata/(name+'.txt')).write_text(camera)
            example,_ = self.example()
            example['key'] = present
            buffer = io.BytesIO()
            torch.save([example],buffer)
            archive = root/'clips.zip'
            with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr('re10k/test/index.json',json.dumps({present:'000000.torch'}))
                z.writestr('re10k/test/000000.torch',buffer.getvalue())
            source = root/'source.json'
            source.write_text(json.dumps({'archive_bytes':archive.stat().st_size,
                                          'archive_sha256':prep.sha(archive)}))
            args = argparse.Namespace(archive=archive,archive_manifest=source,split_file=split,
                                      metadata_root=metadata,data_root=root/'rgb',
                                      index_report=root/'index.json',output_json=root/'result.json',index_only=False)
            first = prep.prepare(args)
            self.assertEqual(first['status'],'covered_prepared_split_incomplete')
            self.assertEqual(first['missing_scene_ids'],[missing])
            self.assertEqual(first['prepared_scene_count'],1)
            self.assertFalse(first['full_scene_coverage_verified'])
            self.assertEqual(prep.prepare(args),first)
            official = root/'official.json'
            official.write_text(json.dumps({'records':{present:{'sha256':prep.sha(metadata/(present+'.txt')),
                                                                'camera_record_count':10}}}))
            verified = verify(args.data_root,metadata,args.output_json,official,args.index_report,split)
            self.assertEqual(verified['verified_frame_count'],10)
            self.assertFalse(verified['full_scene_coverage_verified'])
            (args.data_root/present/'0.jpg').write_bytes(b'changed data')
            with self.assertRaises(ValueError):
                verify(args.data_root,metadata,args.output_json,official,args.index_report,split)


if __name__ == '__main__':
    unittest.main()
