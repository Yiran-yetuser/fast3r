"""Synthetic strict-error/zero-depth/oversampling regressions, not paper data."""
import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import PIL.Image

from co3d_lazy_dataset import StrictLazyCo3d, STRICT_LOAD, process_frame_allow_zero
from probe_co3d_preprocess import process_frame
from probe_co3d_lazy_sample import same_views, json_index
from fast3r.dust3r.datasets.base.easy_dataset import ResizedDataset
from fast3r.dust3r.datasets.co3d_multiview import Co3d_Multiview
from fast3r.dust3r.datasets.utils import cropping
import test_co3d_preprocess_probe as fixture_module
from test_co3d_preprocess_probe import projection


class ZeroJitter:
    def integers(self,*args): return 0


class LazyDatasetTests(unittest.TestCase):
    def processed(self,root,scene,frame,valid=True):
        stem=f'frame{frame:06d}'
        for d in ('images','depths','masks'): (root/'apple'/scene/d).mkdir(parents=True,exist_ok=True)
        image=root/'apple'/scene/'images'/(stem+'.jpg')
        PIL.Image.new('RGB',(512,384),(10,20,30)).save(image)
        cv2.imwrite(str(root/'apple'/scene/'depths'/(stem+'.jpg.geometric.png')),
                    np.full((384,512),65535,np.uint16))
        cv2.imwrite(str(root/'apple'/scene/'masks'/(stem+'.png')),
                    np.full((384,512),255 if valid else 0,np.uint8))
        np.savez(image.with_suffix('.npz'),camera_pose=np.eye(4,dtype=np.float32),
                 camera_intrinsics=np.array([[500,0,256],[0,500,192],[0,0,1]],np.float32),maximum_depth=np.float32(1))

    def dataset(self,root,selected):
        return StrictLazyCo3d(selected,SimpleNamespace(root=root,ensure=lambda *args:None))

    def test_missing_files_raise_not_invalid_skip(self):
        with tempfile.TemporaryDirectory() as d:
            ds=self.dataset(Path(d),{'apple':{'s':[1]}})
            with self.assertRaises(FileNotFoundError): ds[0]
            self.assertEqual(ds.trace[-1]['status'],'hard_error')
            self.assertFalse(any(ds.invalidate[('apple','s')][(512,384)]))
            self.assertFalse(ds.invalid_scene_tracker)

    def test_crc_or_prepare_error_is_not_swallowed(self):
        with tempfile.TemporaryDirectory() as d:
            ds=self.dataset(Path(d),{'apple':{'s':[1]}})
            def fail(*args): raise ValueError('CRC mismatch synthetic')
            ds.preparer.ensure=fail
            with self.assertRaisesRegex(ValueError,'CRC'): ds[0]
            self.assertEqual(ds.trace[-1]['status'],'hard_error')

    def test_zero_mask_marks_invalid_then_oversamples_valid(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); self.processed(root,'s',1,False);self.processed(root,'s',2,True)
            ds=self.dataset(root,{'apple':{'s':[1,2]}});ds.combinations=[tuple([0,1]*5)]
            random.seed(20)
            views=ds._get_views(0,(512,384),ZeroJitter())
            self.assertEqual(len(views),10)
            self.assertTrue(all(v['instance']=='frame000002.jpg' for v in views))
            self.assertEqual(sum(t['status']=='zero_masked_depth_after_crop' for t in ds.trace),1)
            self.assertTrue(ds.invalidate[('apple','s')][(512,384)][0])
            self.assertEqual(len(ds.trace),11)  # 1 zero + 10 valid including oversamples

    def test_all_zero_scene_retries_original_next_scene(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.processed(root,'zero',1,False);self.processed(root,'good',1,True)
            ds=self.dataset(root,{'apple':{'zero':[1],'good':[1]}})
            with contextlib.redirect_stdout(io.StringIO()): views=ds[0]
            self.assertTrue(all(v['label']=='apple/good' for v in views))
            self.assertEqual(ds.invalid_scene_tracker,{('apple','zero')})
            self.assertEqual([x['scene'] for x in ds.pool_attempts],['zero','good'])

    def test_original_loader_outputs_equal_including_rng_and_pts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.processed(root,'s',1)
            ds=self.dataset(root,{'apple':{'s':[1]}})
            random.seed(10);left=ds[0]
            # Use the same dataset state but bypass wrapper for exact source comparison.
            ds.trace=[];ds._load_view_data=lambda *args:Co3d_Multiview._load_view_data(ds,*args)
            random.seed(10);right=ds[0]
            same_views(left,right)

    def test_preprocessing_positive_exact_old_adapter(self):
        with tempfile.TemporaryDirectory() as d:
            paths,a=fixture_module.PreprocessProbeTests().fixture(d)
            ns={'opencv_from_cameras_projection':projection}
            before=process_frame(paths,a,ns,cropping);after=process_frame_allow_zero(paths,a,ns)
            self.assertEqual(before[0].tobytes(),after[0].tobytes())
            for x,y in zip(before[1:6],after[1:6]):np.testing.assert_array_equal(x,y)

    def test_zero_raw_depth_preserves_zero_quantization_and_maximum(self):
        with tempfile.TemporaryDirectory() as d:
            paths,a=fixture_module.PreprocessProbeTests().fixture(d,0.)
            result=process_frame_allow_zero(paths,a,{'opencv_from_cameras_projection':projection})
            self.assertEqual(result[5],0);self.assertTrue(np.all(result[1]==0))
            self.assertTrue(result[-1]['zero_maximum_depth'])

    def test_nonfinite_raw_is_hard_error(self):
        with tempfile.TemporaryDirectory() as d:
            paths,a=fixture_module.PreprocessProbeTests().fixture(d,float('nan'))
            with self.assertRaisesRegex(ValueError,'Malformed'):
                process_frame_allow_zero(paths,a,{'opencv_from_cameras_projection':projection})

    def test_original_pool_and_wrapper_mapping_not_sparse(self):
        ds=self.dataset(Path('/unused'),{'apple':{'s':list(range(202))}})
        wrapper=ResizedDataset(100,ds);wrapper.set_epoch(0)
        self.assertEqual(len(ds.scenes[('apple','s')]),202)
        self.assertEqual(len(wrapper._idxs_mapping),100)
        self.assertEqual(len(ds),len(ds.combinations))

    def test_numpy_base_index_serializes_as_native_int(self):
        view={'idx':(np.int64(12303633),0,3)}
        self.assertEqual(json.loads(json.dumps(json_index(view))),[12303633,0,3])


if __name__=='__main__':unittest.main()
