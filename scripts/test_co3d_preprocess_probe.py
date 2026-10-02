"""Synthetic decode/quantization/NPZ checks; not real dataset readiness."""
import io
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import torch

from probe_co3d_preprocess import process_frame, save_processed, exclusive_bytes, MAX_PIXELS
from fast3r.dust3r.datasets.utils import cropping
from fast3r.dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset


def projection(R,T,f,p,size):
    wh=np.asarray(size)[::-1]
    s=min(wh)/2
    K=np.eye(3);K[:2,2]=wh/2-np.asarray(p)*s
    K[0,0],K[1,1]=np.asarray(f)*s
    flip=np.array([-1.,-1.,1.])
    return torch.from_numpy((R*flip).T),torch.from_numpy(T*flip),torch.from_numpy(K)


class PreprocessProbeTests(unittest.TestCase):
    def fixture(self,root,depth=2.):
        root=Path(root)
        image=root/'raw.jpg';dep=root/'depth.png';mask=root/'mask.png'
        PIL.Image.new('RGB',(16,12),(20,40,60)).save(image)
        bits=np.full((12,16),depth,dtype=np.float16).view(np.uint16)
        cv2.imwrite(str(dep),bits);cv2.imwrite(str(mask),np.full((12,16),255,np.uint8))
        a={'image':{'size':[12,16],'path':'apple/s/images/frame000001.jpg'},
           'depth':{'scale_adjustment':1.,'path':'apple/s/depths/frame000001.jpg.geometric.png'},
           'mask':{'path':'apple/s/masks/frame000001.png'},
           'viewpoint':{'R':np.eye(3).tolist(),'T':[1.,2.,3.],'focal_length':[2.,2.],
                        'principal_point':[0.,0.],'intrinsics_format':'ndc_isotropic'}}
        return {'images':image,'depths':dep,'masks':mask},a

    def test_fp16_bit_decode_quantization_and_float32_pose(self):
        with tempfile.TemporaryDirectory() as d:
            paths,a=self.fixture(d)
            values=process_frame(paths,a,{'opencv_from_cameras_projection':projection},cropping)
            self.assertEqual(values[5],2.)
            self.assertTrue(np.all(values[1]==65535))
            self.assertTrue(np.all(values[2]==255))
            self.assertEqual(values[4].dtype,np.float32)
            self.assertEqual(values[-1]['raw_depth_max'],2.)
            np.testing.assert_array_equal(values[4][:3,3],[-1,-2,-3])

    def test_npz_reuse_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            paths,a=self.fixture(d)
            values=process_frame(paths,a,{'opencv_from_cameras_projection':projection},cropping)
            root=Path(d)/'processed'
            targets,meta=save_processed(root,a,values)
            save_processed(root,a,values)
            with np.load(meta,allow_pickle=False) as data: self.assertEqual(data['camera_pose'].dtype,np.float32)
            damaged=list(values);damaged[4]=values[4].copy();damaged[4][0,3]+=1
            with self.assertRaises(ValueError):save_processed(root,a,damaged)

    def test_nonfinite_or_all_zero_depth_not_repaired(self):
        with tempfile.TemporaryDirectory() as d:
            for depth in (0.,float('nan')):
                paths,a=self.fixture(d,depth)
                with self.assertRaises(ValueError):process_frame(paths,a,{'opencv_from_cameras_projection':projection},cropping)

    def test_wrong_annotation_shape_and_scale_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            paths,a=self.fixture(d)
            a['image']['size']=[13,16]
            with self.assertRaises(ValueError):process_frame(paths,a,{'opencv_from_cameras_projection':projection},cropping)
            a['image']['size']=[12,16];a['depth']['scale_adjustment']=2.
            with self.assertRaises(ValueError):process_frame(paths,a,{'opencv_from_cameras_projection':projection},cropping)

    def test_exclusive_file_preserves_existing(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'frame.bin';exclusive_bytes(p,b'original');exclusive_bytes(p,b'original')
            with self.assertRaises(ValueError):exclusive_bytes(p,b'replacement')
            self.assertEqual(p.read_bytes(),b'original')

    def test_released_near_square_loader_keeps_both_orientations(self):
        base=BaseStereoViewDataset(split='test',resolution=(512,384),seed=777)
        image=PIL.Image.new('RGB',(512,499))
        K=np.array([[500,0,256],[0,500,249],[0,0,1]],np.float32)
        sizes=set()
        for seed in range(8):
            out,depth,_=base._crop_resize_if_necessary(image,np.ones((499,512),np.float32),K,
                                                      (512,384),rng=np.random.default_rng(seed))
            self.assertEqual(depth.shape,out.size[::-1])
            sizes.add(out.size)
        self.assertEqual(sizes,{(512,384),(384,512)})


if __name__=='__main__': unittest.main()
