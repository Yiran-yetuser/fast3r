"""Signed depth is allowed only on reference +Inf/all-zero quantization."""
import tempfile
import unittest
import numpy as np
from PIL import Image
import test_co3d_preprocess_probe as fixture
from co3d_lazy_dataset_v2 import process_frame_allow_zero as v2
from co3d_lazy_dataset_v3 import process_frame_allow_zero as v3


class V3Tests(unittest.TestCase):
    def make(self, d, value=None):
        paths,a=fixture.PreprocessProbeTests().fixture(d)
        if value is not None:
            bits=np.asarray(Image.open(paths['depths']),np.uint16).copy()
            bits[:]=np.float16(value).view(np.uint16)
            Image.fromarray(bits).save(paths['depths'])
        return paths,a,{'opencv_from_cameras_projection':fixture.projection}

    def test_finite_geometry_and_modalities_exact(self):
        with tempfile.TemporaryDirectory() as d:
            args=self.make(d);a,b=v2(*args),v3(*args)
            self.assertEqual(a[0].tobytes(),b[0].tobytes())
            for x,y in zip(a[1:6],b[1:6]):np.testing.assert_array_equal(x,y)

    def test_signed_inf_reference_zero(self):
        with tempfile.TemporaryDirectory() as d:
            paths,a,ns=self.make(d,np.inf)
            bits=np.asarray(Image.open(paths['depths']),np.uint16).copy()
            bits[5,5]=np.float16(-np.inf).view(np.uint16)
            bits[6,6]=np.float16(-12).view(np.uint16)
            Image.fromarray(bits).save(paths['depths'])
            result=v3(paths,a,ns)
            self.assertTrue(np.isposinf(result[5]));self.assertFalse(result[1].any())
            self.assertEqual(result[-1]['raw_negative_depth_count'],2)

    def test_finite_negative_negative_inf_and_nan_rejected(self):
        for value in (-1.,-np.inf,np.nan):
            with self.subTest(value=value),tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError):v3(*self.make(d,value))


if __name__=='__main__':unittest.main()
