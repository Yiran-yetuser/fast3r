"""Finite signed source arithmetic is retained, not filled or skipped."""
import tempfile
import unittest
import numpy as np
from PIL import Image
import test_co3d_preprocess_probe as fixture
from co3d_lazy_dataset_v3 import process_frame_allow_zero as v3
from co3d_lazy_dataset_v4 import process_frame_allow_zero as v4


class V4Tests(unittest.TestCase):
    def make(self, d):
        paths,a=fixture.PreprocessProbeTests().fixture(d)
        return paths,a,{'opencv_from_cameras_projection':fixture.projection}

    def test_positive_outputs_exact_v3(self):
        with tempfile.TemporaryDirectory() as d:
            args=self.make(d);a,b=v3(*args),v4(*args)
            self.assertEqual(a[0].tobytes(),b[0].tobytes())
            for x,y in zip(a[1:6],b[1:6]):np.testing.assert_array_equal(x,y)

    def test_finite_signed_pixels_survive_reference_uint16_cast(self):
        with tempfile.TemporaryDirectory() as d:
            paths,a,ns=self.make(d)
            bits=np.asarray(Image.open(paths['depths']),np.uint16).copy()
            bits[5:15,5:15]=np.float16(-1).view(np.uint16)
            Image.fromarray(bits).save(paths['depths'])
            result=v4(paths,a,ns)
            self.assertTrue(np.isfinite(result[5]));self.assertGreater(result[5],0)
            self.assertGreater(result[-1]['processed_negative_depth_count'],0)
            self.assertEqual(result[-1]['processed_negative_depth_count'],result[-1]['negative_quantized_positive_count'])

    def test_nan_and_unverified_negative_inf_finite_max_rejected(self):
        for value in (np.nan,-np.inf):
            with self.subTest(value=value),tempfile.TemporaryDirectory() as d:
                paths,a,ns=self.make(d)
                bits=np.asarray(Image.open(paths['depths']),np.uint16).copy()
                bits[5:15,5:15]=np.float16(value).view(np.uint16)
                Image.fromarray(bits).save(paths['depths'])
                with self.assertRaises(ValueError):v4(paths,a,ns)

    def test_signed_inf_geometry_exact_v3(self):
        with tempfile.TemporaryDirectory() as d:
            paths,a,ns=self.make(d)
            bits=np.asarray(Image.open(paths['depths']),np.uint16).copy()
            bits[:]=np.float16(np.inf).view(np.uint16)
            bits[5,5]=np.float16(-np.inf).view(np.uint16)
            Image.fromarray(bits).save(paths['depths'])
            left,right=v3(paths,a,ns),v4(paths,a,ns)
            for x,y in zip(left[1:6],right[1:6]):np.testing.assert_array_equal(x,y)


if __name__=='__main__':unittest.main()
