"""V2 positive-Inf semantics regressions; synthetic fixtures, not scores."""
import tempfile
import unittest
import numpy as np
from PIL import Image
from test_co3d_preprocess_probe import PreprocessProbeTests, projection
from co3d_lazy_dataset import process_frame_allow_zero as v1
from co3d_lazy_dataset_v2 import process_frame_allow_zero as v2, audit_maximum


class V2Tests(unittest.TestCase):
    def test_finite_outputs_equal_v1(self):
        with tempfile.TemporaryDirectory() as d:
            paths, a = PreprocessProbeTests().fixture(d)
            ns = {'opencv_from_cameras_projection': projection}
            old, new = v1(paths, a, ns), v2(paths, a, ns)
            self.assertEqual(old[0].tobytes(), new[0].tobytes())
            for left, right in zip(old[1:6], new[1:6]):
                np.testing.assert_array_equal(left, right)

    def test_positive_inf_is_not_replaced_before_reference_arithmetic(self):
        with tempfile.TemporaryDirectory() as d:
            paths, a = PreprocessProbeTests().fixture(d)
            bits = np.asarray(Image.open(paths['depths']), np.uint16).copy()
            bits[:] = np.float16(np.inf).view(np.uint16)
            Image.fromarray(bits).save(paths['depths'])
            result = v2(paths, a, {'opencv_from_cameras_projection': projection})
            self.assertTrue(np.isposinf(result[5]))
            self.assertFalse(result[1].any())
            self.assertTrue(result[-1]['positive_inf_reference_quantization'])
            self.assertEqual(audit_maximum(result[5]), 'positive_infinity')

    def test_nan_and_negative_inf_remain_hard_errors(self):
        for value in (np.nan, -np.inf, -1.):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as d:
                paths, a = PreprocessProbeTests().fixture(d)
                bits = np.asarray(Image.open(paths['depths']), np.uint16).copy()
                bits[:] = np.float16(value).view(np.uint16)
                Image.fromarray(bits).save(paths['depths'])
                with self.assertRaises(ValueError):
                    v2(paths, a, {'opencv_from_cameras_projection': projection})


if __name__ == '__main__': unittest.main()
