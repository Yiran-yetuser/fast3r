import unittest
import numpy as np
from diagnose_co3d_nonfinite_depth import reference_quantize


class NonfiniteDiagnosisTests(unittest.TestCase):
    def test_positive_inf_reference_quantizes_to_zero_not_manual_fill(self):
        q,maximum=reference_quantize(np.array([[1.,2.],[float('inf'),0.]],np.float32))
        self.assertTrue(np.isposinf(maximum));self.assertTrue((q==0).all())
        effective=q.astype(np.float32)/65535*np.nan_to_num(maximum)
        self.assertTrue(np.isfinite(effective).all());self.assertTrue((effective==0).all())
    def test_finite_reference_arithmetic_unchanged(self):
        d=np.array([[1.,2.],[4.,0.]],np.float32)
        q,maximum=reference_quantize(d)
        np.testing.assert_array_equal(q,(d/d.max()*65535).astype(np.uint16))
        self.assertEqual(maximum,4.)


if __name__=='__main__':unittest.main()
