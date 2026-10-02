"""Synthetic precision checks, not model accuracy."""
import unittest
import numpy as np
from audit_co3d_pose_precision import poses, pair_diagnostic, aggregate


class PosePrecisionTests(unittest.TestCase):
    def views(self, translations):
        return [{'R': np.eye(3).tolist(), 'T': t} for t in translations]

    def test_cast_before_inverse_reference(self):
        v = self.views([[1, 2, 3]])
        p = poses(v, np.float32)
        self.assertEqual(str(p.dtype), 'torch.float32')
        np.testing.assert_array_equal(p[0, :3, 3], [-1, -2, -3])

    def test_ordinary_pairs_and_self_metric(self):
        r = pair_diagnostic(self.views([[0, 0, 0], [1, 2, 3], [2, 4, 7]]))
        self.assertEqual(r['pair_count'], 3)
        self.assertEqual(r['direction_defined_pairs'], 3)
        self.assertEqual(r['baseline_above_diagnostic_scale_threshold_pairs'], 3)
        self.assertLess(r['direction_disagreement_max_deg'], .01)
        self.assertLess(r['released_float32_identical_gt_translation_max_deg'], .1)

    def test_large_common_origin_cancellation_remains_visible(self):
        r = pair_diagnostic(self.views([[1e16, 0, 0], [1e16 + 10, 0, 0]]))
        self.assertEqual(r['nonzero64_collapsed32_pairs'], 1)
        self.assertEqual(r['float32_zero_translation_pairs'], 1)
        self.assertEqual(r['released_float32_identical_gt_translation_above_1deg_pairs'], 1)
        self.assertEqual(r['baseline_above_diagnostic_scale_threshold_pairs'], 0)

    def test_zero_baselines_are_counted_not_silently_dropped(self):
        r = pair_diagnostic(self.views([[1, 2, 3], [1, 2, 3]]))
        self.assertEqual(r['float64_zero_translation_pairs'], 1)
        self.assertEqual(r['direction_defined_pairs'], 0)
        self.assertEqual(r['float64_identical_camera_center_pairs'], 1)
        self.assertEqual(r['pair_count'], 1)

    def test_nonfinite_source_rejected(self):
        with self.assertRaises(ValueError): poses(self.views([[float('nan'), 0, 0]]), np.float32)

    def test_sum_and_max_aggregate(self):
        a = pair_diagnostic(self.views([[0, 0, 0], [1, 2, 3]]))
        r = aggregate([a, a])
        self.assertEqual(r['pair_count'], 2)
        self.assertEqual(r['direction_disagreement_max_deg'], a['direction_disagreement_max_deg'])


if __name__ == '__main__': unittest.main()
