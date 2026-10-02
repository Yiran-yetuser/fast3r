"""Synthetic isolation checks, not paper performance."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from diagnose_co3d_pose_orientation import evaluate_branches, BRANCHES


class OrientationDiagnosticTests(unittest.TestCase):
    def test_same_predictions_two_shapes_no_gt_passed_to_PnP(self):
        preds=[{'pts3d_in_other_view':torch.zeros(1,4,2,3),
                'conf':torch.ones(1,4,2)} for _ in range(10)]
        views=[{'true_shape':torch.tensor([[4,2]]),'img':torch.zeros(1,3,2,4)} for _ in range(10)]
        gt=np.tile(np.eye(4,dtype=np.float32),(10,1,1));calls=[]
        def fake(values,seed,niter):
            self.assertEqual(set(values[0]),{'pts3d_in_other_view','conf'})
            calls.append((values[0]['conf'].shape,seed,niter))
            return gt.copy(),10.,[]
        with patch('diagnose_co3d_pose_orientation.predict_poses',side_effect=fake):
            rows=evaluate_branches(preds,views,123,gt)
        self.assertEqual(set(rows),set(BRANCHES))
        self.assertEqual(calls,[(torch.Size([1,2,4]),123,100),(torch.Size([1,4,2]),123,100)])
        self.assertEqual(preds[0]['conf'].shape,(1,4,2))
        self.assertTrue(all(r['pair_count']==45 for r in rows.values()))


if __name__=='__main__':unittest.main()
