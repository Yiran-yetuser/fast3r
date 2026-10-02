"""Synthetic isolation tests; not paper scores."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from diagnose_co3d_pose_focal import evaluate_branches, fixed_focal_poses, BRANCHES


class FocalDiagnosticTests(unittest.TestCase):
    def test_same_global_points_with_two_predicted_focals_no_gt_in_pnp(self):
        preds = [{'pts3d_in_other_view':torch.ones(1,4,2,3),
                  'conf':torch.ones(1,4,2), 'pts3d_local':torch.full((1,4,2,3),2.),
                  'conf_local':torch.full((1,4,2),3.)} for _ in range(10)]
        views = [{'true_shape':torch.tensor([[4,2]]), 'img':torch.zeros(1,3,2,4)} for _ in range(10)]
        gt = np.tile(np.eye(4,dtype=np.float32),(10,1,1)); calls=[]
        def focal(points, confidence, min_conf_thr_percentile):
            self.assertEqual(min_conf_thr_percentile,10)
            return float(points.mean())*100
        def pnp(values, f, seed):
            self.assertEqual(set(values[0]), {'pts3d_in_other_view','conf'})
            self.assertTrue(torch.all(values[0]['pts3d_in_other_view']==1))
            calls.append((values,f,seed))
            return gt.copy(),[]
        with patch('fast3r.models.multiview_dust3r_module.estimate_focal',side_effect=focal), patch('diagnose_co3d_pose_focal.fixed_focal_poses',side_effect=pnp):
            rows=evaluate_branches(preds,views,123,gt)
        self.assertEqual(set(rows),set(BRANCHES))
        self.assertIs(calls[0][0],calls[1][0]); self.assertIs(calls[2][0],calls[3][0])
        self.assertEqual([c[1] for c in calls],[100.,200.,100.,200.])
        self.assertEqual(preds[0]['conf'].shape,(1,4,2))
        self.assertTrue(all(r['pair_count']==45 for r in rows.values()))

    def test_invalid_focal_rejected(self):
        for focal in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError): fixed_focal_poses([],focal,1)

    def test_identity_fallback_is_counted(self):
        pred={'pts3d_in_other_view':torch.ones(1,2,2,3),'conf':torch.ones(1,2,2)*2}
        with patch('fast3r.dust3r.cloud_opt.init_im_poses.fast_pnp',return_value=(None,None)):
            poses,failures=fixed_focal_poses([pred],100.,42)
        self.assertEqual(failures,[0]); np.testing.assert_array_equal(poses[0],np.eye(4))


if __name__=='__main__': unittest.main()
