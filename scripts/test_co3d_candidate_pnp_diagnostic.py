import unittest
from unittest.mock import patch
import numpy as np
import torch
from diagnose_co3d_candidate_pnp import confidence_mask, branch_poses, build_report, check_baseline
from fast3r_hf_re10k_pose_eval import pose_metrics


class DiagnosticTests(unittest.TestCase):
    def test_strict_percentile_and_ties(self):
        mask,stats=confidence_mask(torch.arange(100.).reshape(10,10),True)
        self.assertEqual(int(mask.sum()),15)
        self.assertEqual(stats['retained_points'],15)
        mask,stats=confidence_mask(torch.ones(10,10),True)
        self.assertEqual(int(mask.sum()),0)
        self.assertEqual(stats['at_threshold_points'],100)

    def test_released_mask(self):
        mask,_=confidence_mask(torch.tensor([[1.,2.],[0.,3.]]),False)
        self.assertEqual(mask.tolist(),[[False,True],[False,True]])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): confidence_mask(torch.tensor([[float('nan')]]),True)

    def test_search_failure_keeps_all_identity(self):
        preds=[{'pts3d_in_other_view':torch.ones(1,4,4,3),'conf':torch.ones(1,4,4)} for _ in range(10)]
        with patch('fast3r.dust3r.cloud_opt.init_im_poses.fast_pnp',return_value=(None,None)) as pnp:
            poses,stats=branch_poses(preds,0.,42,True,True)
        self.assertEqual(pnp.call_count,1)
        self.assertIsNone(stats['estimated_focal'])
        self.assertEqual(stats['pnp_failed_view_indices'],list(range(10)))
        self.assertTrue(np.array_equal(poses,np.repeat(np.eye(4)[None],10,axis=0)))

    def test_incomplete_report_rejected(self):
        with self.assertRaises(ValueError): build_report({0:{}},{})

    def test_baseline_mismatch_rejected(self):
        poses=np.repeat(np.eye(4,dtype=np.float32)[None],10,axis=0)
        metrics,_=pose_metrics(poses,poses)
        row={'estimated_focal':0.,'pnp_failed_view_indices':list(range(10)),
             'predicted_c2w':poses.tolist(),'gt_c2w':poses.tolist(),'metrics':metrics}
        self.assertEqual(check_baseline(poses,0.,list(range(10)),row),0.)
        with self.assertRaises(ValueError): check_baseline(poses,1.,list(range(10)),row)


if __name__=='__main__': unittest.main()
