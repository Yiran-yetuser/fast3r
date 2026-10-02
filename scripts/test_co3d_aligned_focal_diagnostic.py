"""Synthetic isolation tests; not paper scores."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from diagnose_co3d_pose_aligned_focal import evaluate_branches, fixed_focal_poses, aligned_focal_maps, BRANCHES


class AlignedFocalDiagnosticTests(unittest.TestCase):
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
        with patch('diagnose_co3d_pose_aligned_focal.aligned_focal_maps',return_value=[{'pts3d_in_other_view':p['pts3d_local'],'conf':p['conf_local']} for p in preds]), patch('fast3r.models.multiview_dust3r_module.estimate_focal',side_effect=focal), patch('diagnose_co3d_pose_aligned_focal.fixed_focal_poses',side_effect=pnp):
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

    def test_released_alignment_known_similarity_no_source_mutation(self):
        x=torch.randn((1,4,5,3),generator=torch.Generator().manual_seed(7))
        y=2*x+torch.tensor([1.,2.,3.])
        p={'pts3d_local':x,'pts3d_in_other_view':y,'conf':torch.ones(1,4,5)*2,
           'conf_local':torch.ones(1,4,5)*3}
        maps=aligned_focal_maps([p],[{'img':torch.zeros(1,3,4,5),'true_shape':torch.tensor([[4,5]])}])
        torch.testing.assert_close(maps[0]['pts3d_in_other_view'],y,rtol=1e-5,atol=1e-5)
        self.assertNotIn('pts3d_local_aligned_to_global',p)
        self.assertTrue(torch.equal(x,p['pts3d_local']))

    def test_gt_mask_rejected(self):
        with self.assertRaises(ValueError):
            aligned_focal_maps([], [{'img':None,'true_shape':None,'valid_mask':None}])


if __name__=='__main__': unittest.main()
