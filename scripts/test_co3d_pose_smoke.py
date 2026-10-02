"""Offline synthetic model-input/orientation/metric tests, not paper scores."""
import unittest

import numpy as np
import torch

from fast3r_hf_co3d_pose_smoke import model_inputs, correct_orientation, signature, validate_smoke
from fast3r_hf_re10k_pose_eval import pose_metrics


class Co3dPoseSmokeTests(unittest.TestCase):
    def test_model_inputs_exclude_all_ground_truth(self):
        view={'img':torch.zeros(3,4,8),'true_shape':np.array([8,4],np.int32),
              'camera_pose':np.eye(4),'camera_intrinsics':np.eye(3),
              'depthmap':np.ones((4,8)),'pts3d':np.ones((4,8,3)),'valid_mask':np.ones((4,8),bool)}
        value=model_inputs([view])[0]
        self.assertEqual(set(value),{'img','true_shape'})
        self.assertEqual(value['img'].shape,(1,3,4,8))
        self.assertEqual(value['true_shape'].tolist(),[[8,4]])

    def test_portrait_corrected_once_and_original_not_mutated(self):
        coords=torch.arange(24).reshape(1,4,2,3).float();conf=torch.arange(8).reshape(1,4,2).float()
        preds=[{'pts3d_in_other_view':coords,'conf':conf}]
        inputs=[{'true_shape':torch.tensor([[4,2]]),'img':torch.zeros(1,3,2,4)}]
        output=correct_orientation(preds,inputs)[0]
        self.assertTrue(torch.equal(output['pts3d_in_other_view'],coords.transpose(1,2)))
        self.assertTrue(torch.equal(output['conf'],conf.transpose(1,2)))
        self.assertTrue(torch.equal(preds[0]['conf'],conf))

    def test_landscape_orientation_stays_unchanged(self):
        pred={'pts3d_in_other_view':torch.zeros(1,2,4,3),'conf':torch.ones(1,2,4)}
        out=correct_orientation([pred],[{'true_shape':torch.tensor([[2,4]]),'img':torch.zeros(1,3,2,4)}])[0]
        self.assertTrue(torch.equal(out['conf'],pred['conf']))

    def test_real_head_wrapper_portrait_contract(self):
        from fast3r.dust3r.utils.misc import transpose_to_landscape
        calls=[]
        def head(features,shape):
            calls.append(shape)
            return {'pts3d':torch.zeros(1,*shape,3),'conf':torch.ones(1,*shape)}
        true_shape=torch.tensor([[4,2]])
        raw=transpose_to_landscape(head,activate=False)([],true_shape)
        self.assertEqual(calls,[(4,2)])
        corrected=correct_orientation([{'pts3d_in_other_view':raw['pts3d'],'conf':raw['conf']}],
            [{'true_shape':true_shape,'img':torch.zeros(1,3,2,4)}])[0]
        self.assertEqual(corrected['conf'].shape,(1,2,4))

    def test_duplicate_pairs_retained_by_metric(self):
        gt=np.tile(np.eye(4,dtype=np.float32),(10,1,1));gt[:,0,3]=np.arange(10)
        gt[7]=gt[0]
        metrics,errors=pose_metrics(gt,gt)
        self.assertEqual(len(errors['rotation_deg']),45)
        self.assertEqual(len(errors['translation_deg']),45)
        self.assertLess(metrics['RTA_at_30'],1.) # zero-baseline behavior, not filtered

    def test_validation_rejects_formal_claim_or_changed_gt(self):
        gt=np.tile(np.eye(4,dtype=np.float32),(2,1,1));gt[1,0,3]=1
        proto={'scope':'smoke'};inputs={'frames':[{'label':'a'}, {'label':'a'}]}
        metrics,errors=pose_metrics(gt,gt)
        row={'scene':'a','protocol':proto,'protocol_sha256':signature(proto),'inputs':inputs,
             'metrics':metrics,'relative_errors':errors,'predicted_c2w':gt.tolist(),'gt_c2w':gt.tolist(),
             'pair_count':1,'pnp_failed_view_indices':[],'formal_pose_metrics_available':False,
             'all_100_draws_executed':False,'model_inference_completed':True}
        validate_smoke(row,inputs,proto,gt)
        row['formal_pose_metrics_available']=True
        with self.assertRaises(ValueError):validate_smoke(row,inputs,proto,gt)
        row['formal_pose_metrics_available']=False
        with self.assertRaises(ValueError):validate_smoke(row,inputs,proto,gt+1)


if __name__=='__main__':unittest.main()
