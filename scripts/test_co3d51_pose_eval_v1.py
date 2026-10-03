import argparse
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import fast3r_hf_co3d51_pose_eval_v1 as runner
from fast3r_hf_co3d_pose_smoke import model_inputs,signature
from fast3r_hf_re10k_pose_eval import pose_metrics,METRICS


def full_claims():
    names={f'request_{i:03d}.json':'a'*64 for i in range(1000)}
    p={'verified_request_count':1000,'expected_full_request_count':1000,
        'all_1000_requests_prepared_and_replayed':True,
        'raw_SHA_CRC_processed_SHA_and_receipts_checked':True,
        'input_tensor_GT_rng_trace_and_shared_state_exact_equal':True,
        'network_bytes':0,'model_forward_count':0,'author_protocol_equivalence_verified':False,
        'formal_Table1_result':False,'full_paper_completed':False,'request_sha256':names}
    s={'request_count':1000,'model_forward_count':0,'author_protocol_equivalence_verified':False,
        'formal_Table1_result':False,'full_paper_completed':False,'request_sha256':names}
    return p,s


def synthetic_row():
    gt=np.tile(np.eye(4,dtype=np.float32),(10,1,1));gt[:,0,3]=np.arange(10);gt[7]=gt[0]
    metrics,errors=pose_metrics(gt,gt)
    views=[{'label':'cat/scene','instance':str(0 if j==7 else j),
        'img_tensor_sha256':'a'*64,'img_shape':[3,384,512],'true_shape_hw':[384,512],
        'camera_pose':g.tolist()} for j,g in enumerate(gt)]
    proto={'seed':42};proof={'request_sha256':{'request_000.json':'b'*64}}
    prep={'base_index':123,'returned_views':views}
    row={'request':0,'protocol_sha256':signature(proto),'preparation_request_sha256':'b'*64,
        'base_index':123,'scene':'cat/scene','request_seed':42,
        'input_tensor_sha256':['a'*64]*10,'input_shape':[[1,3,384,512]]*10,
        'true_shape':[[[384,512]]]*10,'gt_c2w':gt.tolist(),'predicted_c2w':gt.tolist(),
        'metrics':metrics,'relative_errors':errors,'pair_count':45,'estimated_focal':100.,
        'duplicate_camera_pairs':[[0,7]],'pnp_failed_view_indices':[],
        'model_inference_completed':True,'formal_Table1_result':False}
    return row,proto,proof,prep


class Source51PoseAdapterTests(unittest.TestCase):
    def test_requires_full1000_not_old100_or_prefix1(self):
        p,s=full_claims();runner.assert_full_input_claims(p,s)
        for count in (1,100,999):
            q=copy.deepcopy(p);q['verified_request_count']=count
            with self.assertRaises(ValueError):runner.assert_full_input_claims(q,s)

    def test_every_proof_honesty_and_identity_flag_required(self):
        p,s=full_claims()
        for key in ('all_1000_requests_prepared_and_replayed','raw_SHA_CRC_processed_SHA_and_receipts_checked',
            'input_tensor_GT_rng_trace_and_shared_state_exact_equal','formal_Table1_result',
            'author_protocol_equivalence_verified','full_paper_completed'):
            q=copy.deepcopy(p);q[key]=not q[key]
            with self.assertRaises(ValueError):runner.assert_full_input_claims(q,s)
            q=copy.deepcopy(p);del q[key]
            with self.assertRaises(ValueError):runner.assert_full_input_claims(q,s)

    def test_missing_or_changed_request_set_rejected(self):
        p,s=full_claims();q=copy.deepcopy(p);q['request_sha256'].pop('request_999.json')
        with self.assertRaises(ValueError):runner.assert_full_input_claims(q,s)
        q=copy.deepcopy(s);q['request_sha256']['request_000.json']='changed'
        with self.assertRaises(ValueError):runner.assert_full_input_claims(p,q)

    def test_non_dry_run_refuses_model_path_when_preparation_pending(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(runner,'PREP_SUMMARY',Path(directory)/'missing'), \
             mock.patch.object(runner.subprocess,'check_output') as gpu_query:
            with self.assertRaises(runner.PreparationPending):
                runner.run(argparse.Namespace(dry_run=False,verify_only=False,save_gate_snapshot=False))
            gpu_query.assert_not_called()

    def test_dry_readiness_does_not_promote_observed_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'request_000.json').write_text('{}')
            with mock.patch.object(runner,'PREP_ROOT',root),mock.patch.object(runner,'PREP_SUMMARY',root/'missing'), \
                 mock.patch.object(runner,'INPUT_PROOF',root/'proof'):
                snapshot=runner.readiness_snapshot()
                self.assertEqual(snapshot['observed_committed_request_file_count'],1)
                self.assertTrue(snapshot['observed_prefix_count_is_not_independent_input_replay'])
                self.assertFalse(snapshot['model_loaded']);self.assertFalse(snapshot['formal_Table1_result'])

    def test_noncontiguous_preparation_filename_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'request_001.json').write_text('{}')
            with mock.patch.object(runner,'PREP_ROOT',root):
                with self.assertRaises(ValueError):runner.readiness_snapshot()

    def test_new_source_inputs_still_exclude_all_GT(self):
        view={'img':torch.zeros(3,384,512),'true_shape':np.array([384,512]),
            'camera_pose':np.eye(4),'camera_intrinsics':np.eye(3),'depthmap':np.ones((1,1))}
        self.assertEqual(set(model_inputs([view])[0]),{'img','true_shape'})

    def test_saved_pose_input_GT_duplicate_and_scope_checks(self):
        row,proto,proof,prep=synthetic_row();runner.validate_pose_row(row,0,proto,proof,prep)
        for key,value in [('formal_Table1_result',True),('request_seed',43),
                ('input_tensor_sha256',['wrong']*10),('duplicate_camera_pairs',[])]:
            bad=copy.deepcopy(row);bad[key]=value
            with self.assertRaises(ValueError):runner.validate_pose_row(bad,0,proto,proof,prep)
        bad=copy.deepcopy(row);bad['gt_c2w'][0][0][3]=10.
        with self.assertRaises(ValueError):runner.validate_pose_row(bad,0,proto,proof,prep)

    def test_fallback_not_silently_replaced(self):
        row,proto,proof,prep=synthetic_row();row['pnp_failed_view_indices']=[1]
        with self.assertRaisesRegex(ValueError,'identity fallback'):
            runner.validate_pose_row(row,0,proto,proof,prep)

    def test_all45_pairs_retained_and_partial_aggregate_refused(self):
        row,*_=synthetic_row();a=runner.aggregate([row],expected=1)
        self.assertEqual(a['total_pair_count'],45);self.assertEqual(a['duplicate_pair_count'],1)
        self.assertEqual(set(a['macro_mean_request_metrics']),set(METRICS))
        with self.assertRaises(ValueError):runner.aggregate([row])
        bad=copy.deepcopy(row);bad['relative_errors']['rotation_deg'].pop()
        with self.assertRaises(ValueError):runner.aggregate([bad],expected=1)

    def test_pose_byte_cap_and_free_reserve_reject(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(runner,'PROGRESS',Path(directory)), \
             mock.patch.object(runner.shutil,'disk_usage') as usage:
            usage.return_value=mock.Mock(free=runner.RESERVE+1)
            runner.pose_budget(1)
            with self.assertRaises(RuntimeError):runner.pose_budget(2)
            usage.return_value=mock.Mock(free=4*runner.RESERVE)
            with self.assertRaises(RuntimeError):runner.pose_budget(runner.MAX_POSE_JOURNAL+1)


if __name__=='__main__':unittest.main()
