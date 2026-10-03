import copy
import unittest
from pathlib import Path
from fast3r_hf_re10k_pose_eval import sha
from test_co3d51_pose_eval_v1 import full_claims,synthetic_row
from verify_co3d51_candidate_pose_v1 import strict_header,inspect_pose,check_bound_source


class IndependentSource51PoseTests(unittest.TestCase):
    def test_actual_main_file_absolute_path_bound_inside_project(self):
        path=Path('scripts/fast3r_hf_co3d51_pose_eval_v1.py')
        check_bound_source(str(path),sha(path))
        check_bound_source(str(path.resolve()),sha(path))
        for bad in ('/etc/passwd','scripts/../README.md','README.md'):
            with self.assertRaises(ValueError):check_bound_source(bad,'bad')
        with self.assertRaises(ValueError):check_bound_source(str(path),'bad')

    def test_header_rejects_partial_counts_and_paper_promotion(self):
        proof,prep=full_claims()
        row={'request_count':1000,'total_pair_count':45000,
            'request_checkpoint_sha256':proof['request_sha256'],
            'formal_Table1_result':False,'full_paper_completed':False,
            'author_protocol_equivalence_verified':False,'paper_checkpoint_mapping_verified':False,
            'saved_predictions_independently_reinferred':False}
        strict_header(row,proof,prep)
        for key,value in [('request_count',100),('total_pair_count',44999),
            ('formal_Table1_result',True),('saved_predictions_independently_reinferred',True)]:
            bad=copy.deepcopy(row);bad[key]=value
            with self.assertRaises(ValueError):strict_header(bad,proof,prep)
        bad=copy.deepcopy(proof);bad['expected_full_request_count']=999
        with self.assertRaises(ValueError):strict_header(row,bad,prep)
        for key,value in [('model_forward_count',1),('formal_Table1_result',True)]:
            bad=copy.deepcopy(prep);bad[key]=value
            with self.assertRaises(ValueError):strict_header(row,proof,bad)

    def test_each_saved_input_and_pair_metric_corruption_rejected(self):
        row,proto,proof,prep=synthetic_row()
        args=(prep,0,row['protocol_sha256'],'b'*64,42)
        result,r,t,duplicates=inspect_pose(row,*args)
        self.assertEqual((len(r),len(t),duplicates),(45,45,[[0,7]]))
        for key,value in [('input_tensor_sha256',['bad']*10),('duplicate_camera_pairs',[]),
            ('request_seed',0),('formal_Table1_result',True)]:
            bad=copy.deepcopy(row);bad[key]=value
            with self.assertRaises(ValueError):inspect_pose(bad,*args)
        bad=copy.deepcopy(row);bad['metrics']['mAA_30']+=.1
        with self.assertRaises(ValueError):inspect_pose(bad,*args)
        bad=copy.deepcopy(row);bad['relative_errors']['translation_deg'][0]+=.01
        with self.assertRaises(ValueError):inspect_pose(bad,*args)

    def test_independent_identity_and_GT_checks(self):
        row,_,_,prep=synthetic_row();args=(prep,0,row['protocol_sha256'],'b'*64,42)
        bad=copy.deepcopy(row);bad['pnp_failed_view_indices']=[1]
        with self.assertRaisesRegex(ValueError,'Identity fallback'):inspect_pose(bad,*args)
        bad=copy.deepcopy(row);bad['gt_c2w'][0][0][3]=1.
        with self.assertRaises(ValueError):inspect_pose(bad,*args)


if __name__=='__main__':unittest.main()
