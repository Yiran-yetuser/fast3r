import copy
import unittest
from verify_re10k_missing_full_source import check_report,TOTAL,SOURCE_SHA,REVISION


class MissingSourceVerificationTests(unittest.TestCase):
    def setUp(self):
        self.index={'missing_scene_ids':['a'],'covered_scene_ids':['b']}
        self.report={'status':'source_scanned_candidates_not_promoted','full_archive_sha_verified':True,
            'source_expected_bytes':TOTAL,'compressed_bytes_read':TOTAL,
            'source_expected_sha256':SOURCE_SHA,'compressed_sha256':SOURCE_SHA,'revision':REVISION,
            'target_ids_not_observed':['a'],'missing_target_scene_count':1,'target_ids_observed':[],
            'complete_candidate_timestamp_scene_ids':[],'frames':{},'extra_timestamps':{},
            'staged_frame_count':0,'formal_pose_metrics_available':False,'full_prescribed_split_ready':False,
            'official_metadata_sha256':{'a':'fixture'}}
    def test_completed_no_recovery_consistency(self):
        check_report(self.report,self.index,['a','b'])
    def test_truncated_or_wrong_sha_rejected(self):
        for key,value in [('compressed_bytes_read',TOTAL-1),('compressed_sha256','bad'),('full_archive_sha_verified',False)]:
            r=copy.deepcopy(self.report);r[key]=value
            with self.assertRaises(ValueError):check_report(r,self.index,['a','b'])
    def test_changed_set_or_promotion_rejected(self):
        for key,value in [('target_ids_not_observed',['c']),('full_prescribed_split_ready',True),('staged_frame_count',1)]:
            r=copy.deepcopy(self.report);r[key]=value
            with self.assertRaises(ValueError):check_report(r,self.index,['a','b'])


if __name__=='__main__':unittest.main()
