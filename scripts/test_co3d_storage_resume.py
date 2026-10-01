"""Synthetic journal verification/migration tests, not dataset metrics."""
import copy
import hashlib
import unittest

from plan_co3d_seen41_storage import V1_CODE_SHA, expected_paths, validate_saved


class StorageJournalTests(unittest.TestCase):
    def fixture(self):
        selected={'s':[1]}
        fingerprint={'candidate_manifest_sha256':'input','protocol_sha256':'protocol',
                     'links_sha256':'links','checksums_sha256':'checksums',
                     'planner_sha256':'newplanner','range_reader_sha256':'newreader'}
        url='https://example.invalid/apple_001.zip'
        count={k:1 for k in ('images','depths','masks')}
        size={k:9 for k in count}
        record={'url':url,'expected_full_archive_sha256':'official','archive_bytes':1000,
                'index_transport_bytes':10,'etag':'etag','ranges':[{'start':900,'end':909}],
                'matched_member_counts':count,'matched_uncompressed_bytes':size}
        saved={'fingerprint':fingerprint,'category':'apple',
               'status':'complete_category_name_and_size_inventory_not_data_ready',
               'expected_frame_count':1,'cross_archive_duplicate_count':0,
               'expected_paths_sha256':hashlib.sha256('\n'.join(sorted(expected_paths('apple',selected))).encode()).hexdigest(),
               'full_archive_sha_verified':False,'member_crc_verified':False,
               'archives':[record],'matched_member_counts':count,'matched_uncompressed_bytes':size,
               'index_transport_bytes':10}
        return saved,selected,[url],{'apple_001.zip':'official'},dict(fingerprint)

    def check(self,saved,selected,urls,checksums,fingerprint,legacy=False):
        validate_saved(saved,'apple',selected,urls,checksums,fingerprint,legacy)

    def test_same_version_and_known_v1(self):
        saved,*rest=self.fixture()
        self.check(saved,*rest)
        saved=copy.deepcopy(saved)
        saved['fingerprint'].update(V1_CODE_SHA)
        self.check(saved,*rest,legacy=True)
        with self.assertRaises(ValueError):self.check(saved,*rest)

    def test_changed_input_or_unapproved_code_rejected(self):
        saved,*rest=self.fixture()
        for key in ('candidate_manifest_sha256','planner_sha256','range_reader_sha256'):
            altered=copy.deepcopy(saved)
            altered['fingerprint'][key]='changed'
            with self.assertRaises(ValueError):self.check(altered,*rest)

    def test_paths_or_aggregate_corruption_rejected(self):
        saved,*rest=self.fixture()
        for change in ('paths','sum','bounds','official'):
            altered=copy.deepcopy(saved)
            if change=='paths':altered['expected_paths_sha256']='other'
            if change=='sum':altered['index_transport_bytes']=11
            if change=='bounds':altered['archives'][0]['ranges'][0]['end']=1001
            if change=='official':altered['archives'][0]['expected_full_archive_sha256']='other'
            with self.assertRaises(ValueError):self.check(altered,*rest)


if __name__=='__main__':unittest.main()
