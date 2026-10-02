"""Small synthetic journals: verification tests, NOT real dataset scores."""
import copy
import hashlib
import unittest

from verify_co3d_storage_budget import verify_category


class IndependentBudgetTests(unittest.TestCase):
    def fixture(self):
        selected = {'scene': list(range(10))}
        paths = {f'apple/scene/{kind}/frame{n:06d}{suffix}' for n in range(10)
                 for kind, suffix in [('images', '.jpg'), ('depths', '.jpg.geometric.png'), ('masks', '.png')]}
        url = 'https://example.invalid/apple_001.zip'
        record = {'url': url, 'expected_full_archive_sha256': 'checksum', 'etag': 'etag',
                  'archive_bytes': 1000, 'zip_member_count': 30, 'index_transport_bytes': 100,
                  'ranges': [{'start': 900, 'end': 999, 'sha256': '0' * 64}],
                  'matched_member_counts': {k: 10 for k in ('images', 'depths', 'masks')},
                  'matched_uncompressed_bytes': {k: 200 for k in ('images', 'depths', 'masks')}}
        saved = {'category': 'apple', 'fingerprint': {'code': 'hash'},
                 'status': 'complete_category_name_and_size_inventory_not_data_ready',
                 'expected_frame_count': 10, 'cross_archive_duplicate_count': 0,
                 'expected_paths_sha256': hashlib.sha256('\n'.join(sorted(paths)).encode()).hexdigest(),
                 'full_archive_sha_verified': False, 'member_crc_verified': False,
                 'archives': [record], 'index_transport_bytes': 100,
                 'matched_member_counts': record['matched_member_counts'].copy(),
                 'matched_uncompressed_bytes': record['matched_uncompressed_bytes'].copy()}
        footer = {url: {'etag': 'etag', 'archive_bytes': 1000, 'member_count': 30, 'central_directory_bytes': 100}}
        return saved, 'apple', selected, [url], {'apple_001.zip': 'checksum'}, {'code': 'hash'}, footer

    def test_valid_complete(self):
        verify_category(*self.fixture())

    def test_coverage_and_honesty_flags(self):
        args = self.fixture()
        for key, value in [('expected_frame_count', 9), ('expected_paths_sha256', 'wrong'),
                           ('cross_archive_duplicate_count', 1), ('member_crc_verified', True)]:
            changed = copy.deepcopy(args); changed[0][key] = value
            with self.assertRaises(ValueError): verify_category(*changed)

    def test_identity_and_bounds(self):
        args = self.fixture()
        for key, value in [('etag', 'other'), ('archive_bytes', 1001), ('zip_member_count', 31),
                           ('index_transport_bytes', 70000)]:
            changed = copy.deepcopy(args); changed[0]['archives'][0][key] = value
            with self.assertRaises(ValueError): verify_category(*changed)

    def test_bad_aggregate_and_range(self):
        args = self.fixture()
        changed = copy.deepcopy(args); changed[0]['matched_uncompressed_bytes']['images'] += 1
        with self.assertRaises(ValueError): verify_category(*changed)
        changed = copy.deepcopy(args); changed[0]['archives'][0]['ranges'][0]['end'] = 1000
        with self.assertRaises(ValueError): verify_category(*changed)

    def test_duplicate_candidates(self):
        args = self.fixture(); args[2]['scene'][-1] = 0
        with self.assertRaises(ValueError): verify_category(*args)


if __name__ == '__main__': unittest.main()
