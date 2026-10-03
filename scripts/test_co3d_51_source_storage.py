import unittest
import tempfile
from pathlib import Path

from plan_co3d_51_source_storage import nominal_frames, summarize, validate_entry, save_checkpoint


class Source51StorageTests(unittest.TestCase):
    def entry(self, name='ball/s/images/frame000001.jpg'):
        return {'filename': name, 'url':'https://dl.fbaipublicfiles.com/co3d/ball_001.zip',
            'expected_full_archive_sha256':'a'*64, 'file_size':100, 'compress_size':70,
            'flag_bits':0, 'compress_type':8, 'header_offset':5,
            'central_directory_offset':500, 'archive_bytes':1000, 'etag':'tag', 'CRC':123}

    def test_nominal_duplicates_do_not_inflate_unique_storage(self):
        plan = {'rows':[{'category':'ball','scene':'s','frame_numbers':[1,1,2]}]}
        self.assertEqual(nominal_frames(plan), {'ball':{'s':[1,2]}})

    def test_exact_size_aggregation(self):
        names = ['ball/s/images/frame000001.jpg','ball/s/depths/frame000001.jpg.geometric.png',
                 'ball/s/masks/frame000001.png']
        entries = {n:self.entry(n) for n in names}
        r = summarize(entries, set(names))
        self.assertEqual(r['advertised_raw_bytes'],300)
        self.assertEqual(r['advertised_compressed_bytes'],210)
        self.assertEqual(r['member_count_by_kind'],{'images':1,'depths':1,'masks':1})

    def test_missing_member_rejected(self):
        with self.assertRaises(ValueError): summarize({}, {'ball/s/images/frame000001.jpg'})

    def test_source_size_and_method_corruption_rejected(self):
        e=self.entry(); urls=[e['url']]; checksums={'ball_001.zip':'a'*64}
        validate_entry(e['filename'],e,urls,checksums)
        for key,value in (('file_size',0),('compress_size',40*1024**2),('flag_bits',1),
                          ('compress_type',99),('CRC',-1),('etag','')):
            bad=dict(e);bad[key]=value
            with self.assertRaises(ValueError): validate_entry(e['filename'],bad,urls,checksums)

    def test_foreign_url_or_checksum_rejected(self):
        e=self.entry()
        with self.assertRaises(ValueError): validate_entry(e['filename'],e,[],{'ball_001.zip':'a'*64})
        with self.assertRaises(ValueError): validate_entry(e['filename'],e,[e['url']],{'ball_001.zip':'b'*64})

    def test_checkpoint_is_immutable_and_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'receipt.json'
            save_checkpoint(path,{'budget':123})
            original=path.read_bytes()
            save_checkpoint(path,{'budget':123})
            self.assertEqual(path.read_bytes(),original)
            with self.assertRaises(ValueError): save_checkpoint(path,{'budget':124})
            self.assertEqual(path.read_bytes(),original)


if __name__ == '__main__': unittest.main()
