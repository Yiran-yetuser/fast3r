import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from co3d51_source_lazy_v1 import (atomic_payload,checked_payload,check_budget,
    RAW_LIMIT,PROCESSED_LIMIT,RESERVE,Source51Landscape)
from prepare_co3d51_source_v1 import JOURNAL_LIMIT,validate_result


class Source51ActualPreparationTests(unittest.TestCase):
    def test_atomic_raw_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'raw.png'
            atomic_payload(p,b'abc')
            with self.assertRaises(FileExistsError):atomic_payload(p,b'wrong')
            self.assertEqual(p.read_bytes(),b'abc')

    def test_interrupted_receipt_can_verify_atomic_raw_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'raw.png';atomic_payload(p,b'abc')
            entry={'file_size':3,'CRC':zlib.crc32(b'abc')&0xffffffff}
            self.assertEqual(len(checked_payload(p,entry)),64)
            bad=dict(entry);bad['CRC']=0
            with self.assertRaises(ValueError):checked_payload(p,bad)
            bad=dict(entry);bad['file_size']=2
            with self.assertRaises(ValueError):checked_payload(p,bad)
            self.assertEqual(p.read_bytes(),b'abc')

    def test_explicit_new_caps_leave_historical_caps_unchanged(self):
        from co3d_range_cache import MAX_CACHE
        from co3d_lazy_dataset_v3 import MAX_PROCESSED
        self.assertEqual((RAW_LIMIT,PROCESSED_LIMIT,JOURNAL_LIMIT),(8*1024**3,3*1024**3,4*1024**3))
        self.assertEqual((MAX_CACHE,MAX_PROCESSED),(2*1024**3,512*1024**2))

    def test_byte_ceiling_or_free_reserve_rejects(self):
        with mock.patch('co3d51_source_lazy_v1.shutil.disk_usage') as usage:
            usage.return_value=mock.Mock(free=2*RESERVE)
            check_budget(0,1,10,'.')
            with self.assertRaises(RuntimeError):check_budget(10,1,10,'.')
            usage.return_value=mock.Mock(free=RESERVE)
            with self.assertRaises(RuntimeError):check_budget(0,1,10,'.')

    def test_landscape_crop_does_not_reverse_portrait_target(self):
        d=Source51Landscape.__new__(Source51Landscape);d.aug_crop=0
        image=Image.new('RGB',(600,1000));depth=np.ones((1000,600),np.float32)
        K=np.array([[500,0,300],[0,500,500],[0,0,1]],np.float32)
        im,dm,k=d._crop_resize_if_necessary(image,depth,K,(512,384),np.random.default_rng(1))
        self.assertEqual(im.size,(512,384));self.assertEqual(dm.shape,(384,512))
        self.assertGreater(np.linalg.det(k),0)

    def test_transaction_rejects_paper_promotion(self):
        with self.assertRaises(ValueError):validate_result({'request':0,'returned_views':[{}]*10,
            'model_forward_count':0,'formal_Table1_result':True,'load_trace':[]},0)

    def test_transaction_rejects_hard_error(self):
        with self.assertRaises(ValueError):validate_result({'request':0,'returned_views':[{}]*10,
            'model_forward_count':0,'formal_Table1_result':False,'load_trace':[{'status':'hard_error'}]},0)


if __name__=='__main__':unittest.main()
