import unittest
from verify_co3d_51_source_storage import measure, assert_honesty, FALSE_FLAGS


class Independent51BudgetTests(unittest.TestCase):
    def test_independent_member_measurement(self):
        name='ball/s/images/frame000001.jpg'
        entry={'filename':name,'file_size':100,'compress_size':70,'compress_type':8,
               'flag_bits':0,'header_offset':0,'central_directory_offset':500,'archive_bytes':1000}
        actual=measure({name:entry},{name})
        self.assertEqual(actual['advertised_raw_bytes'],100)
        self.assertEqual(actual['member_count_by_kind'],{'images':1,'depths':0,'masks':0})
        bad=dict(entry);bad['file_size']=True
        with self.assertRaises(ValueError):measure({name:bad},{name})

    def test_missing_independent_member_fails(self):
        with self.assertRaises(ValueError):measure({}, {'ball/s/images/frame000001.jpg'})

    def test_each_promotion_flag_and_model_work_rejected(self):
        report={k:False for k in FALSE_FLAGS};report['model_forward_count']=0
        assert_honesty(report)
        for k in FALSE_FLAGS:
            bad=dict(report);bad[k]=True
            with self.assertRaises(ValueError):assert_honesty(bad)
        bad=dict(report);bad['model_forward_count']=1
        with self.assertRaises(ValueError):assert_honesty(bad)


if __name__ == '__main__':unittest.main()
