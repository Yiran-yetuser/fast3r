import unittest
from unittest import mock
from verify_co3d51_source_inputs_v1 import safe_relative,ReadOnlyInputs,compare_replay


class Source51InputReplayTests(unittest.TestCase):
    def test_readonly_rejects_unsafe_paths(self):
        for name in ('/absolute/a/b','../a/b','a/../b','a\\b/c/d'):
            with self.assertRaises(ValueError):safe_relative(name)
        self.assertEqual(safe_relative('bowl/scene/images/frame000001.jpg'),
            'bowl/scene/images/frame000001.jpg')

    def test_readonly_rejects_conflicting_receipts(self):
        key='bowl/scene/images/frame000001.jpg'
        rows=[{'result':{'prepared_frames':[{'image_path':key,'sha':1},{'image_path':key,'sha':2}]}}]
        with self.assertRaises(ValueError):ReadOnlyInputs(rows,{})

    def test_shared_state_mismatch_rejected(self):
        dataset=mock.Mock();dataset.preparer=mock.Mock()
        row={'before_state_sha256':'good','result':{'returned_views':[],
            'load_trace':[],'pool_attempts':[],'prepared_frames':[],'base_index':0},'after_state':{'n':1}}
        with mock.patch('verify_co3d51_source_inputs_v1.capture',side_effect=[{},{}]), \
             mock.patch('verify_co3d51_source_inputs_v1.digest',return_value='good'), \
             mock.patch('verify_co3d51_source_inputs_v1.inputs',return_value=[]):
            with self.assertRaisesRegex(ValueError,'after-state'):
                compare_replay(dataset,[0],{},[row],[[]])

    def test_input_mismatch_rejected_before_state_promotion(self):
        dataset=mock.Mock();dataset.preparer=mock.Mock()
        row={'before_state_sha256':'good','result':{'returned_views':['old']}}
        with mock.patch('verify_co3d51_source_inputs_v1.capture',return_value={}), \
             mock.patch('verify_co3d51_source_inputs_v1.digest',return_value='good'), \
             mock.patch('verify_co3d51_source_inputs_v1.inputs',return_value=['changed']):
            with self.assertRaisesRegex(ValueError,'returned_views'):
                compare_replay(dataset,[0],{},[row],[[]])


if __name__=='__main__':unittest.main()
