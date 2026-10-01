"""Synthetic selection tests, not CO3D camera pose measurements."""
import unittest
import random
import tempfile
from pathlib import Path

from prepare_co3d_test_metadata import select, write_identical


class MetadataSelectionTests(unittest.TestCase):
    def fixture(self,count=3):
        scenes=[f'seq{i}' for i in range(count)]
        rows=[[s,f,f'apple/{s}/images/frame{f:06d}.jpg'] for s in scenes for f in range(10)]
        seq=[{'sequence_name':s,'viewpoint_quality_score':.7} for s in scenes]
        return scenes,rows,seq

    def test_test_key_only_and_strict_quality(self):
        scenes,rows,seq=self.fixture()
        seq[0]['viewpoint_quality_score']=.5
        seq[1]['viewpoint_quality_score']=None
        selected=select('apple',0,[{'test':rows,'train':[['training',0,'ignored']]}],seq)
        self.assertEqual(list(selected),['seq2'])
        self.assertEqual(selected['seq2'],list(range(10)))

    def test_seed_and_max_sequences(self):
        scenes,rows,seq=self.fixture(60)
        selected=select('apple',2,[{'test':rows}],seq)
        self.assertEqual(list(selected),random.Random(44).sample(sorted(scenes),50))
        self.assertEqual(select('apple',2,[{'test':rows}],seq),selected)

    def test_no_quality_test_sequences_is_explicit_empty(self):
        self.assertEqual(select('apple',0,[{'test':[]}],[]),{})
        self.assertEqual(select('apple',0,[],[]),{})

    def test_bad_frame_path_duplicate_and_insufficient_rejected(self):
        _,rows,seq=self.fixture(1)
        for variant in [rows+[rows[0]],rows[:9],rows[:-1]+[['seq0',9,'apple/seq0/../frame000009.jpg']]]:
            with self.assertRaises(ValueError):select('apple',0,[{'test':variant}],seq)

    def test_existing_file_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'reference.json'
            write_identical(path,b'original')
            write_identical(path,b'original')
            with self.assertRaises(ValueError):write_identical(path,b'different')
            self.assertEqual(path.read_bytes(),b'original')


if __name__=='__main__':unittest.main()
