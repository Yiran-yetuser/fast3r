import copy
import tempfile
import unittest
from pathlib import Path
from co3d_request_journal import atomic_new, read_prefix, commit


class JournalTests(unittest.TestCase):
    def state(self,n):return {'identity':{'pool':'x'},'scope':'test','version':1,'next_request':n}
    def test_result_and_state_commit_together_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            commit(d,0,self.state(0),{'value':7},self.state(1))
            rows,state=read_prefix(d,self.state(0),lambda r,i:self.assertEqual(r['value'],7))
            self.assertEqual(state,self.state(1));self.assertEqual(len(rows),1)
            with self.assertRaises(FileExistsError):commit(d,0,self.state(0),{},self.state(1))
            self.assertEqual(len(list(Path(d).glob('request_*.json'))),1)
    def test_gap_and_corrupt_chain_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            commit(d,1,self.state(1),{},self.state(2))
            with self.assertRaises(ValueError):read_prefix(d,self.state(0),lambda *x:None)
        with tempfile.TemporaryDirectory() as d:
            before=self.state(0);before['extra']='changed'
            commit(d,0,before,{},self.state(1))
            with self.assertRaises(ValueError):read_prefix(d,self.state(0),lambda *x:None)
    def test_uncommitted_temporary_file_not_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            atomic_new(Path(d)/'.request-interrupted',{'bad':True})
            rows,state=read_prefix(d,self.state(0),lambda *x:None)
            self.assertEqual(rows,[]);self.assertEqual(state,self.state(0))


if __name__=='__main__':unittest.main()
