"""Request-boundary resume with synthetic invalid depth; no RGB/model scores."""
import copy
import random
import unittest
import numpy as np
from audit_co3d_sampling_trace import AllValidTrace, make_trace
from co3d_sampling_state import capture, restore


class InvalidTrace(AllValidTrace):
    def _load_view_data(self,obj,instance,pool,idx,res,rng):
        if instance=='bad' or idx%7==0:
            self.invalidate[obj,instance][res][idx]=True
            return None
        return super()._load_view_data(obj,instance,pool,idx,res,rng)


def fixture():
    d=make_trace({'toy':{'bad':list(range(100)),'ok':list(range(100)),'other':list(range(100))}})
    d.__class__=InvalidTrace
    return d


def draw(d,idx):
    d._rng=np.random.default_rng(777+idx)
    return d._get_views(idx,(512,384),d._rng)


class SamplingStateTests(unittest.TestCase):
    def test_continuous_100_equal_after_boundary_resume(self):
        mapping=list(range(100));bindings={'synthetic':True,'workers':0}
        d=fixture();random.seed(314)
        first=[draw(d,i) for i in range(37)]
        state=capture(d,mapping,37,bindings)
        expected=[draw(d,i) for i in range(37,100)]
        end=capture(d,mapping,100,bindings)
        resumed=fixture(); random.seed(999)
        self.assertEqual(restore(resumed,mapping,state,bindings),37)
        self.assertEqual(expected,[draw(resumed,i) for i in range(37,100)])
        self.assertEqual(end,capture(resumed,mapping,100,bindings))
        self.assertIn(('toy','bad'),resumed.invalid_scene_tracker)
        self.assertTrue(any(any(f) for r in resumed.invalidate.values() for f in r.values()))
        self.assertEqual(len(first)+len(expected),100)

    def test_changed_mapping_or_protocol_rejected(self):
        d=fixture();s=capture(d,[0,1],0,{'seed':42})
        for mapping,b in (([1,0],{'seed':42}),([0,1],{'seed':43})):
            with self.assertRaises(ValueError): restore(d,mapping,s,b)

    def test_malformed_mask_rejected_without_mutating_live_state(self):
        d=fixture();draw(d,0);s=capture(d,[0,1],1,{})
        bad=copy.deepcopy(s);bad['invalidate'][0]['resolutions'][0]['invalid'][0]=1
        with self.assertRaises(ValueError): restore(d,[0,1],bad,{})
        self.assertEqual(s,capture(d,[0,1],1,{}))


if __name__=='__main__':unittest.main()
