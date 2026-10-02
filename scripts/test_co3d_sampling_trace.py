"""Nominal source sampler tests, no images/depth or benchmark scores."""
import unittest
from audit_co3d_sampling_trace import make_trace, trace_indices, counts
from fast3r.dust3r.datasets.base.easy_dataset import ResizedDataset


class SamplingTraceTests(unittest.TestCase):
    def selected(self):
        return {'apple': {'first': list(range(100, 300)), 'second': list(range(1000, 1010))}}

    def test_seeded_source_combinations_repeat(self):
        a = make_trace(self.selected()); b = make_trace(self.selected())
        self.assertEqual(a.combinations, b.combinations)
        self.assertEqual(trace_indices(a, [0, 1]), trace_indices(b, [0, 1]))

    def test_requests_stay_in_original_pools(self):
        selected = self.selected(); d = make_trace(selected)
        rows = trace_indices(d, [0, 1, 500])
        for r in rows:
            self.assertEqual(len(r['frame_numbers']), 10)
            pool = selected[r['category']][r['scene']]
            self.assertTrue(set(r['frame_numbers']).issubset(pool))
            self.assertEqual(r['frame_numbers'], [pool[i] for i in r['pool_indices']])

    def test_clamping_can_duplicate(self):
        d = make_trace(self.selected()); d.combinations = [tuple(range(10))]
        row = trace_indices(d, [1])[0]
        self.assertLess(row['distinct_frame_count'], 10)
        self.assertEqual(counts([row])['samples_with_duplicate_views'], 1)

    def test_length_wrapper_epoch_mapping(self):
        d = make_trace(self.selected()); w = ResizedDataset(3, d); w.set_epoch(0)
        first = w._idxs_mapping.copy(); w.set_epoch(0)
        self.assertEqual(len(w), 3)
        self.assertEqual(list(first), list(w._idxs_mapping))
        self.assertTrue(all(0 <= i < len(d) for i in first))


if __name__ == '__main__': unittest.main()
