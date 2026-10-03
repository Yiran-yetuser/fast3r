"""Source-trace tests; independent draw arithmetic; no network/images/model."""
import random
import unittest

import numpy as np

from audit_co3d_51_released_sampler import (build_source_plan, potential_frames,
                                           validate_manifest)


class Released51SamplerTests(unittest.TestCase):
    def fixture(self):
        return {'apple': {'long': list(range(100, 300)), 'short': list(range(10))},
                'ball': {'third': list(range(700, 800))}}

    def test_source_mapping_and_draws_against_independent_arithmetic(self):
        selected = self.fixture()
        plan = build_source_plan(selected, requests=8)
        expected = np.random.default_rng(777).permutation(
            3 * plan['generated_combination_count'])[:8].tolist()
        self.assertEqual(plan['mapping'], expected)
        pools = [(c, s, f) for c, ss in selected.items() for s, f in ss.items()]
        for row, index in zip(plan['rows'], expected):
            category, scene, pool = pools[index % 3]
            rng = np.random.default_rng(777 + index)
            positions = [max(0, min(center + int(rng.integers(-4, 5)), len(pool)-1))
                         for center in plan['combination_actually_used']][::-1]
            self.assertEqual((row['category'], row['scene']), (category, scene))
            self.assertEqual(row['pool_indices'], positions)
            self.assertEqual(row['frame_numbers'], [pool[i] for i in positions])

    def test_python_rng_unchanged(self):
        state = random.getstate()
        build_source_plan(self.fixture(), requests=2)
        self.assertEqual(random.getstate(), state)

    def test_repeat_and_epoch_change(self):
        a = build_source_plan(self.fixture(), requests=4)
        self.assertEqual(a, build_source_plan(self.fixture(), requests=4))
        b = build_source_plan(self.fixture(), requests=4, epoch=1)
        self.assertNotEqual(a['mapping'], b['mapping'])
        self.assertEqual(b['resized_seed'], 778)

    def test_source_duplicates_not_removed(self):
        plan = build_source_plan({'ball': {'short': list(range(10))}}, requests=1)
        self.assertEqual(len(plan['rows'][0]['frame_numbers']), 10)
        self.assertLess(plan['rows'][0]['distinct_frame_count'], 10)

    def test_candidate_order_not_sorted_or_edited(self):
        selected = self.fixture()
        selected['apple']['long'] = selected['apple']['long'][::-1]
        original = selected['apple']['long'][:]
        plan = build_source_plan(selected, requests=3)
        self.assertEqual(selected['apple']['long'], original)
        for row in plan['rows']:
            self.assertEqual(row['frame_numbers'],
                [selected[row['category']][row['scene']][i] for i in row['pool_indices']])

    def test_closure_includes_scene_retry_wrap_and_extremal_jitter(self):
        selected = self.fixture()
        closure = potential_frames(selected, [2], [0, 5, 99], retries=5)
        self.assertEqual(set(closure), set(selected))
        self.assertEqual(set(closure['apple']), {'long', 'short'})
        self.assertTrue({0, 9} <= set(closure['apple']['short']))
        self.assertTrue({700, 799} <= set(closure['ball']['third']))

    def test_invalid_manifest_rejected(self):
        for bad in ({}, {'apple': {}}, {'apple': {'s': [1]*10}},
                    {'apple': {'s': list(range(9))}},
                    {'apple': {'s': [True]+list(range(1, 10))}}):
            with self.assertRaises(ValueError):
                validate_manifest(bad)

    def test_invalid_requests_epoch_retry_rejected(self):
        for args in ({'requests': 0}, {'requests': True}, {'epoch': -1}):
            with self.assertRaises(ValueError):
                build_source_plan(self.fixture(), **args)
        with self.assertRaises(ValueError):
            potential_frames(self.fixture(), [0], [0], retries=6)

    def test_honesty_flags(self):
        plan = build_source_plan(self.fixture(), requests=1)
        self.assertTrue(plan['all_valid_stub'])
        for key in ('actual_depth_validity_verified', 'author_rng_recovered', 'formal_Table1_result'):
            self.assertFalse(plan[key])


if __name__ == '__main__':
    unittest.main()
