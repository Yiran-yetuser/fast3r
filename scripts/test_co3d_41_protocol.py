"""Synthetic protocol/manifest invariants, not pose metrics."""
import unittest

from audit_co3d_41_categories import CATEGORIES, EXCLUDED, parse_categories, filter_selection


class Seen41Tests(unittest.TestCase):
    def fixture(self):
        seen=[c for c in CATEGORIES if c not in EXCLUDED]
        source=f'TRAINING_CATEGORIES={seen!r}\nTEST_CATEGORIES={EXCLUDED!r}\n'
        selected={c:{'seq0':list(range(10))} for c in CATEGORIES}
        return seen,source,selected

    def test_literals_and_filter_preserve_parent(self):
        seen,source,selected=self.fixture()
        self.assertEqual(parse_categories(source),(seen,EXCLUDED))
        filtered=filter_selection(selected,seen)
        self.assertEqual(list(filtered),seen)
        self.assertEqual(filtered['apple'],selected['apple'])
        self.assertEqual(len(selected),51)

    def test_no_remote_code_execution(self):
        seen,source,_=self.fixture()
        self.assertEqual(parse_categories('raise RuntimeError("never execute")\n'+source)[0],seen)
        with self.assertRaises(ValueError):parse_categories(source.replace(repr(seen),'get_categories()'))

    def test_wrong_count_or_duplicate_constant(self):
        seen,source,_=self.fixture()
        for altered in [source.replace(repr(seen),repr(seen[:-1])),source+'TRAINING_CATEGORIES=[]\n']:
            with self.assertRaises(ValueError):parse_categories(altered)

    def test_overlap_or_other_excluded_list(self):
        _,source,_=self.fixture()
        with self.assertRaises(ValueError):parse_categories(source.replace(repr(EXCLUDED),repr(EXCLUDED[:-1]+['apple'])))

    def test_parent_universe_and_frame_validation(self):
        seen,_,selected=self.fixture()
        for frames in [list(range(9)),list(range(9))+[0],list(range(9))+['9']]:
            selected['apple']={'seq0':frames}
            with self.assertRaises(ValueError):filter_selection(selected,seen)
        selected.pop('ball')
        with self.assertRaises(ValueError):filter_selection(selected,seen)


if __name__=='__main__':unittest.main()
