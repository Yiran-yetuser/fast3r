import unittest
import copy
import numpy as np
import torch
from PIL import Image
from audit_co3d_portrait_geometry import (force_landscape_method, token_layout_probe,
    synthetic_geometry_probe, FixedLandscapeViews)
from fast3r.dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from diagnose_co3d_landscape_inputs import validate_row, build_report, BRANCHES
from fast3r_hf_co3d_pose_smoke import signature
from fast3r_hf_re10k_pose_eval import pose_metrics


class PortraitTests(unittest.TestCase):
    def fixture(self, H, W):
        image = Image.fromarray(np.random.default_rng(42).integers(0, 255, (H, W, 3), dtype=np.uint8))
        depth = np.ones((H, W), np.float32)
        K = np.array([[500, 0, W/2], [0, 500, H/2], [0, 0, 1]], np.float32)
        return image, depth, K

    def test_portrait_forced_without_rotation(self):
        d = BaseStereoViewDataset(resolution=(512, 384))
        image, depth, K = self.fixture(800, 600)
        old = d._crop_resize_if_necessary(image, depth, K, (512, 384), rng=np.random.default_rng(1))
        new = force_landscape_method()(d, image, depth, K, (512, 384), rng=np.random.default_rng(1))
        self.assertEqual(old[0].size, (384, 512)); self.assertEqual(new[0].size, (512, 384))
        self.assertGreater(np.linalg.det(new[2]), 0)
        self.assertTrue(np.array_equal(K, self.fixture(800, 600)[2]))

    def test_landscape_is_identical_control(self):
        d = BaseStereoViewDataset(resolution=(512, 384)); image, depth, K = self.fixture(600, 800)
        old = d._crop_resize_if_necessary(image, depth, K, (512, 384), rng=np.random.default_rng(1))
        new = force_landscape_method()(d, image, depth, K, (512, 384), rng=np.random.default_rng(1))
        for a, b in zip(old, new): self.assertTrue(np.array_equal(np.asarray(a), np.asarray(b)))

    def test_square_target_fixed_and_no_orientation_draw(self):
        d = BaseStereoViewDataset(resolution=(512, 384)); image, depth, K = self.fixture(700, 700)
        rng = np.random.default_rng(1); before = str(rng.bit_generator.state)
        out = force_landscape_method()(d, image, depth, K, (512, 384), rng=rng)
        self.assertEqual(out[0].size, (512, 384)); self.assertEqual(before, str(rng.bit_generator.state))

    def test_actual_patch_class_ignores_shape(self):
        r = token_layout_probe()
        self.assertTrue(r['encoder_tokens_and_positions_ignore_true_shape'])
        self.assertTrue(r['head_reshape_is_not_spatial_transpose'])
        self.assertGreater(r['token_label_mismatch_fraction'], .9)

    def test_pixel_swap_geometry_perfect_points(self):
        r = synthetic_geometry_probe()
        self.assertLess(r['swapped_K_exact_max_pixel_error'], 1e-10)
        self.assertLess(r['PnP'][0]['rotation_error_deg'], .01)
        self.assertLess(r['PnP'][0]['max_reprojection_error_px'], .001)
        self.assertGreater(r['PnP'][1]['rotation_error_deg'], 150)
        self.assertGreater(r['standard_K_true_pose_median_pixel_error'], 5)

    def result_fixture(self):
        matrices = np.repeat(np.eye(4, dtype=np.float32)[None], 10, 0)
        metrics, errors = pose_metrics(matrices, matrices)
        archived = {'scene': 'fixture/scene', 'gt_c2w': matrices.tolist(),
            'duplicate_camera_pairs': [], 'input_tensor_sha256': ['h']*10}
        proto = {'scope': 'test'}; expected = {name: [] for name in BRANCHES}
        row = {'request': 0, 'protocol_sha256': signature(proto), 'scene': archived['scene'],
            'gt_c2w': archived['gt_c2w'], 'duplicate_camera_pairs': [],
            'original_input_tensor_sha256': ['h']*10, 'model_forward_count': 1,
            'paired_same_forward_across_different_inputs': False, 'formal_Table1_result': False, 'branches': {}}
        for j, name in enumerate(BRANCHES):
            row['branches'][name] = {'input_signature': [], 'pair_count': 45,
                'predicted_c2w': matrices.tolist(), 'pnp_failed_view_indices': [],
                'estimated_focal': 100., 'metrics': metrics, 'relative_errors': errors,
                'prediction_signature': [], 'new_forward_count': int(j == 0)}
        return row, proto, archived, expected

    def test_identical_input_reuses_one_forward(self):
        validate_row(*self.result_fixture())

    def test_false_same_forward_claim_rejected(self):
        row, proto, archived, expected = self.result_fixture()
        row['paired_same_forward_across_different_inputs'] = True
        with self.assertRaises(ValueError): validate_row(row, proto, archived, expected)

    def test_bad_metric_or_forward_count_rejected(self):
        for field in ('metrics', 'new_forward_count'):
            row, proto, archived, expected = self.result_fixture()
            b = row['branches'][BRANCHES[0]]
            if field == 'metrics': b['metrics'] = {**b['metrics'], 'mAA_30': 1.}
            else: b['new_forward_count'] = 0
            with self.assertRaises(ValueError): validate_row(row, proto, archived, expected)

    def test_failed_pose_must_be_identity(self):
        row, proto, archived, expected = self.result_fixture()
        b = row['branches'][BRANCHES[0]]; b['pnp_failed_view_indices'] = [0]
        b['predicted_c2w'] = copy.deepcopy(b['predicted_c2w']); b['predicted_c2w'][0][0][3] = 1.
        with self.assertRaises(ValueError): validate_row(row, proto, archived, expected)

    def test_incomplete_report_rejected(self):
        with self.assertRaises(ValueError): build_report({0: {}}, {})


if __name__ == '__main__': unittest.main()
