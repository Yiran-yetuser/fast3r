"""Synthetic camera algebra checks; not actual image/GT registration."""
import copy
import unittest
import numpy as np
from audit_co3d_camera_metadata import cameras, annotation_mapping


class CameraMetadataTests(unittest.TestCase):
    def view(self):
        return {'R': np.eye(3).tolist(), 'T': [1., 2., 3.], 'focal_length': [2., 3.],
                'principal_point': [.1, -.2], 'intrinsics_format': 'ndc_isotropic'}

    def test_ndc_scale_and_center(self):
        K, pose, stats = cameras([self.view()], [[480, 640]])
        np.testing.assert_allclose(K[0], [[480, 0, 296], [0, 720, 288], [0, 0, 1]])
        np.testing.assert_allclose(pose[0][:3, :3], np.diag([-1, -1, 1]))
        np.testing.assert_allclose(pose[0][:3, 3], [-1, -2, -3])
        self.assertLess(stats['w2c_c2w_max_abs_error'], 1e-10)

    def test_rotated_row_vector_projection(self):
        view = self.view(); view['R'] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        K, pose, _ = cameras([view], [[480, 640]])
        X = np.array([.3, .4, 5.])
        pytorch3d = X @ np.asarray(view['R']) + np.asarray(view['T'])
        opencv = np.linalg.inv(pose[0]) @ np.r_[X, 1]
        np.testing.assert_allclose(opencv[:3], pytorch3d * [-1, -1, 1])

    def test_nonfinite_bad_rotation_and_format_rejected(self):
        for key, value in [('T', [float('nan'), 0, 0]), ('R', (np.eye(3) * 2).tolist()),
                           ('intrinsics_format', 'other')]:
            view = copy.deepcopy(self.view()); view[key] = value
            with self.assertRaises(ValueError): cameras([view], [[480, 640]])

    def test_crop_outside_rejected(self):
        view = self.view(); view['principal_point'] = [10, 0]
        with self.assertRaises(ValueError): cameras([view], [[480, 640]])

    def test_large_source_coordinates_retained_and_flagged(self):
        view = self.view(); view['T'] = [1e16, -2e16, 3e16]
        K, pose, stats = cameras([view], [[480, 640]])
        self.assertEqual(stats['maximum_abs_source_translation'], 3e16)
        self.assertEqual(stats['source_camera_count_abs_translation_above_1e6'], 1)
        self.assertLess(stats['w2c_c2w_max_component_scaled_error'], 1e-12)
        np.testing.assert_allclose(pose[0][:3, 3], [-1e16, 2e16, -3e16])

    def test_filename_number_is_not_annotation_id(self):
        path = 'apple/scene/images/frame000019.jpg'
        mapping = annotation_mapping('apple', {'scene': [19]}, [('scene', 18, path)])
        self.assertEqual(mapping[path], ('scene', 18))

    def test_mapping_missing_and_conflicting_rejected(self):
        path = 'apple/scene/images/frame000019.jpg'
        for rows in [[], [('scene', 18, path), ('scene', 19, path)]]:
            with self.assertRaises(ValueError): annotation_mapping('apple', {'scene': [19]}, rows)


if __name__ == '__main__': unittest.main()
