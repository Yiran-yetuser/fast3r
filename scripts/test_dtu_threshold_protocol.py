"""Offline regression for the fixed historical ten-view diagnostic protocol."""
import ast
import json
import unittest
from pathlib import Path

import numpy as np


class ThresholdProtocolTests(unittest.TestCase):
    def test_uniform_selection_and_saved_identity(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / 'scripts/diagnose_dtu_threshold_sensitivity.py').read_text())
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                try:
                    constants[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
        labels = [f'scan1/{i:08d}.jpg' for i in range(48, -1, -1)]
        indices = np.linspace(0, len(labels) - 1, 10).round().astype(int).tolist()
        self.assertEqual(constants['KF_EVERY'], 1)
        self.assertEqual([labels[i] for i in indices], constants['EXPECTED_LABELS'])
        self.assertNotEqual([labels[i] for i in range(3, 49, 5)], constants['EXPECTED_LABELS'])
        p = json.loads((root / 'results/diagnostics/dtu_scan1_threshold_sensitivity_seed42_v1.json').read_text())
        self.assertEqual(p['selected_indices'], indices)
        self.assertEqual(p['input_labels'], constants['EXPECTED_LABELS'])
        self.assertEqual(p['scene_seed'], 1052)
        self.assertEqual(p['forward_count'], 1)
        self.assertEqual(p['raw_prediction_sha256_before_evaluation'], p['raw_prediction_sha256_after_evaluation'])


if __name__ == '__main__':
    unittest.main()
