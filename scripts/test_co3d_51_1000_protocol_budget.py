import copy
import unittest
from pathlib import Path

from audit_co3d_51_1000_protocol_budget import audit, build_plan


class Co3d51BudgetTests(unittest.TestCase):
    def test_real_manifest_has_51_categories_and_expected_delta(self):
        result = audit(Path("."), requests=1000)
        self.assertEqual(result["official_51_manifest"]["category_count"], 51)
        self.assertEqual(result["official_51_manifest"]["sequence_count"], 2511)
        self.assertEqual(result["archived_seen41_candidate"]["sequence_count"], 2011)
        self.assertEqual(result["additional_sequence_count"], 500)
        self.assertEqual(result["additional_frame_count"], 99553)

    def test_plan_is_deterministic_and_covers_all_categories(self):
        import json
        manifest = json.loads(Path("data/co3d_test_metadata/selected_seqs_test_reconstructed.json").read_text())
        left = build_plan(manifest, requests=1000, seed=42)
        right = build_plan(manifest, requests=1000, seed=42)
        self.assertEqual(left, right)
        self.assertEqual(len(left), 1000)
        self.assertEqual({row["category"] for row in left}, set(manifest))
        self.assertTrue(all(row["distinct_frame_count"] == 10 for row in left))

    def test_honesty_flags_are_fail_closed(self):
        result = audit(Path("."), requests=1000)
        self.assertFalse(result["author_protocol_equivalence_verified"])
        self.assertFalse(result["rgb_depth_mask_camera_gt_ready"])
        self.assertFalse(result["formal_Table1_result"])
        self.assertFalse(result["full_paper_completed"])
        self.assertEqual(result["storage_envelope"]["network_bytes_transferred"], 0)
        self.assertEqual(result["storage_envelope"]["model_forward_count"], 0)

    def test_plan_rejects_more_requests_than_pool(self):
        import json
        manifest = json.loads(Path("data/co3d_test_metadata/selected_seqs_test_reconstructed.json").read_text())
        with self.assertRaises(ValueError):
            build_plan(manifest, requests=2512, seed=42)


if __name__ == "__main__":
    unittest.main()
