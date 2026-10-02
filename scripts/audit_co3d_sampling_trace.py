#!/usr/bin/env python3
"""Exercise RELEASED local sampling methods with an explicitly all-valid stub.

No images/GT/model are read. Results are nominal requests only: real depth
validity, duplicate oversampling and scene retries can change actual inputs.
Neither 100 draws nor a 2011-scene adaptation is certified as the paper split.
"""
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from fast3r.dust3r.datasets.co3d_multiview import Co3d_Multiview
from fast3r.dust3r.datasets.base.easy_dataset import ResizedDataset
from prepare_re10k_rgb_from_archive import sha, save_identical


class AllValidTrace(Co3d_Multiview):
    def _load_view_data(self, obj, instance, image_pool, im_idx, resolution, rng):
        return {'category': obj, 'scene': instance, 'pool_index': int(im_idx),
                'frame_number': int(image_pool[im_idx])}


def make_trace(selected, seed=42):
    random.seed(seed)
    dataset = AllValidTrace.__new__(AllValidTrace)
    dataset.num_views = 10
    dataset.scenes = {(c, s): f for c, scenes in selected.items() for s, f in scenes.items()}
    dataset.scene_list = list(dataset.scenes)
    dataset.invalid_scene_tracker = set()
    dataset.invalidate = {s: {} for s in dataset.scene_list}
    dataset._generate_combinations(100, 360, 100)
    return dataset


def trace_indices(dataset, indices):
    rows = []
    for idx in indices:
        idx = int(idx)
        views = dataset._get_views(idx, (512, 384), np.random.default_rng(777 + idx))
        rows.append({'base_index': idx, 'category': views[0]['category'], 'scene': views[0]['scene'],
                     'pool_indices': [v['pool_index'] for v in views],
                     'frame_numbers': [v['frame_number'] for v in views],
                     'distinct_frame_count': len({v['frame_number'] for v in views})})
    return rows


def counts(rows):
    return {'nominal_sample_count': len(rows),
            'nominal_distinct_trajectory_count': len({(r['category'], r['scene']) for r in rows}),
            'nominal_distinct_category_count': len({r['category'] for r in rows}),
            'nominal_distinct_frame_count': len({(r['category'], r['scene'], f) for r in rows for f in r['frame_numbers']}),
            'samples_with_duplicate_views': sum(r['distinct_frame_count'] < 10 for r in rows),
            'min_distinct_frames_in_a_sample': min(r['distinct_frame_count'] for r in rows)}


def main():
    manifest = Path('data/co3d_test_metadata/selected_seqs_test_seen41_candidate.json')
    protocol = json.loads(Path('results/co3d_seen41_protocol_20261002.json').read_text())
    if sha(manifest) != protocol['candidate_manifest_sha256']:
        raise ValueError('Candidate changed')
    selected = json.loads(manifest.read_text())
    dataset = make_trace(selected)
    wrapper = ResizedDataset(100, dataset)
    wrapper.set_epoch(0)
    nominal_100 = trace_indices(dataset, wrapper._idxs_mapping)
    nominal_all = trace_indices(dataset, range(len(dataset.scene_list)))
    full_path = Path('data/co3d_test_metadata/nominal_all_sequences_draws_seed42.json')
    save_identical(full_path, {'scope': 'all_valid_nominal_2011_scene_adaptation_not_paper_draws',
                             'candidate_manifest_sha256': sha(manifest), 'rows': nominal_all})
    report = {'status': 'released_sampling_all_valid_trace_not_real_views_or_pose_metrics',
              'paper_mapping': 'section 4.2 / Table 1 sampling and sparse-cache feasibility audit',
              'candidate_manifest_sha256': sha(manifest), 'audit_script_sha256': sha(Path(__file__)),
              'source_sha256': {p: sha(p) for p in ['fast3r/dust3r/datasets/co3d_multiview.py',
                  'fast3r/dust3r/datasets/base/easy_dataset.py',
                  'fast3r/dust3r/datasets/base/base_stereo_view_dataset.py']},
              'candidate_category_count': len(selected), 'candidate_sequence_count': len(dataset.scene_list),
              'python_combination_seed': 42, 'dataset_seed': 777, 'resized_epoch': 0,
              'num_views': 10, 'generated_combination_count': len(dataset.combinations),
              'combination_actually_used': list(dataset.combinations[0]),
              'base_dataset_length': len(dataset), 'wrapper_length': len(wrapper),
              'nominal_100_wrapper': counts(nominal_100), 'nominal_100_rows': nominal_100,
              'nominal_all_sequence_adaptation': counts(nominal_all),
              'nominal_all_sequence_manifest_sha256': sha(full_path),
              'all_valid_stub': True, 'actual_depth_validity_verified': False,
              'realized_samples_verified': False, 'sparse_download_ready': False,
              'formal_pose_metrics_available': False, 'paper_sampling_equivalence_verified': False,
              'limitations': ['Existing source uses combinations[0], not the mapped combination index',
                  'Jitter plus clamping can produce duplicate views even when all frames are valid',
                  'Real mask/depth validity can oversample valid views or retry another trajectory',
                  '100 @ is a length wrapper; it does not enumerate all 2011 trajectories',
                  'Seed42 combination construction and epoch0 are declared audit choices, not recovered paper draws',
                  'Nominal trace is not sufficient to certify image CRC, GT crops or a sparse-cache bound']}
    save_identical(Path('results/co3d_sampling_trace_20261002.json'), report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('nominal_100_rows', 'source_sha256')}, indent=2))


if __name__ == '__main__':
    main()
