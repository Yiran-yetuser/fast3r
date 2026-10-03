#!/usr/bin/env python3
"""All-valid 1000@ trace of the RELEASED sampler on the 51-class candidate.

No RGB/depth/mask is read and no model is run. Real depth validity and scene
retry can change inputs. Python combination seed42 and epoch0 are declared
choices, not recovered author RNG. The historical distinct-sequence proposal
is explicitly NOT reused as the evaluation sampling protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from audit_co3d_sampling_trace import make_trace, trace_indices, counts
from fast3r.dust3r.datasets.base.easy_dataset import ResizedDataset
from prepare_re10k_rgb_from_archive import sha, save_identical

MANIFEST = Path('data/co3d_test_metadata/selected_seqs_test_reconstructed.json')
PARENT = Path('results/co3d_test_selection_manifest.json')
PROOF = Path('results/co3d_test_selection_verified_20261002.json')
AUTHOR = Path('results/co3d_author_protocol_update_20261003.json')
LEGACY = Path('results/co3d_51_1000_protocol_budget_20261003.json')
PLAN = Path('data/co3d_test_metadata/released_nominal_51_1000_seed777_v1.json')
OUT = Path('results/co3d_51_released_sampler_20261003.json')
SOURCE_FILES = ('scripts/audit_co3d_sampling_trace.py',
    'fast3r/dust3r/datasets/co3d_multiview.py',
    'fast3r/dust3r/datasets/base/easy_dataset.py',
    'fast3r/dust3r/datasets/base/base_stereo_view_dataset.py')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_manifest(selected):
    if not isinstance(selected, dict) or not selected:
        raise ValueError('Empty/malformed category manifest')
    for category, scenes in selected.items():
        if not isinstance(category, str) or not scenes or not isinstance(scenes, dict):
            raise ValueError('Empty/malformed category')
        for scene, frames in scenes.items():
            if (not isinstance(scene, str) or not isinstance(frames, list)
                    or len(frames) < 10 or len(set(frames)) != len(frames)
                    or any(type(f) is not int or not 0 <= f <= 999999 for f in frames)):
                raise ValueError('Invalid original frame pool; never repair/sort it')


def potential_frames(selected, mapping, combination, retries=5):
    """Finite superset for jitter/oversampling and up to five source scene tries.

    Source jitter is +/-4 before deque.pop. Oversampling uses only valid
    indices already attempted. Each scene retry can draw a different jitter,
    so include every possible clamped jitter for every eligible retry scene.
    This is a conservative reachability bound, NOT the realized GT-valid set.
    """
    if not 1 <= retries <= 5:
        raise ValueError('Unexpected source scene-retry bound')
    scenes = [(c, s, f) for c, ss in selected.items() for s, f in ss.items()]
    result = {}
    for index in mapping:
        for attempt in range(retries):
            category, scene, pool = scenes[(int(index) + attempt) % len(scenes)]
            numbers = result.setdefault(category, {}).setdefault(scene, set())
            for center in combination:
                for jitter in range(-4, 5):
                    numbers.add(pool[max(0, min(center + jitter, len(pool)-1))])
    # Sorting serialization only, never the candidate pool or request draw.
    return {c: {s: sorted(f) for s, f in ss.items()} for c, ss in sorted(result.items())}


def build_source_plan(selected, requests=1000, combination_seed=42, epoch=0):
    validate_manifest(selected)
    if type(requests) is not int or requests <= 0 or type(epoch) is not int or epoch < 0:
        raise ValueError('Invalid request count/epoch')
    python_state = random.getstate()
    try:
        dataset = make_trace(selected, seed=combination_seed)
        wrapper = ResizedDataset(requests, dataset)
        wrapper.set_epoch(epoch)
        mapping = [int(i) for i in wrapper._idxs_mapping]
        rows = trace_indices(dataset, mapping)
    finally:
        random.setstate(python_state)
    for request, row in enumerate(rows):
        row['request'] = request
    closure = potential_frames(selected, mapping, dataset.combinations[0])
    for row in rows:
        if not set(row['frame_numbers']) <= set(closure[row['category']][row['scene']]):
            raise ValueError('Nominal draw escaped the retry/jitter superset')
    return {'scope': 'released_all_valid_nominal_51_1000_candidate_not_actual_or_paper',
        'python_combination_seed': combination_seed, 'dataset_seed': 777,
        'resized_epoch': epoch, 'resized_seed': epoch+777, 'num_views': 10,
        'generated_combination_count': len(dataset.combinations),
        'combination_actually_used': list(dataset.combinations[0]),
        'base_dataset_length': len(dataset), 'mapping': mapping, 'rows': rows,
        'potential_retry_jitter_frames': closure, 'maximum_scene_retries': 5,
        'all_valid_stub': True, 'actual_depth_validity_verified': False,
        'author_rng_recovered': False, 'formal_Table1_result': False}


def audit():
    parent = json.loads(PARENT.read_text())
    proof = json.loads(PROOF.read_text())
    author = json.loads(AUTHOR.read_text())
    if (sha(PARENT) != proof['selection_report_sha256']
            or sha(MANIFEST) != parent['selected_manifest_sha256']
            or proof['nonempty_category_count'] != 51
            or author['author_stated_request_count'] != 1000):
        raise ValueError('Pinned candidate or archived source evidence changed')
    selected = json.loads(MANIFEST.read_text())
    if selected != parent['selected_by_category']:
        raise ValueError('Original candidate pool/order differs')
    plan = build_source_plan(selected)
    plan['candidate_manifest_sha256'] = sha(MANIFEST)
    plan['source_sha256'] = {p: sha(Path(p)) for p in SOURCE_FILES}
    nominal = counts(plan['rows'])
    closure = plan['potential_retry_jitter_frames']
    legacy = json.loads(LEGACY.read_text())
    report = {'status': 'released_51_1000_all_valid_trace_verified_not_real_inputs_or_pose',
        'paper_mapping': 'section 4.2 / Table 1 source sampling and storage prerequisites',
        'audit_script_sha256': sha(Path(__file__)),
        'input_sha256': {str(p): sha(p) for p in (MANIFEST, PARENT, PROOF, AUTHOR, LEGACY)},
        'source_sha256': plan['source_sha256'], 'plan_content_sha256': digest(plan),
        'candidate_category_count': len(selected),
        'candidate_sequence_count': sum(len(s) for s in selected.values()),
        'candidate_frame_count': sum(len(f) for ss in selected.values() for f in ss.values()),
        'request_count': 1000, 'views_per_request': 10,
        'python_combination_seed': 42, 'dataset_seed': 777, 'resized_epoch': 0,
        'generated_combination_count': plan['generated_combination_count'],
        'combination_actually_used': plan['combination_actually_used'],
        'base_dataset_length': plan['base_dataset_length'], 'nominal_counts': nominal,
        'potential_retry_jitter_sequence_count': sum(len(s) for s in closure.values()),
        'potential_retry_jitter_frame_count': sum(len(f) for ss in closure.values() for f in ss.values()),
        'legacy_proposal': {'sha256': sha(LEGACY),
            'distinct_sequence_count': legacy['fixed_plan']['distinct_sequence_count'],
            'unique_frame_slots': legacy['fixed_plan']['unique_frame_slots'],
            'is_released_sampling_protocol': False,
            'is_storage_fit_proof': False, 'will_be_used_for_evaluation': False},
        'directory_storage_budget_complete': False, 'rgb_depth_mask_camera_gt_ready': False,
        'actual_depth_validity_verified': False, 'author_protocol_equivalence_verified': False,
        'checkpoint_paper_mapping_verified': False, 'formal_Table1_result': False,
        'model_forward_count': 0, 'network_bytes_transferred': 0, 'full_paper_completed': False,
        'limitations': [
            'Uses RELEASED _get_views/_fetch_views_for_pool and ResizedDataset, not random.sample replacement',
            'Dataset seed777 is source/author stated; combination seed42 and epoch0 are declared choices',
            'The exact author processed JSON/order, combination RNG and HF training variant are unresolved',
            'All-valid stub does not test real GT, masked positive depth, crop geometry, CRC or decode',
            'Source jitter/clamping can duplicate frames; mapping can reuse trajectories',
            'Zero-depth invalidation/oversampling/scene retries must be realized and checkpointed, not skipped',
            'Reachability includes all +/-4 jitter positions for up to five scene tries, not actual frames',
            'Historical raw2GiB/processed512MiB are ceilings, not measured full1000 storage requirements']}
    return plan, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-json', type=Path, default=PLAN)
    parser.add_argument('--output-json', type=Path, default=OUT)
    args = parser.parse_args()
    plan, report = audit()
    save_identical(args.plan_json, plan)
    report['plan_file'] = str(args.plan_json)
    report['plan_file_sha256'] = sha(args.plan_json)
    save_identical(args.output_json, report)
    print(json.dumps({k: report[k] for k in ('status', 'nominal_counts',
        'potential_retry_jitter_sequence_count', 'potential_retry_jitter_frame_count',
        'legacy_proposal', 'formal_Table1_result')}, indent=2))


if __name__ == '__main__':
    main()
