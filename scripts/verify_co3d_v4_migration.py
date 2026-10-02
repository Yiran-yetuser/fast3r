"""Verify immutable v4 replay of three frozen v1 requests, without downloads."""
import json
from pathlib import Path
from co3d_request_journal import read_prefix
from prepare_co3d_continuous_v4 import ROOT, validate_result, verify_v1_prefix
from prepare_re10k_rgb_from_archive import sha, save_identical

OUTPUT = Path('results/co3d_v4_prefix3_migration_verified_20261002.json')


def verify():
    initial = json.loads((ROOT/'initial.json').read_text())
    rows, _ = read_prefix(ROOT, initial, validate_result)
    if len(rows) < 3: raise ValueError('V2 replay prefix incomplete')
    for envelope in rows[:3]:
        verify_v1_prefix(envelope['result'], envelope['after_state'], envelope['request'])
    report = {'status': 'v4_first_three_real_requests_equal_frozen_v1',
        'paper_mapping': 'section 4.2 / Table 1 input preparation only',
        'verified_request_count': 3, 'old_journal_preserved': True,
        'input_tensors_GT_rng_trace_and_shared_state_equal': True,
        'modality_bytes_and_npz_arrays_equal': True,
        'v4_initial_sha256': sha(ROOT/'initial.json'),
        'v4_request_sha256': {f'request_{i:03d}.json': sha(ROOT/f'request_{i:03d}.json') for i in range(3)},
        'verification_code_sha256': sha(__file__),
        'runner_sha256': sha('scripts/prepare_co3d_continuous_v4.py'),
        'inf_proof_sha256': sha('results/co3d_finite_signed_reference_v4_verified_20261002.json'),
        'network_bytes_this_verifier': 0, 'model_forward_count': 0,
        'all_100_requests_prepared': False, 'formal_pose_metrics_available': False,
        'author_split_equivalence_verified': False, 'full_paper_completed': False}
    save_identical(OUTPUT, report)
    return report


if __name__ == '__main__': print(json.dumps(verify(), indent=2))
