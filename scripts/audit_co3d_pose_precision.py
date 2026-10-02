#!/usr/bin/env python3
"""CPU-only GT precision diagnostic, not predicted-camera benchmark scores.

All candidate within-trajectory camera pairs are compared. The pinned DUSt3R
float32 w2c inversion is retained, alongside a diagnostic float64 counterpart.
No GT rebasing, scene filtering, model inference or RGB access is performed.
Print an apply_patch result artifact so the caller explicitly saves it.
"""
import gzip
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

from audit_co3d_camera_metadata import ROOT, MAX_PACKED, MAX_JSON, annotation_mapping
from prepare_re10k_rgb_from_archive import sha
from fast3r.eval.cam_pose_metric import closed_form_inverse, translation_angle, camera_to_rel_deg


def poses(viewpoints, dtype):
    R = np.asarray([v['R'] for v in viewpoints], dtype=np.float64)
    T = np.asarray([v['T'] for v in viewpoints], dtype=np.float64)
    if R.shape != (len(T), 3, 3) or T.shape != (len(T), 3) or not np.isfinite(R).all() or not np.isfinite(T).all():
        raise ValueError('Invalid source cameras')
    # Reference constructs float32 w2c BEFORE inversion, not float64 c2w cast.
    w2c = np.tile(np.eye(4, dtype=dtype), (len(T), 1, 1))
    flip = np.array([-1., -1., 1.])
    w2c[:, :3, :3] = (R * flip[None, None, :]).transpose(0, 2, 1)
    w2c[:, :3, 3] = T * flip
    result = np.linalg.inv(w2c)
    if result.dtype != dtype or not np.isfinite(result).all():
        raise ValueError('Invalid inverse or unexpected precision')
    return torch.from_numpy(result)


def pair_diagnostic(viewpoints):
    p32, p64 = poses(viewpoints, np.float32), poses(viewpoints, np.float64)
    n = len(viewpoints)
    if n < 2:
        raise ValueError('At least two candidate views required')
    i, j = torch.combinations(torch.arange(n), 2).unbind(-1)
    r32 = closed_form_inverse(p32[i]).bmm(p32[j])
    r64 = closed_form_inverse(p64[i]).bmm(p64[j])
    t32, t64 = r32[:, :3, 3].double(), r64[:, :3, 3]
    norm32, norm64 = torch.linalg.vector_norm(t32, dim=1), torch.linalg.vector_norm(t64, dim=1)
    defined = (norm32 > 0) & (norm64 > 0)
    angle = translation_angle(t64[defined], t32[defined])
    # Direct double-precision centre differences distinguish physical baseline
    # from a tiny nonzero closed-form product residual. This scale-aware 1e-6
    # diagnostic bin is NOT a scene/pair filter for a formal benchmark.
    baseline = torch.linalg.vector_norm(p64[j, :3, 3] - p64[i, :3, 3], dim=1)
    coordinate_scale = torch.maximum(torch.linalg.vector_norm(p64[i, :3, 3], dim=1),
                                     torch.linalg.vector_norm(p64[j, :3, 3], dim=1))
    resolved = baseline > 1e-6 * torch.maximum(coordinate_scale, torch.ones_like(baseline))
    resolved_angle = translation_angle(t64[defined & resolved], t32[defined & resolved])
    self_r, self_t = camera_to_rel_deg(p32, p32, 'cpu', n)
    if not all(torch.isfinite(x).all() for x in (r32, r64, angle, self_r, self_t)):
        raise ValueError('Nonfinite released metric; do not silently skip pairs')
    return {
        'candidate_frame_count': n, 'pair_count': len(i),
        'float64_zero_translation_pairs': int((norm64 == 0).sum()),
        'float32_zero_translation_pairs': int((norm32 == 0).sum()),
        'nonzero64_collapsed32_pairs': int(((norm64 > 0) & (norm32 == 0)).sum()),
        'zero64_nonzero32_pairs': int(((norm64 == 0) & (norm32 > 0)).sum()),
        'direction_defined_pairs': int(defined.sum()),
        'direction_disagreement_above_1deg_pairs': int((angle > 1).sum()),
        'direction_disagreement_max_deg': float(angle.max()) if len(angle) else 0.,
        'float64_identical_camera_center_pairs': int((baseline == 0).sum()),
        'baseline_above_diagnostic_scale_threshold_pairs': int(resolved.sum()),
        'resolved_baseline_collapsed32_pairs': int((resolved & (norm32 == 0)).sum()),
        'resolved_baseline_direction_above_1deg_pairs': int((resolved_angle > 1).sum()),
        'resolved_baseline_direction_max_deg': float(resolved_angle.max()) if len(resolved_angle) else 0.,
        'relative_translation_max_component_abs_difference': float((t32 - t64).abs().max()),
        'rotation_matrix_max_abs_difference': float((r32[:, :3, :3].double() - r64[:, :3, :3]).abs().max()),
        'released_float32_identical_gt_translation_above_1deg_pairs': int((self_t > 1).sum()),
        'released_float32_identical_gt_translation_max_deg': float(self_t.max()),
        'released_float32_identical_gt_rotation_max_deg': float(self_r.max()),
        'maximum_abs_source_translation': float(np.abs([v['T'] for v in viewpoints]).max()),
    }


SUM_KEYS = ('candidate_frame_count', 'pair_count', 'float64_zero_translation_pairs',
            'float32_zero_translation_pairs', 'nonzero64_collapsed32_pairs', 'zero64_nonzero32_pairs',
            'direction_defined_pairs', 'direction_disagreement_above_1deg_pairs',
            'float64_identical_camera_center_pairs', 'baseline_above_diagnostic_scale_threshold_pairs',
            'resolved_baseline_collapsed32_pairs', 'resolved_baseline_direction_above_1deg_pairs',
            'released_float32_identical_gt_translation_above_1deg_pairs')
MAX_KEYS = ('direction_disagreement_max_deg', 'relative_translation_max_component_abs_difference',
            'resolved_baseline_direction_max_deg',
            'rotation_matrix_max_abs_difference', 'released_float32_identical_gt_translation_max_deg',
            'released_float32_identical_gt_rotation_max_deg', 'maximum_abs_source_translation')


def aggregate(rows):
    return {**{k: sum(r[k] for r in rows) for k in SUM_KEYS},
            **{k: max(r[k] for r in rows) for k in MAX_KEYS}}


def category_records(category, selected, parent, expected_sha):
    archive = ROOT / 'archives' / (category + '_000.zip')
    if archive.stat().st_size > MAX_PACKED or sha(archive) != expected_sha:
        raise ValueError('Metadata archive identity changed')
    with zipfile.ZipFile(archive) as z:
        member = category + '/frame_annotations.jgz'
        if z.getinfo(member).file_size > MAX_PACKED:
            raise ValueError('Packed annotation exceeds guard')
        packed = z.read(member)  # member CRC verified by zipfile
        set_rows = []
        for m in parent['metadata_archives'][category]['set_list_member_order']:
            data = z.read(m)
            if sha(ROOT / 'files' / m) != hashlib.sha256(data).hexdigest():
                raise ValueError('Set-list identity changed')
            set_rows.extend(json.loads(data)['test'])
    with gzip.GzipFile(fileobj=io.BytesIO(packed)) as g:
        raw = g.read(MAX_JSON + 1)
    if len(raw) > MAX_JSON:
        raise ValueError('Annotation JSON exceeds guard')
    mapping = annotation_mapping(category, selected, set_rows)
    wanted = set(mapping.values())
    found = {}
    for a in json.loads(raw):
        key = (a['sequence_name'], int(a['frame_number']))
        if key not in wanted:
            continue
        if key in found or mapping.get(a['image']['path']) != key:
            raise ValueError('Nonunique/inconsistent candidate annotation')
        found[key] = a['viewpoint']
    if set(found) != wanted:
        raise ValueError('Candidate annotation incomplete')
    return {s: [found[mapping[f'{category}/{s}/images/frame{f:06d}.jpg']] for f in frames]
            for s, frames in selected.items()}


def main():
    torch.set_num_threads(2)
    protocol_path = Path('results/co3d_seen41_protocol_20261002.json')
    camera_path = Path('results/co3d_camera_metadata_audit_v2_20261002.json')
    camera = json.loads(camera_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    manifest = ROOT / 'selected_seqs_test_seen41_candidate.json'
    parent = json.loads(Path('results/co3d_test_selection_manifest.json').read_text())
    checksums_path = ROOT / 'references/co3d_sha256.json'
    reference_path = ROOT / 'references/preprocess_co3d_reference.py'
    if (sha(manifest) != protocol['candidate_manifest_sha256']
            or sha(checksums_path) != camera['fingerprint']['checksums_sha256']
            or sha(reference_path) != camera['fingerprint']['preprocessor_sha256']):
        raise ValueError('Pinned candidate/source changed')
    selected = json.loads(manifest.read_text())
    checksums = json.loads(checksums_path.read_text())['full']
    categories, flagged, all_rows = [], [], []
    for c, scenes in selected.items():
        records = category_records(c, scenes, parent, checksums[c + '_000.zip'])
        rows = [dict(category=c, sequence=s, **pair_diagnostic(v)) for s, v in records.items()]
        categories.append(dict(category=c, selected_sequence_count=len(rows), **aggregate(rows)))
        all_rows.extend(rows)
        flagged.extend(r for r in rows if r['maximum_abs_source_translation'] > 1e6
                       or r['direction_disagreement_above_1deg_pairs']
                       or r['nonzero64_collapsed32_pairs'] or r['zero64_nonzero32_pairs']
                       or r['released_float32_identical_gt_translation_above_1deg_pairs'])
        print(f'PRECISION COMPLETE {c}: {len(rows)} trajectories', file=sys.stderr, flush=True)
    totals = aggregate(all_rows)
    if len(categories) != 41 or len(all_rows) != 2011 or totals['candidate_frame_count'] != 399204:
        raise ValueError('Not the full audited candidate set')
    report = {
        'status': 'all_candidate_gt_pair_precision_diagnostic_complete_not_pose_benchmark',
        'paper_mapping': 'section 4.2 / Table 1 relative-pose numerical prerequisites',
        'fingerprint': {'candidate_manifest_sha256': sha(manifest), 'protocol_report_sha256': sha(protocol_path),
                        'camera_report_sha256': sha(camera_path), 'checksums_sha256': sha(checksums_path),
                        'preprocessor_sha256': sha(reference_path), 'auditor_sha256': sha(Path(__file__)),
                        'metric_sha256': sha(Path('fast3r/eval/cam_pose_metric.py')),
                        'so3_sha256': sha(Path('fast3r/utils/so3_utils.py')),
                        'annotation_mapper_sha256': sha(Path('scripts/audit_co3d_camera_metadata.py'))},
        'runtime': {'numpy': np.__version__, 'torch': torch.__version__, 'device': 'cpu', 'threads': 2},
        'category_count': len(categories), 'selected_sequence_count': len(all_rows),
        'aggregate': totals, 'categories': categories,
        'flagged_sequence_count': len(flagged),
        'all_flagged_sequence_diagnostics_sha256': hashlib.sha256(json.dumps(flagged, sort_keys=True).encode()).hexdigest(),
        'flagged_sequences_listing_is_subset': True,
        'flagged_sequences': sorted(flagged, key=lambda r: (r['maximum_abs_source_translation'] > 1e6,
                                  r['direction_disagreement_max_deg']), reverse=True)[:20],
        'comparison': 'float32 w2c inversion + released torch float32 pair metric vs float64 counterpart; no origin/scale changes',
        'diagnostic_tolerance_deg': 1., 'predicted_poses_available': False,
        'resolved_baseline_diagnostic_threshold': 'norm(c64_j-c64_i) > 1e-6 * max(1,norm(c64_i),norm(c64_j)); diagnostic bins only, no formal pair omission',
        'formal_pose_metrics_available': False, 'gpu_precision_verified': False,
        'image_geometry_verified': False, 'paper_sampling_equivalence_verified': False,
        'note': 'All candidate pairs, not the unknown paper draws. Computationally nonzero64 is not necessarily a physical baseline. Direction angles exclude zero vectors but all pairs remain counted; a separate direct-centre scale bin diagnoses near-zero baselines. Identical-GT self-check is not inference accuracy. CPU evidence does not certify CUDA/TF32 or ill-conditioned predictions.'}
    payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
    path = 'results/co3d_pose_precision_20261002.json'
    if Path(path).exists():
        raise FileExistsError('Preserve historical result')
    print('*** Begin Patch\n*** Add File: ' + path)
    print(''.join('+' + line for line in payload.splitlines(True)), end='')
    print('*** End Patch')


if __name__ == '__main__': main()
