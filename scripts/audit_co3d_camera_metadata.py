#!/usr/bin/env python3
"""Metadata-only CO3D camera preflight, bounded local ZIP/gzip, no RGB download.

Uses pinned DUSt3R camera conventions. Does not establish image/depth decode,
processed crop equivalence, validity-dependent sampling or formal pose results.
"""
import fcntl
import gzip
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from prepare_re10k_rgb_from_archive import sha, save_identical

ROOT = Path('data/co3d_test_metadata')
JOURNAL = Path('results/co3d_camera_metadata_v2_progress')
MAX_PACKED = 128 * 1024**2
MAX_JSON = 512 * 1024**2


def cameras(viewpoints, sizes):
    """Pinned ndc_isotropic, PyTorch3D row-vector -> OpenCV w2c -> c2w."""
    R = np.asarray([v['R'] for v in viewpoints], dtype=np.float64)
    T = np.asarray([v['T'] for v in viewpoints], dtype=np.float64)
    f = np.asarray([v['focal_length'] for v in viewpoints], dtype=np.float64)
    p = np.asarray([v['principal_point'] for v in viewpoints], dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    n = len(viewpoints)
    if (R.shape != (n, 3, 3) or T.shape != (n, 3) or f.shape != (n, 2)
            or p.shape != (n, 2) or sizes.shape != (n, 2)
            or any(v['intrinsics_format'] != 'ndc_isotropic' for v in viewpoints)
            or not all(np.isfinite(a).all() for a in (R, T, f, p, sizes))
            or (f <= 0).any() or (sizes <= 0).any()):
        raise ValueError('Unsupported/nonfinite camera format or shape')
    rotation_error = float(np.max(np.abs(R.transpose(0, 2, 1) @ R - np.eye(3))))
    determinant_error = float(np.max(np.abs(np.linalg.det(R) - 1)))
    if max(rotation_error, determinant_error) > 1e-4:
        raise ValueError('Invalid rotation, no silent orthogonalization')
    wh = sizes[:, ::-1]
    scale = np.min(wh, axis=1) / 2
    K = np.tile(np.eye(3), (n, 1, 1))
    K[:, 0, 0], K[:, 1, 1] = (f * scale[:, None]).T
    K[:, :2, 2] = wh / 2 - p * scale[:, None]
    center = np.rint(K[:, :2, 2]).astype(np.int64)
    margins = np.minimum(center, wh - center)
    if (margins <= 0).any():
        raise ValueError('Principal-point reference crop outside image')
    flip = np.array([-1., -1., 1.])
    w2c = np.tile(np.eye(4), (n, 1, 1))
    w2c[:, :3, :3] = (R * flip[None, None, :]).transpose(0, 2, 1)
    w2c[:, :3, 3] = T * flip
    c2w = np.linalg.inv(w2c)
    inverse_residual = np.abs(w2c @ c2w - np.eye(4))
    inverse_error = float(np.max(inverse_residual))
    # Unit-aware backward-error guard. Keep absolute residual and coordinate
    # magnitude visible: large CO3D translations are NOT normalized or dropped.
    inverse_scaled_error = float(np.max(inverse_residual / (1 + np.abs(w2c) @ np.abs(c2w))))
    if not np.isfinite(c2w).all() or inverse_scaled_error > 1e-12:
        raise ValueError('Invalid camera inverse')
    return K, c2w, {'rotation_orthogonality_max_abs_error': rotation_error,
                    'rotation_det_max_abs_error': determinant_error,
                    'w2c_c2w_max_abs_error': inverse_error,
                    'w2c_c2w_max_component_scaled_error': inverse_scaled_error,
                    'maximum_abs_source_translation': float(np.max(np.abs(T))),
                    'source_camera_count_abs_translation_above_1e6': int(np.sum(np.max(np.abs(T), axis=1) > 1e6))}


def annotation_mapping(category, selected, set_rows):
    """Official set rows carry annotation IDs, distinct from filename numbers."""
    wanted = {f'{category}/{s}/images/frame{f:06d}.jpg'
              for s, frames in selected.items() for f in frames}
    mapping = {}
    for scene, number, path in set_rows:
        if path not in wanted:
            continue
        key = (scene, int(number))
        if path in mapping and mapping[path] != key:
            raise ValueError('Conflicting official filename/annotation mapping')
        mapping[path] = key
    if set(mapping) != wanted or len(set(mapping.values())) != len(mapping):
        raise ValueError('Incomplete/nonunique official annotation mapping')
    return mapping


def audit_category(category, selected, expected_sha, fingerprint):
    archive = ROOT / 'archives' / (category + '_000.zip')
    if archive.stat().st_size > MAX_PACKED or sha(archive) != expected_sha:
        raise ValueError('Metadata archive hash/size changed')
    with zipfile.ZipFile(archive) as z:
        name = category + '/frame_annotations.jgz'
        if z.getinfo(name).file_size > MAX_PACKED:
            raise ValueError('Packed frame annotations exceed budget')
        packed = z.read(name)  # zipfile verifies this member's CRC
    with gzip.GzipFile(fileobj=io.BytesIO(packed)) as g:
        raw = g.read(MAX_JSON + 1)
    if len(raw) > MAX_JSON:
        raise ValueError('Uncompressed annotation JSON exceeds 512MiB guard')
    annotations = json.loads(raw)
    parent = json.loads(Path('results/co3d_test_selection_manifest.json').read_text())
    set_rows = []
    for member in parent['metadata_archives'][category]['set_list_member_order']:
        path = ROOT / 'files' / member
        with zipfile.ZipFile(archive) as z:
            if sha(path) != hashlib.sha256(z.read(member)).hexdigest():
                raise ValueError('Official set list differs from saved file')
        set_rows.extend(json.loads(path.read_text())['test'])
    mapping = annotation_mapping(category, selected, set_rows)
    wanted = set(mapping.values())
    found = {}
    for a in annotations:
        key = (a['sequence_name'], int(a['frame_number']))
        if key not in wanted:
            continue
        if key in found:
            raise ValueError('Duplicate candidate annotation')
        scene = key[0]
        filename = a['image']['path']
        if filename not in mapping or mapping[filename] != key:
            raise ValueError('Annotation identity differs from official set list')
        frame = int(Path(filename).name[5:11])
        stem = f'frame{frame:06d}'
        if (a['image']['path'] != f'{category}/{scene}/images/{stem}.jpg'
                or a['depth']['path'] != f'{category}/{scene}/depths/{stem}.jpg.geometric.png'
                or a['mask']['path'] != f'{category}/{scene}/masks/{stem}.png'
                or a['depth']['scale_adjustment'] != 1.0):
            raise ValueError('Candidate paths/depth scale differ from reference')
        found[key] = a
    if set(found) != wanted:
        raise ValueError('Missing selected frame annotations')
    records = [found[mapping[f'{category}/{s}/images/frame{f:06d}.jpg']]
               for s, frames in selected.items() for f in frames]
    K, c2w, stats = cameras([a['viewpoint'] for a in records], [a['image']['size'] for a in records])
    return {'status': 'selected_camera_annotations_verified_not_image_geometry_ready',
            'category': category, 'fingerprint': fingerprint, 'archive_sha256': expected_sha,
            'annotation_packed_sha256': hashlib.sha256(packed).hexdigest(),
            'annotation_json_bytes': len(raw), 'selected_sequence_count': len(selected),
            'candidate_frame_count': len(records), 'unique_candidate_annotation_count': len(found),
            'filename_annotation_number_mismatch_count': sum(int(Path(a['image']['path']).name[5:11]) != a['frame_number'] for a in records),
            'intrinsics_format': 'ndc_isotropic', 'depth_scale_adjustment': 1.0,
            'focal_pixels_min': float(np.min(K[:, (0, 1), (0, 1)])),
            'focal_pixels_max': float(np.max(K[:, (0, 1), (0, 1)])), **stats,
            'rgb_depth_mask_decode_verified': False, 'processed_crop_equivalence_verified': False,
            'formal_pose_metrics_available': False}


def main():
    JOURNAL.mkdir(parents=True, exist_ok=True)
    lock = (JOURNAL / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = ROOT / 'selected_seqs_test_seen41_candidate.json'
    protocol = json.loads(Path('results/co3d_seen41_protocol_20261002.json').read_text())
    if sha(manifest) != protocol['candidate_manifest_sha256']:
        raise ValueError('Candidate changed')
    selected = json.loads(manifest.read_text())
    checksums = json.loads((ROOT / 'references/co3d_sha256.json').read_text())['full']
    fingerprint = {'candidate_manifest_sha256': sha(manifest), 'checksums_sha256': sha(ROOT / 'references/co3d_sha256.json'),
                   'preprocessor_sha256': sha(ROOT / 'references/preprocess_co3d_reference.py'),
                   'auditor_sha256': sha(Path(__file__))}
    parent = json.loads(Path('results/co3d_test_selection_manifest.json').read_text())
    if any(fingerprint[k] != parent[k] for k in ('checksums_sha256', 'preprocessor_sha256')):
        raise ValueError('Pinned camera reference/checksums changed')
    rows = []
    for c, scenes in selected.items():
        if shutil.disk_usage(JOURNAL).free < 1024**3 + 1024**2:
            raise RuntimeError('Preserve 1GiB reserve')
        p = JOURNAL / (c + '.json')
        # Even on resume re-audit bounded local annotations; no network transfer.
        result = audit_category(c, scenes, checksums[c + '_000.zip'], fingerprint)
        save_identical(p, result)
        rows.append(result)
        print(f'CAMERA METADATA COMPLETE {c}: {result["candidate_frame_count"]}', flush=True)
    report = {'status': 'all_selected_camera_metadata_preflight_complete_not_image_geometry_ready',
              'paper_mapping': 'section 4.2 / Table 1 GT convention prerequisites only',
              'fingerprint': fingerprint, 'category_count': len(rows),
              'selected_sequence_count': sum(r['selected_sequence_count'] for r in rows),
              'candidate_frame_count': sum(r['candidate_frame_count'] for r in rows),
              'filename_annotation_number_mismatch_count': sum(r['filename_annotation_number_mismatch_count'] for r in rows),
              'max_errors': {k: max(r[k] for r in rows) for k in ('rotation_orthogonality_max_abs_error',
                         'rotation_det_max_abs_error', 'w2c_c2w_max_abs_error', 'w2c_c2w_max_component_scaled_error')},
              'maximum_abs_source_translation': max(r['maximum_abs_source_translation'] for r in rows),
              'source_camera_count_abs_translation_above_1e6': sum(r['source_camera_count_abs_translation_above_1e6'] for r in rows),
              'categories': rows, 'rgb_depth_mask_decode_verified': False,
              'processed_crop_equivalence_verified': False, 'float32_pose_metric_stability_verified': False,
              'formal_pose_metrics_available': False,
              'note': 'Large source translations retained; scaled inverse residual is not a float32 metric stability certificate'}
    if (report['category_count'] != 41 or report['selected_sequence_count'] != protocol['selected_sequence_count']
            or report['candidate_frame_count'] != protocol['candidate_frame_count']):
        raise ValueError('Camera aggregate differs from full candidate')
    save_identical(Path('results/co3d_camera_metadata_audit_v2_20261002.json'), report)
    print('ALL 41 CAMERA METADATA PREFLIGHTS COMPLETE', flush=True)


if __name__ == '__main__': main()
