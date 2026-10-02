#!/usr/bin/env python3
"""Three fixed requests, metadata-only versus forced-landscape crop.

Archived original results remain unchanged. New branches have separate
forwards, never a same-forward claim. Identical landscape control inputs share
one new forward; GT is excluded from the network/focal/PnP.
"""
import argparse
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from audit_co3d_portrait_geometry import (REQUESTS, fixed_inputs, load_context,
    OUTPUT as GEOMETRY_AUDIT)
from co3d_request_journal import atomic_new
from diagnose_co3d_candidate_pnp import prediction_signature, check_baseline
from fast3r_hf_co3d_100_pose_eval import CHECKPOINT, PREP_ROOT, predict_poses_reference_fallback, tensor_sha
from fast3r_hf_co3d_pose_smoke import model_inputs, correct_orientation, signature
from fast3r_hf_re10k_pose_eval import pose_metrics, sha
from verify_co3d_candidate_pose_report import REPORT

BRANCHES = ('metadata_landscape', 'forced_landscape_crop')
PROGRESS = Path('results/co3d_landscape_input_diagnostic_v1_progress')
OUTPUT = Path('results/co3d_landscape_input_diagnostic_v1_20261003.json')


def input_signature(views, shape_only=False):
    return [{'label': v['label'], 'instance': v['instance'], 'img_tensor_sha256': tensor_sha(v['img']),
        'true_shape_hw': [384, 512] if shape_only else v['true_shape'].tolist(),
        'img_shape': list(v['img'].shape)} for v in views]


def protocol(context):
    baseline, initial, rows, proof, _ = context
    audit = json.loads(GEOMETRY_AUDIT.read_text())
    for p, h in audit['source_sha256'].items():
        if sha(p) != h: raise ValueError('Geometry audit source changed')
    if audit['baseline_report_sha256'] != sha(REPORT): raise ValueError('Geometry audit baseline changed')
    paths = [__file__, 'scripts/audit_co3d_portrait_geometry.py',
        'scripts/verify_co3d_landscape_inputs.py', 'scripts/diagnose_co3d_candidate_pnp.py',
        'scripts/fast3r_hf_co3d_100_pose_eval.py', 'scripts/fast3r_hf_co3d_pose_smoke.py',
        'scripts/fast3r_hf_re10k_pose_eval.py', 'fast3r/dust3r/heads/dpt_head.py',
        'fast3r/dust3r/patch_embed.py', 'fast3r/models/fast3r.py',
        'fast3r/models/multiview_dust3r_module.py', 'fast3r/eval/cam_pose_metric.py',
        'fast3r/dust3r/cloud_opt/init_im_poses.py']
    return {'scope': 'selected_fixed_frame_input_geometry_diagnostic_not_Table1_not_unbiased_sample',
        'request_indices': list(REQUESTS), 'new_branches': list(BRANCHES),
        'selection': 'Same performance-selected requests 0/2/3 as archived PnP diagnostic; no overall mean',
        'baseline_report_sha256': sha(REPORT), 'baseline_protocol_sha256': baseline['protocol_sha256'],
        'geometry_audit_sha256': sha(GEOMETRY_AUDIT),
        'preparation_request_sha256': {str(i): proof['request_sha256'][f'request_{i:03d}.json'] for i in REQUESTS},
        'source_sha256': {str(p): sha(p) for p in paths},
        'checkpoint_weight_sha256': baseline['protocol']['checkpoint_weight_sha256'],
        'checkpoint_config_sha256': baseline['protocol']['checkpoint_config_sha256'],
        'metadata_landscape': 'Original RGB tensor bytes unchanged; override only true_shape to actual loader tensor [384,512]; this does NOT repair transposed-pixel camera geometry',
        'forced_landscape_crop': 'Exact archived selected frame order/GT, pinned crop with only portrait/square resolution reversal removed; no image rotation; RGB pixels change for portrait frames',
        'sampler': 'Native inputs replay saved before-state and match trace/after-state. Alternative crop uses fixed selected frames with independent seed777+mapped index; altered crop RNG is recorded, not claimed to match sampler after-state.',
        'network_input_keys': ['img', 'true_shape'], 'GT_used_by_network_focal_or_PnP': False,
        'network_forward_seed': 'Reset Python/NumPy/Torch/CUDA to 42+request before each distinct input forward',
        'precision': '16-mixed', 'head_chunk_size': 2,
        'focal_PnP': 'Unchanged released first-global focal p10, conf>1,100 iterations, per-view OpenCV seed42+request+view; failures identity retained',
        'paired_same_forward_across_different_inputs': False,
        'identical_new_inputs_reuse_one_forward': True, 'archived_original_model_forward_rerun': False,
        'duplicate_pairs_and_identity_fallbacks_retained': True,
        'formal_Table1_result': False, 'full_paper_completed': False}


def expected_inputs(native, forced):
    return {'metadata_landscape': input_signature(native, True),
            'forced_landscape_crop': input_signature(forced)}


def validate_row(row, proto, archived, expected):
    if (row['request'] not in REQUESTS or row['protocol_sha256'] != signature(proto)
            or row['paired_same_forward_across_different_inputs'] is not False
            or row['formal_Table1_result'] is not False or row['scene'] != archived['scene']
            or row['gt_c2w'] != archived['gt_c2w'] or row['duplicate_camera_pairs'] != archived['duplicate_camera_pairs']
            or row['original_input_tensor_sha256'] != archived['input_tensor_sha256']
            or set(row['branches']) != set(BRANCHES)):
        raise ValueError('Input diagnostic scope/fixed GT/frames mismatch')
    gt = np.asarray(row['gt_c2w'], np.float32); distinct = set(); forwards = 0
    for name in BRANCHES:
        branch = row['branches'][name]
        if branch['input_signature'] != expected[name] or branch['pair_count'] != 45:
            raise ValueError('Changed input or pair count')
        key = signature(branch['input_signature']); was_seen = key in distinct
        if branch['new_forward_count'] != (0 if was_seen else 1): raise ValueError('Forward accounting mismatch')
        distinct.add(key); forwards += branch['new_forward_count']
        pred = np.asarray(branch['predicted_c2w'], np.float32); failures = branch['pnp_failed_view_indices']
        if (pred.shape != (10, 4, 4) or not np.isfinite(pred).all() or len(failures) != len(set(failures))
                or any(type(j) is not int or not 0 <= j < 10 for j in failures)
                or any(not np.array_equal(pred[j], np.eye(4, dtype=np.float32)) for j in failures)
                or not np.isfinite(branch['estimated_focal']) or branch['estimated_focal'] < 0):
            raise ValueError('Invalid predicted matrices/focal/fallbacks')
        metrics, errors = pose_metrics(pred, gt)
        if set(metrics) != set(branch['metrics']) or any(abs(metrics[k]-branch['metrics'][k]) > 1e-7 for k in metrics):
            raise ValueError('Pose metric mismatch')
        for k in errors:
            if np.asarray(branch['relative_errors'][k]).shape != (45,) or not np.allclose(errors[k], branch['relative_errors'][k], atol=1e-7, rtol=0):
                raise ValueError('Pairwise errors mismatch')
        if was_seen:
            first = row['branches'][BRANCHES[0]]
            if any(branch[k] != first[k] for k in ('predicted_c2w', 'prediction_signature', 'metrics', 'estimated_focal', 'pnp_failed_view_indices')):
                raise ValueError('Identical control did not share prediction')
        if i_is_control(row['request']):
            check_baseline(pred, branch['estimated_focal'], failures, archived)
    if row['model_forward_count'] != forwards: raise ValueError('Request forward accounting mismatch')


def i_is_control(i):
    return i == 2  # landscape cup, asserted exact original RGB/true_shape before inference


def build_report(rows, proto):
    if set(rows) != set(REQUESTS): raise ValueError('Incomplete input diagnostic')
    return {'status': 'three_fixed_frame_landscape_input_diagnostics_complete',
        'protocol': proto, 'protocol_sha256': signature(proto), 'requests': [rows[i] for i in REQUESTS],
        'model_forward_count': sum(r['model_forward_count'] for r in rows.values()), 'network_bytes': 0,
        'formal_Table1_result': False, 'full_paper_completed': False,
        'no_diagnostic_average_reported_because_performance_selected': True}


def run(args):
    torch.set_num_threads(2); context = load_context(); baseline = context[0]; proto = protocol(context)
    if not {p.name for p in PROGRESS.glob('request_*.json')}.issubset({f'request_{i:03d}.json' for i in REQUESTS}):
        raise ValueError('Unexpected diagnostic checkpoint')
    saved = {}; preflight = {}
    for i in REQUESTS:
        native, forced = fixed_inputs(i, context); expected = expected_inputs(native, forced)
        if i_is_control(i) and (expected[BRANCHES[0]] != expected[BRANCHES[1]]
                or [v['true_shape'].tolist() for v in native] != [[384,512]]*10):
            raise ValueError('Landscape control input not exact')
        preflight[i] = {'inputs': expected, 'changed_RGB_count': sum(tensor_sha(a['img']) != tensor_sha(b['img']) for a,b in zip(native,forced))}
        path = PROGRESS/f'request_{i:03d}.json'
        if path.exists():
            saved[i] = json.loads(path.read_text()); validate_row(saved[i], proto, baseline['requests'][i], expected)
    if OUTPUT.exists():
        if json.loads(OUTPUT.read_text()) != build_report(saved, proto): raise ValueError('Final result/checkpoint mismatch')
        print('EXISTING INPUT DIAGNOSTIC VERIFIED; no GPU/network/forward', flush=True); return
    if args.verify_only or args.dry_run:
        print(json.dumps({'saved_requests': len(saved), 'preflight': preflight,
            'model_forward_count': 0, 'network_bytes': 0, 'formal_Table1_result': False}), flush=True); return
    free = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'], text=True).splitlines()[0])
    if free < 10240 or not torch.cuda.is_available(): raise RuntimeError('Queue for >=10240MiB; other jobs untouched')
    if shutil.disk_usage(Path('results')).free < 1024**3+4*1024**2: raise RuntimeError('Preserve 1GiB disk reserve')
    from fast3r.models.fast3r import Fast3R
    from fast3r.dust3r.inference_multiview import inference
    model = Fast3R.from_pretrained(str(CHECKPOINT)).cuda().eval(); model.set_max_parallel_views_for_head(2)
    for i in REQUESTS:
        if i in saved: continue
        native, forced = fixed_inputs(i, context); expected = expected_inputs(native, forced)
        gt = np.stack([v['camera_pose'] for v in native]).astype(np.float32); branches = {}; cache = {}
        for name, views in [('metadata_landscape', native), ('forced_landscape_crop', forced)]:
            key = signature(expected[name]); new_forward = key not in cache
            if new_forward:
                network_views = model_inputs(views)
                if name == 'metadata_landscape':
                    for v in network_views: v['true_shape'] = torch.tensor([[384,512]], dtype=torch.int32)
                seed = 42+i; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                torch.cuda.synchronize(); start = time.perf_counter()
                with torch.inference_mode():
                    out = inference(network_views, model, torch.device('cuda'), dtype='16-mixed', verbose=False)
                    torch.cuda.synchronize(); seconds = time.perf_counter()-start
                    preds = correct_orientation(out['preds'], network_views)
                    pred, focal, failures = predict_poses_reference_fallback(preds, seed, niter=100)
                    metrics, errors = pose_metrics(pred, gt)
                    data = {'predicted_c2w': pred.tolist(), 'estimated_focal': focal,
                        'pnp_failed_view_indices': failures, 'metrics': metrics, 'relative_errors': errors,
                        'prediction_signature': prediction_signature(preds), 'forward_seconds': seconds}
                cache[key] = data; del out, preds, network_views
            branches[name] = {**cache[key], 'input_signature': expected[name], 'pair_count':45,
                'new_forward_count': int(new_forward), 'same_forward_as_identical_other_new_input': not new_forward}
            print(json.dumps({'request': i, 'branch': name, 'mAA_30': branches[name]['metrics']['mAA_30'],
                'focal': branches[name]['estimated_focal'], 'failed_views': branches[name]['pnp_failed_view_indices']}), flush=True)
        row = {'request':i, 'scene':native[0]['label'], 'protocol_sha256':signature(proto),
            'gt_c2w':gt.tolist(), 'duplicate_camera_pairs':baseline['requests'][i]['duplicate_camera_pairs'],
            'original_input_tensor_sha256':[tensor_sha(v['img']) for v in native], 'branches':branches,
            'forced_crop_intrinsics':[v['camera_intrinsics'].tolist() for v in forced],
            'forced_crop_rng_markers':[v['rng'] for v in forced],
            'model_forward_count':len(cache), 'paired_same_forward_across_different_inputs':False,
            'formal_Table1_result':False}
        validate_row(row, proto, baseline['requests'][i], expected); atomic_new(PROGRESS/f'request_{i:03d}.json', row); saved[i]=row
    atomic_new(OUTPUT, build_report(saved, proto)); print('FIXED-FRAME INPUT DIAGNOSTIC COMPLETE; not Table1', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--dry-run', action='store_true')
    p.add_argument('--verify-only', action='store_true'); run(p.parse_args())
