#!/usr/bin/env python3
"""Fixed-input 2x2 PnP diagnostic; selected probes, never a benchmark.

One forward per request is shared by all four branches. Archived baseline
poses/metrics must reproduce before a changed branch is attempted. GT never
enters the model, focal estimator or PnP. All fallback poses/pairs are retained.
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

from co3d_request_journal import atomic_new
from co3d_sampling_state import capture, digest, restore
from fast3r_hf_co3d_100_pose_eval import (CHECKPOINT, PREP_ROOT, build_dataset,
    checked_prefix, predict_poses_reference_fallback, tensor_sha)
from fast3r_hf_co3d_pose_smoke import correct_orientation, model_inputs, signature
from fast3r_hf_re10k_pose_eval import pose_metrics, sha
from prepare_co3d_continuous_v4 import inputs
from verify_co3d_candidate_pose_report import REPORT, SUMMARY, require_equal_metrics

REQUESTS = (0, 2, 3)
BRANCHES = ('released', 'top15_only', 'search_only', 'search_top15')
PROGRESS = Path('results/co3d_candidate_pnp_diagnostic_v1_progress')
OUTPUT = Path('results/co3d_candidate_pnp_diagnostic_v1_20261003.json')


def confidence_mask(confidence, top15):
    if confidence.ndim != 2 or not torch.isfinite(confidence).all():
        raise ValueError('Invalid confidence map')
    threshold = float(torch.quantile(confidence.float().flatten(), .85)) if top15 else 1.
    mask = confidence > threshold
    return mask, {'rule': 'conf > quantile(0.85)' if top15 else 'conf > 1',
        'threshold': threshold, 'retained_points': int(mask.sum()),
        'total_points': mask.numel(), 'at_threshold_points': int((confidence == threshold).sum()),
        'mask_sha256': tensor_sha(mask)}


def prediction_signature(preds):
    return [{'points': tensor_sha(p['pts3d_in_other_view']),
        'confidence': tensor_sha(p['conf']), 'shape': list(p['conf'].shape)} for p in preds]


def branch_poses(preds, baseline_focal, seed, search, top15):
    """Hold global points/orientation/seeds/iterations fixed, vary two factors."""
    import cv2
    from fast3r.dust3r.cloud_opt.init_im_poses import fast_pnp
    arrays = []
    mask_stats = []
    for p in preds:
        pts = p['pts3d_in_other_view'][0].float().cpu()
        conf = p['conf'][0].float().cpu()
        if pts.shape != (*conf.shape, 3) or not torch.isfinite(pts).all():
            raise ValueError('Invalid pointmap')
        mask, stats = confidence_mask(conf, top15)
        arrays.append((pts, mask)); mask_stats.append(stats)
    focal = float(baseline_focal)
    search_failed = False
    if search:
        cv2.setRNGSeed(int(seed % (2**31-1)))
        focal, first_pose = fast_pnp(arrays[0][0], None, arrays[0][1], 'cpu',
            niter_PnP=100, num_guessed_focals=100)
        search_failed = focal is None or first_pose is None
        if not search_failed and (not np.isfinite(focal) or focal <= 0):
            raise ValueError('Invalid focal selected by PnP search')
    poses = []; failures = []
    for i, (pts, mask) in enumerate(arrays):
        if search_failed:
            found, pose = None, None
        else:
            cv2.setRNGSeed(int((seed+i) % (2**31-1)))
            found, pose = fast_pnp(pts, focal, mask, 'cpu', niter_PnP=100)
        if found is None or pose is None:
            poses.append(np.eye(4, dtype=np.float32)); failures.append(i)
        else:
            value = pose.cpu().numpy()
            if value.shape != (4,4) or not np.isfinite(value).all():
                raise ValueError('Invalid PnP pose')
            poses.append(value)
    return np.asarray(poses), {'estimated_focal': None if search_failed else float(focal),
        'first_view_focal_search_failed': search_failed,
        'pnp_failed_view_indices': failures, 'confidence_masks': mask_stats,
        'pnp_calls': (1 if search else 0) + (0 if search_failed else 10)}


def check_baseline(poses, focal, failures, archived):
    if (float(focal) != archived['estimated_focal'] or failures != archived['pnp_failed_view_indices']
        or not np.allclose(poses, np.asarray(archived['predicted_c2w']), atol=1e-5, rtol=0)):
        raise ValueError('Fixed-input baseline did not reproduce archived poses/focal/failures')
    metrics, _ = pose_metrics(poses, np.asarray(archived['gt_c2w'], np.float32))
    require_equal_metrics(metrics, archived['metrics'])
    return float(np.max(np.abs(poses-np.asarray(archived['predicted_c2w']))))


def protocol(baseline, proof):
    summary = json.loads(SUMMARY.read_text())
    if sha(REPORT) != summary['complete_report_sha256'] or baseline['formal_Table1_result']:
        raise ValueError('Archived candidate baseline identity or scope changed')
    for name, expected in baseline['protocol']['source_code_sha256'].items():
        if sha(name) != expected: raise ValueError('Bound baseline source changed: '+name)
    for name, key in [('model.safetensors','checkpoint_weight_sha256'),
                      ('config.json','checkpoint_config_sha256')]:
        if sha(CHECKPOINT/name) != baseline['protocol'][key]:
            raise ValueError('Public checkpoint changed')
    code = [__file__, 'scripts/verify_co3d_candidate_pose_report.py',
        'scripts/fast3r_hf_co3d_100_pose_eval.py', 'fast3r/dust3r/cloud_opt/init_im_poses.py',
        'fast3r/models/multiview_dust3r_module.py', 'fast3r/eval/cam_pose_metric.py']
    return {'scope':'selected_fixed_input_diagnostic_not_Table1_not_unbiased_sample',
        'request_indices':list(REQUESTS), 'selection':'request 0 low mAA; request 2 high mAA; request 3 zero focal, selected from archived candidate results',
        'baseline_report_sha256':sha(REPORT), 'baseline_summary_sha256':sha(SUMMARY),
        'baseline_protocol_sha256':baseline['protocol_sha256'],
        'preparation_request_sha256':{str(i):proof['request_sha256'][f'request_{i:03d}.json'] for i in REQUESTS},
        'checkpoint_weight_sha256':baseline['protocol']['checkpoint_weight_sha256'],
        'checkpoint_config_sha256':baseline['protocol']['checkpoint_config_sha256'],
        'source_sha256':{str(p):sha(p) for p in code}, 'branches':list(BRANCHES),
        'factors':{'focal':['published first-global estimate p10','first-view fast_pnp(None), 100 deterministic geomspace guesses S/2..3S, reused for all views'],
                  'mask':['confidence > 1','confidence > torch.quantile(0.85), strict ties excluded']},
        'search_is_paper_random_guess_replication':False,
        'search_failure':'all ten identity fallbacks, no GT focal or alternate-view substitution',
        'pnp_iterations':100, 'reprojection_error_pixels':5, 'solver':'published SOLVEPNP_SQPNP',
        'seed':'42 + request index, per-view OpenCV seed + view index, same in every branch',
        'same_forward':True, 'network_input_keys':['img','true_shape'],
        'GT_used_by_network_focal_or_PnP':False, 'orientation':'published correction once',
        'precision':'16-mixed', 'head_chunk_size':2, 'pair_count_per_request_per_branch':45,
        'duplicate_pairs_and_identity_fallbacks_retained':True,
        'baseline_pose_atol':1e-5, 'baseline_metric_atol':1e-7,
        'formal_Table1_result':False, 'full_paper_completed':False,
        'paper_source':'https://arxiv.org/html/2501.13928v2#S4.SS2'}


def validate_row(row, proto, archived):
    if (row['request'] not in REQUESTS or row['protocol_sha256'] != signature(proto)
        or row['paired_same_forward'] is not True or row['forward_count'] != 1
        or row['prediction_before'] != row['prediction_after']
        or row['baseline_reproduced'] is not True or row['formal_Table1_result'] is not False
        or set(row['branches']) != set(BRANCHES)):
        raise ValueError('Diagnostic provenance/scope mismatch')
    if (row['input_tensor_sha256'] != archived['input_tensor_sha256']
        or row['gt_c2w'] != archived['gt_c2w'] or row['scene'] != archived['scene']
        or row['duplicate_camera_pairs'] != archived['duplicate_camera_pairs']):
        raise ValueError('Fixed input/GT/duplicate accounting mismatch')
    gt = np.asarray(row['gt_c2w'], np.float32)
    for name, branch in row['branches'].items():
        pred = np.asarray(branch['predicted_c2w'], np.float32)
        failures = branch['pnp_failed_view_indices']
        if (pred.shape != (10,4,4) or not np.isfinite(pred).all()
            or len(failures) != len(set(failures)) or any(type(i) is not int or not 0<=i<10 for i in failures)
            or branch['pair_count'] != 45 or len(branch['confidence_masks']) != 10):
            raise ValueError('Invalid diagnostic matrices/failures/masks')
        if any(not np.array_equal(pred[i],np.eye(4,dtype=np.float32)) for i in failures):
            raise ValueError('Identity fallback changed')
        focal = branch['estimated_focal']
        if branch['first_view_focal_search_failed']:
            if name not in ('search_only','search_top15') or focal is not None or failures != list(range(10)):
                raise ValueError('Invalid first-view search failure policy')
        elif focal is None or not np.isfinite(focal) or focal < 0:
            raise ValueError('Invalid diagnostic focal')
        for stats in branch['confidence_masks']:
            if (not np.isfinite(stats['threshold']) or not 0<=stats['retained_points']<=stats['total_points']
                or not 0<=stats['at_threshold_points']<=stats['total_points']):
                raise ValueError('Invalid confidence mask accounting')
        metrics, errors = pose_metrics(pred, gt)
        require_equal_metrics(metrics, branch['metrics'])
        for key in errors:
            if np.asarray(branch['relative_errors'][key]).shape != (45,) or not np.allclose(
                errors[key], branch['relative_errors'][key], atol=1e-7, rtol=0):
                raise ValueError('Diagnostic pair error mismatch')
    a = row['branches']['released']
    delta = check_baseline(np.asarray(a['predicted_c2w']),a['estimated_focal'],a['pnp_failed_view_indices'],archived)
    if abs(delta-row['baseline_pose_max_abs_difference'])>1e-12:
        raise ValueError('Baseline difference accounting mismatch')


def load_fixed_request(i, initial, prep_rows, dataset, wrapper, mapping, bindings):
    before = initial if i == 0 else prep_rows[i-1]['after_state']
    if restore(dataset,mapping,before,bindings) != i:
        raise ValueError('Sampler request cursor mismatch')
    if digest(capture(dataset,mapping,i,bindings)) != prep_rows[i]['before_state_sha256']:
        raise ValueError('Sampler before-state mismatch')
    dataset.trace=[];dataset.pool_attempts=[];dataset.active_request=i
    views=wrapper[i]; expected=prep_rows[i]['result']
    if (inputs(views) != expected['returned_views'] or dataset.trace != expected['load_trace']
        or dataset.pool_attempts != expected['pool_attempts']
        or capture(dataset,mapping,i+1,bindings) != prep_rows[i]['after_state']):
        raise ValueError('Prepared input/trace/after-state changed')
    return views


def run(args):
    torch.set_num_threads(2)
    baseline = json.loads(REPORT.read_text())
    initial, prep_rows, proof = checked_prefix()
    proto = protocol(baseline,proof)
    dataset, wrapper, mapping, bindings = build_dataset(initial,prep_rows)
    saved = {}
    paths = {p.name for p in PROGRESS.glob('request_*.json')}
    if not paths.issubset({f'request_{i:03d}.json' for i in REQUESTS}):
        raise ValueError('Unexpected diagnostic checkpoint')
    for i in REQUESTS:
        path=PROGRESS/f'request_{i:03d}.json'
        if path.exists():
            saved[i]=json.loads(path.read_text());validate_row(saved[i],proto,baseline['requests'][i])
        # Dry-run/recovery still checks the exact archived RGB tensors/GT.
        load_fixed_request(i,initial,prep_rows,dataset,wrapper,mapping,bindings)
    if OUTPUT.exists():
        final=json.loads(OUTPUT.read_text())
        if final != build_report(saved,proto): raise ValueError('Final diagnostic differs from checkpoints')
        print('EXISTING DIAGNOSTIC VERIFIED; no model, network or forward',flush=True);return
    if args.verify_only:
        print(f'VERIFIED {len(saved)}/3 diagnostic requests; no model/network/forward',flush=True);return
    if args.dry_run:
        print(json.dumps({'status':'fixed_inputs_and_baseline_bindings_verified','requests':list(REQUESTS),
            'saved':len(saved),'network_bytes':0,'model_forward_count':0,'formal_Table1_result':False}),flush=True);return
    if len(saved)<3:
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        if free<10240 or not torch.cuda.is_available(): raise RuntimeError('Queue until GPU >=10240MiB; no other jobs stopped')
        if shutil.disk_usage(Path('results')).free < 1024**3+1024**2: raise RuntimeError('Preserve 1GiB disk reserve')
        from fast3r.models.fast3r import Fast3R
        from fast3r.dust3r.inference_multiview import inference
        model=Fast3R.from_pretrained(str(CHECKPOINT)).cuda().eval()
        model.set_max_parallel_views_for_head(2)
    PROGRESS.mkdir(parents=True,exist_ok=True)
    for i in REQUESTS:
        if i in saved: continue
        views=load_fixed_request(i,initial,prep_rows,dataset,wrapper,mapping,bindings)
        gt=np.stack([v['camera_pose'] for v in views]).astype(np.float32)
        network_views=model_inputs(views);seed=42+i
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize();started=time.perf_counter()
        with torch.inference_mode():
            output=inference(network_views,model,torch.device('cuda'),dtype='16-mixed',verbose=False)
            torch.cuda.synchronize();forward_seconds=time.perf_counter()-started
            preds=correct_orientation(output['preds'],network_views)
            before=prediction_signature(preds)
            poses,focal,failures=predict_poses_reference_fallback(preds,seed,niter=100)
            delta=check_baseline(poses,focal,failures,baseline['requests'][i])
            branches={}
            for name, search, top15 in [('released',False,False),('top15_only',False,True),
                                         ('search_only',True,False),('search_top15',True,True)]:
                print(f'RUN request {i}: {name}',flush=True);start=time.perf_counter()
                predicted,stats=branch_poses(preds,focal,seed,search,top15)
                metrics,errors=pose_metrics(predicted,gt)
                branches[name]={**stats,'predicted_c2w':predicted.tolist(),'metrics':metrics,
                    'relative_errors':errors,'pair_count':45,'pnp_seconds':time.perf_counter()-start}
            after=prediction_signature(preds)
        row={'request':i,'scene':views[0]['label'],'protocol_sha256':signature(proto),
            'input_tensor_sha256':[tensor_sha(v['img']) for v in views],
            'gt_c2w':gt.tolist(),'duplicate_camera_pairs':baseline['requests'][i]['duplicate_camera_pairs'],
            'prediction_before':before,'prediction_after':after,'paired_same_forward':True,
            'forward_count':1,'forward_seconds':forward_seconds,'branches':branches,
            'baseline_reproduced':True,'baseline_pose_max_abs_difference':delta,'formal_Table1_result':False}
        validate_row(row,proto,baseline['requests'][i]);atomic_new(PROGRESS/f'request_{i:03d}.json',row);saved[i]=row
        print(json.dumps({'status':'COMMITTED diagnostic request','request':i,
            'mAA_30':{name:b['metrics']['mAA_30'] for name,b in branches.items()}}),flush=True)
        del output,preds,views,network_views
    atomic_new(OUTPUT,build_report(saved,proto))
    print('THREE SELECTED SAME-FORWARD PNP DIAGNOSTICS COMPLETE; not Table1',flush=True)


def build_report(rows, proto):
    if set(rows)!=set(REQUESTS): raise ValueError('Incomplete diagnostic request set')
    return {'status':'three_selected_same_forward_pnp_diagnostics_complete',
        'protocol':proto,'protocol_sha256':signature(proto),'requests':[rows[i] for i in REQUESTS],
        'request_count':3,'branches_per_request':4,'model_forward_count':3,'network_bytes':0,
        'formal_Table1_result':False,'full_paper_completed':False,
        'no_diagnostic_average_reported_because_probes_selected_using_archived_performance':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--verify-only',action='store_true')
    run(parser.parse_args())
