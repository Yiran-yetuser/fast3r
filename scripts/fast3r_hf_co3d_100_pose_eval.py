#!/usr/bin/env python3
"""Evaluate the verified continuous 100-request CO3D candidate, not paper Table 1.

Replays the exact prepared sampler states, feeds only RGB and true_shape to the
public Fast3R checkpoint, and commits one pose result per request atomically.
This is a candidate-split adaptation; author split equivalence is unverified.
"""
import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from co3d_lazy_dataset_v4 import StrictLazyCo3d
from co3d_request_journal import atomic_new, read_prefix
from co3d_sampling_state import capture, digest, restore
from fast3r.dust3r.datasets.base.easy_dataset import ResizedDataset
from fast3r.eval.cam_pose_metric import calculate_auc
from fast3r_hf_co3d_pose_smoke import (correct_orientation, model_inputs,
    predict_poses, signature)
from fast3r_hf_re10k_pose_eval import METRICS, pose_metrics, sha
from prepare_co3d_continuous_v4 import CANDIDATE, ROOT, inputs, validate_result
from verify_co3d_continuous_v4_prefix import verify as verify_full_prefix

PREP_ROOT = Path('results/co3d_continuous_prepare_v4_20261002')
PREFIX_PROOF = Path('results/co3d_v4_prefix_full_20261003.json')
PROGRESS = Path('results/co3d_pose_100_seed42_progress_v3')
OUTPUT = Path('results/co3d_pose_100_seed42_adaptation_v3.json')
CHECKPOINT = Path('checkpoints/Fast3R_ViT_Large_512')
AUDIT = Path('results/protocol_audit_20261001.json')
METRIC_KEYS = tuple(METRICS)


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().cpu().numpy().tobytes()).hexdigest()


class ReadOnlyPrepared:
    """Serve only CRC/SHA-verified prepared records; never download or write."""
    root = Path('data/co3d_lazy_v4_processed')
    raw_root = Path('data/co3d_range_cache/raw')

    def __init__(self, envelopes):
        self.frames = {}
        for envelope in envelopes:
            for row in envelope['result']['prepared_frames']:
                previous = self.frames.setdefault(row['image_path'], row)
                if previous != row:
                    raise ValueError('Conflicting prepared frame records')
        self.verified = set()

    def ensure(self, category, scene, frame):
        key = f'{category}/{scene}/images/frame{frame:06d}.jpg'
        row = self.frames[key]
        if key not in self.verified:
            for name, expected in row['processed_sha256'].items():
                if sha(self.root/name) != expected:
                    raise ValueError('Prepared modality SHA changed: '+name)
            for member in row['raw_members']:
                path = self.raw_root/member['path']
                if path.stat().st_size != member['bytes'] or sha(path) != member['sha256']:
                    raise ValueError('Raw archive member changed: '+member['path'])
            self.verified.add(key)
        return row


def checked_prefix():
    if not PREFIX_PROOF.is_file():
        raise FileNotFoundError('Independent 100-request prefix proof is required')
    proof = json.loads(PREFIX_PROOF.read_text())
    if (proof.get('verified_request_count') != 100 or proof.get('expected_request_count') != 100
        or proof.get('input_tensors_GT_rng_and_shared_state_exact_equal') is not True
        or proof.get('all_100_requests_prepared') is not True
        or proof.get('network_bytes') != 0 or proof.get('model_forward_count') != 0
        or proof.get('author_split_equivalence_verified') is not False):
        raise ValueError('Full candidate-prefix proof is invalid or mis-scoped')
    if proof.get('verifier_sha256') != sha('scripts/verify_co3d_continuous_v4_prefix.py'):
        raise ValueError('Prefix verifier code changed')
    # Re-run the independent check immediately before inference. It uses saved
    # bytes and performs no network or model forward.
    replay = verify_full_prefix()
    for key in ('verified_request_count', 'initial_sha256', 'final_state_sha256', 'request_sha256'):
        if replay.get(key) != proof.get(key):
            raise ValueError('Candidate prefix changed since independent verification')
    initial = json.loads((PREP_ROOT/'initial.json').read_text())
    envelopes, state = read_prefix(PREP_ROOT, initial, validate_result)
    if len(envelopes) != 100 or state['next_request'] != 100:
        raise ValueError('Preparation journal is not a complete 100-request prefix')
    if {f'request_{i:03d}.json': sha(PREP_ROOT/f'request_{i:03d}.json')
        for i in range(100)} != proof['request_sha256']:
        raise ValueError('Request journal differs from independent proof')
    return initial, envelopes, proof


def protocol(args, prefix_proof):
    checkpoint = args.checkpoint_dir
    audit = json.loads(AUDIT.read_text())['checkpoint']
    weight_sha, config_sha = sha(checkpoint/'model.safetensors'), sha(checkpoint/'config.json')
    if weight_sha != audit['local_weight_sha256'] or config_sha != audit['local_config_sha256']:
        raise ValueError('Public checkpoint differs from audited local files')
    code_paths = [__file__, 'scripts/prepare_co3d_continuous_v4.py',
        'scripts/co3d_lazy_dataset_v4.py', 'scripts/co3d_lazy_dataset_v3.py',
        'scripts/co3d_sampling_state.py', 'scripts/co3d_request_journal.py',
        'scripts/verify_co3d_continuous_v4_prefix.py',
        'scripts/fast3r_hf_co3d_pose_smoke.py',
        'scripts/fast3r_hf_re10k_pose_eval.py',
        'fast3r/models/multiview_dust3r_module.py', 'fast3r/eval/cam_pose_metric.py',
        'fast3r/dust3r/cloud_opt/init_im_poses.py', 'fast3r/models/fast3r.py',
        'fast3r/dust3r/inference_multiview.py']
    return {'dataset':'CO3D','benchmark_scope':'verified_100_candidate_requests_adaptation_not_Table1',
        'author_split_equivalence_verified':False, 'candidate_sha256':sha(CANDIDATE),
        'preparation_root':str(PREP_ROOT), 'full_prefix_proof_sha256':sha(PREFIX_PROOF),
        'full_prefix_final_state_sha256':prefix_proof['final_state_sha256'],
        'preparation_request_count':100, 'request_count':100, 'views_per_request':10,
        'all_45_pairs_retained_including_duplicate_views':True,
        'aggregate':'arithmetic mean over 100 request-level metrics; each request averages 45 pairs',
        'network_input_keys':['img','true_shape'], 'GT_pose_intrinsics_depth_mask_used_by_network_or_focal_PnP':False,
        'focal':'first-view global head, confidence percentile 10; no GT intrinsics',
        'pnp':'released fast_pnp, confidence > 1, 100 iterations; deterministic per-view OpenCV seed; zero estimated focal is passed through so upstream solvePnP failure yields explicit identity fallback',
        'failed_pnp':'released identity fallback retained and counted',
        'orientation':'released landscape correction applied once to model outputs',
        'evaluation_seed_scheme':'seed + request index; sampler RNG restored independently from saved preparation state',
        'seed':args.seed,'head_chunk_size':args.head_chunk_size,'precision':'16-mixed',
        'checkpoint_weight_sha256':weight_sha,'checkpoint_config_sha256':config_sha,
        'checkpoint_paper_mapping_verified':False,
        'source_code_sha256':{p:sha(p) for p in code_paths},
        'torch':torch.__version__,'cuda':torch.version.cuda}


def validate_eval(row, request, proto_sha, source_sha):
    if (row.get('request') != request or row.get('protocol_sha256') != proto_sha
        or row.get('preparation_request_sha256') != source_sha
        or row.get('pair_count') != 45 or row.get('model_inference_completed') is not True):
        raise ValueError('Evaluation journal identity mismatch')
    if set(row.get('metrics', {})) != set(METRIC_KEYS):
        raise ValueError('Metric schema mismatch')
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in row['metrics'].values()):
        raise ValueError('Invalid pose metric')
    predicted = np.asarray(row.get('predicted_c2w'), dtype=np.float32)
    gt = np.asarray(row.get('gt_c2w'), dtype=np.float32)
    if predicted.shape != (10,4,4) or gt.shape != (10,4,4) or not np.isfinite(predicted).all() or not np.isfinite(gt).all():
        raise ValueError('Invalid saved c2w matrices')
    failures = row.get('pnp_failed_view_indices', [])
    if len(failures) != len(set(failures)) or any(type(i) is not int or i < 0 or i >= 10 for i in failures):
        raise ValueError('Invalid PnP failure indices')
    metrics, errors = pose_metrics(predicted, gt)
    if any(abs(metrics[k]-row['metrics'][k]) > 1e-7 for k in METRIC_KEYS):
        raise ValueError('Saved poses and aggregate metrics disagree')
    if any(not np.allclose(errors[k], row['relative_errors'][k], atol=1e-7, rtol=0)
           for k in errors):
        raise ValueError('Saved pairwise pose errors disagree')
    focal=row.get('estimated_focal')
    if not np.isfinite(focal) or focal < 0:
        raise ValueError('Invalid saved estimated focal')


def read_eval_prefix(root, proto_sha, request_hashes):
    paths=sorted(root.glob('request_*.json'))
    rows=[]
    for i,path in enumerate(paths):
        if path.name != f'request_{i:03d}.json' or i >= 100:
            raise ValueError('Evaluation checkpoint is not a contiguous prefix')
        row=json.loads(path.read_text())
        validate_eval(row,i,proto_sha,request_hashes[f'request_{i:03d}.json'])
        rows.append(row)
    return rows


def build_dataset(initial, prep_rows):
    selected = json.loads(CANDIDATE.read_text())
    dataset = StrictLazyCo3d(selected, ReadOnlyPrepared(prep_rows), 42, 777)
    wrapper = ResizedDataset(100, dataset); wrapper.set_epoch(0)
    mapping = [int(x) for x in wrapper._idxs_mapping]
    bindings = initial['identity']['bindings']
    random.seed(bindings['initial_python_seed'])
    if capture(dataset, mapping, 0, bindings) != initial:
        raise ValueError('Initial sampling state changed')
    return dataset, wrapper, mapping, bindings


def predict_poses_reference_fallback(preds, seed, niter=100):
    """Keep the released PnP identity fallback when the released focal estimate is zero."""
    import cv2
    from fast3r.dust3r.cloud_opt.init_im_poses import fast_pnp
    from fast3r.models.multiview_dust3r_module import estimate_focal
    focal=estimate_focal(preds[0]['pts3d_in_other_view'].float(),
        preds[0]['conf'].float(),min_conf_thr_percentile=10)
    if not np.isfinite(focal) or focal < 0:
        raise ValueError('Nonfinite or negative estimated focal; preserve request as failed')
    poses=[];failures=[]
    for i,pred in enumerate(preds):
        points=pred['pts3d_in_other_view'][0].float().cpu()
        confidence=pred['conf'][0].float().cpu()
        if not torch.isfinite(points).all() or not torch.isfinite(confidence).all():
            raise ValueError('Nonfinite pointmap/confidence')
        cv2.setRNGSeed(int((seed+i)%(2**31-1)))
        found,pose=fast_pnp(points,float(focal),confidence>1.0,'cpu',niter_PnP=niter)
        if pose is None or found is None:
            failures.append(i);poses.append(np.eye(4,dtype=np.float32))
        else:
            pose=pose.cpu().numpy()
            if pose.shape!=(4,4) or not np.isfinite(pose).all():
                raise ValueError('Invalid PnP pose')
            poses.append(pose)
    return np.asarray(poses),float(focal),failures


def run(args):
    torch.set_num_threads(2)
    initial, prep_rows, proof = checked_prefix()
    proto = protocol(args, proof); proto_sha = signature(proto)
    bindings = initial['identity']['bindings']
    eval_root = args.progress_dir
    eval_root.mkdir(parents=True, exist_ok=True)
    initial_eval = {'protocol':proto,'protocol_sha256':proto_sha,
        'preparation_initial_sha256':sha(PREP_ROOT/'initial.json'),
        'preparation_request_sha256':proof['request_sha256'],
        'scope':'candidate_adaptation_only_not_paper_Table1'}
    initial_eval_path = eval_root/'initial.json'
    if initial_eval_path.exists():
        if json.loads(initial_eval_path.read_text()) != initial_eval:
            raise ValueError('Existing evaluation identity changed')
    else:
        atomic_new(initial_eval_path, initial_eval)
    rows = read_eval_prefix(eval_root, proto_sha, proof['request_sha256'])
    if len(rows) > 100: raise ValueError('Unexpected evaluation rows')
    if args.verify_only:
        print(f'VERIFIED {len(rows)}/100 saved candidate pose requests; no inference',flush=True);return
    if args.dry_run:
        print(json.dumps({'status':'candidate_100_input_prefix_verified_ready_for_GPU_eval',
            'verified_preparation_requests':len(prep_rows),'existing_pose_requests':len(rows),
            'model_loaded':False,'network_bytes':0,'model_forward_count':0,
            'author_split_equivalence_verified':False,'formal_Table1_claim_allowed':False},indent=2),flush=True);return
    if OUTPUT.exists(): raise FileExistsError('Existing final report; verify, never overwrite')
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    if free < 10240: raise RuntimeError('GPU free memory below 10240 MiB; queue instead of competing')
    if torch.cuda.is_available() is False: raise RuntimeError('Host CUDA unavailable')
    if __import__('shutil').disk_usage(eval_root).free < 1024**3:
        raise RuntimeError('Preserve at least 1 GiB free disk')
    dataset, wrapper, mapping, sample_bindings = build_dataset(initial, prep_rows)
    from fast3r.models.fast3r import Fast3R
    from fast3r.dust3r.inference_multiview import inference
    model=Fast3R.from_pretrained(str(args.checkpoint_dir)).cuda().eval()
    model.set_max_parallel_views_for_head(args.head_chunk_size)
    # Initialization must not consume a request's inference RNG stream.
    result_rows=[e['result'] for e in prep_rows]
    pnp_failures=0
    for i, prep_row in enumerate(result_rows):
        source=proof['request_sha256'][f'request_{i:03d}.json']
        path=eval_root/f'request_{i:03d}.json'
        envelope_before = initial if i == 0 else json.loads((PREP_ROOT/f'request_{i-1:03d}.json').read_text())['after_state']
        expected_before = json.loads((PREP_ROOT/f'request_{i:03d}.json').read_text())['before_state_sha256']
        if digest(envelope_before) != expected_before: raise ValueError('Preparation RNG state link broke')
        if path.exists():
            saved=json.loads(path.read_text());validate_eval(saved,i,proto_sha,source)
            if saved['preparation_request_sha256'] != source: raise ValueError('Resume source request changed')
            pnp_failures += len(saved['pnp_failed_view_indices'])
            print(f'RESUMED saved pose request {i+1}/100',flush=True)
            continue
        state=restore(dataset,mapping,envelope_before,sample_bindings)
        if state != i: raise ValueError('Sampler cursor mismatch')
        if digest(capture(dataset,mapping,i,sample_bindings)) != expected_before:
            raise ValueError('Sampler before-state mismatch')
        dataset.trace=[];dataset.pool_attempts=[]
        dataset.active_request=i
        views=wrapper[i]
        if inputs(views) != prep_row['returned_views']:
            raise ValueError(f'Exact sampled input changed at request {i}')
        if dataset.trace != prep_row['load_trace'] or dataset.pool_attempts != prep_row['pool_attempts']:
            raise ValueError(f'Exact sampling trace changed at request {i}')
        after=capture(dataset,mapping,i+1,sample_bindings)
        if after != json.loads((PREP_ROOT/f'request_{i:03d}.json').read_text())['after_state']:
            raise ValueError(f'Exact shared sampler after-state changed at request {i}')
        gt=np.stack([v['camera_pose'] for v in views]).astype(np.float32)
        network_views=model_inputs(views)  # depth, mask, K and GT pose are excluded
        request_seed=(args.seed+i)%(2**32)
        random.seed(request_seed);np.random.seed(request_seed);torch.manual_seed(request_seed);torch.cuda.manual_seed_all(request_seed)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
        with torch.inference_mode():
            output=inference(network_views,model,torch.device('cuda'),dtype='16-mixed',verbose=False)
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            corrected=correct_orientation(output['preds'],network_views)
            predicted,focal,failures=predict_poses_reference_fallback(corrected,request_seed,niter=100)
            metrics,errors=pose_metrics(predicted,gt)
        pnp_failures += len(failures)
        duplicate_pairs=[[a,b] for a in range(10) for b in range(a+1,10)
            if views[a]['label']==views[b]['label'] and views[a]['instance']==views[b]['instance']]
        row={'request':i,'base_index':prep_row['base_index'],'scene':views[0]['label'],
            'protocol_sha256':proto_sha,'preparation_request_sha256':source,
            'input_tensor_sha256':[tensor_sha(v['img']) for v in views],
            'input_shape':[list(v['img'].shape) for v in network_views],
            'true_shape':[v['true_shape'].tolist() for v in network_views],
            'gt_c2w':gt.tolist(),'predicted_c2w':predicted.tolist(),
            'metrics':metrics,'relative_errors':errors,'pair_count':45,
            'pnp_failed_view_indices':failures,'estimated_focal':float(focal),
            'duplicate_camera_pairs':duplicate_pairs,'request_seed':request_seed,
            'model_inference_completed':True,
            'runtime':{'forward_seconds':seconds,'cuda_max_memory_allocated_bytes':torch.cuda.max_memory_allocated(),
                'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda}}
        validate_eval(row,i,proto_sha,source)
        atomic_new(path,row)
        print(json.dumps({'status':'COMMITTED pose request','completed':i+1,'expected':100,
            'scene':row['scene'],'pnp_failures':len(failures),'mAA_30':metrics['mAA_30'],
            'forward_seconds':seconds}),flush=True)
        del output,corrected,network_views,views
    saved=[]
    for i in range(100):
        row=json.loads((eval_root/f'request_{i:03d}.json').read_text())
        validate_eval(row,i,proto_sha,proof['request_sha256'][f'request_{i:03d}.json']);saved.append(row)
    if len(saved)!=100:raise ValueError('Incomplete pose evaluation')
    pooled_r=[];pooled_t=[]
    for row in saved:
        pooled_r.extend(row['relative_errors']['rotation_deg']);pooled_t.extend(row['relative_errors']['translation_deg'])
    r=torch.tensor(pooled_r,dtype=torch.float32);t=torch.tensor(pooled_t,dtype=torch.float32)
    pooled={f'RRA_at_{k}':float((r<k).float().mean()) for k in (5,15,30)}
    pooled.update({f'RTA_at_{k}':float((t<k).float().mean()) for k in (5,15,30)})
    pooled['mAA_30']=float(calculate_auc(r,t,max_threshold=30))
    macro={k:statistics.mean(row['metrics'][k] for row in saved) for k in METRIC_KEYS}
    final={'status':'candidate_100_request_pose_evaluation_complete_not_Table1',
        'protocol':proto,'protocol_sha256':proto_sha,'request_count':100,
        'total_pair_count':4500,'pnp_failed_view_count':pnp_failures,
        'macro_mean_request_metrics':macro,'pooled_all_pair_metrics':pooled,
        'requests':saved,'author_split_equivalence_verified':False,
        'paper_checkpoint_mapping_verified':False,'formal_Table1_result':False,
        'full_paper_completed':False}
    json.dumps(final,allow_nan=False);atomic_new(args.output_json,final)
    print('CO3D CANDIDATE 100-REQUEST POSE EVALUATION COMPLETE; not Table 1',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir',type=Path,default=CHECKPOINT)
    p.add_argument('--progress-dir',type=Path,default=PROGRESS)
    p.add_argument('--output-json',type=Path,default=OUTPUT)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--head-chunk-size',type=int,default=2)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    if args.head_chunk_size<1:p.error('head chunk must be >=1')
    run(args)
