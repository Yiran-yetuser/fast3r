#!/usr/bin/env python3
"""Strict full1000 source51 candidate pose adapter; NOT formal paper Table1.

Full actual input preparation and independent offline replay must finish before
any model is loaded. No data downloads, no sparse pool or GT input to inference.
Historical100-request reports/inputs/weights/code are never modified or reused.
"""
import argparse
import hashlib
import json
import random
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from co3d51_source_lazy_v1 import Source51Landscape,MANIFEST
from co3d_request_journal import atomic_new,read_prefix
from co3d_sampling_state import capture,digest,restore
from fast3r.dust3r.datasets.base.easy_dataset import ResizedDataset
from fast3r.eval.cam_pose_metric import calculate_auc
from fast3r_hf_co3d_100_pose_eval import predict_poses_reference_fallback,validate_eval
from fast3r_hf_co3d_pose_smoke import model_inputs,correct_orientation,signature
from fast3r_hf_re10k_pose_eval import METRICS,pose_metrics,sha
from prepare_co3d_continuous_v4 import inputs
from prepare_co3d51_source_v1 import ROOT as PREP_ROOT,SUMMARY as PREP_SUMMARY,REQUESTS,validate_result
from verify_co3d51_source_inputs_v1 import ReadOnlyInputs,verify as verify_input_replay

INPUT_PROOF=Path('results/co3d51_source_inputs_full_verified_20261003.json')
PROGRESS=Path('results/co3d51_pose_seed42_progress_v1')
OUTPUT=Path('results/co3d51_pose_seed42_candidate_v1.json')
CHECKPOINT=Path('checkpoints/Fast3R_ViT_Large_512')
GATE_SNAPSHOT=Path('results/co3d51_pose_input_gate_20261003.json')
RESERVE=1024**3
MAX_POSE_JOURNAL=512*1024**2
MAX_FINAL_BYTES=16*1024**2


class PreparationPending(RuntimeError):
    pass


def assert_full_input_claims(proof,summary):
    if (proof.get('verified_request_count')!=REQUESTS
            or proof.get('expected_full_request_count')!=REQUESTS
            or proof.get('all_1000_requests_prepared_and_replayed') is not True
            or proof.get('raw_SHA_CRC_processed_SHA_and_receipts_checked') is not True
            or proof.get('input_tensor_GT_rng_trace_and_shared_state_exact_equal') is not True
            or proof.get('network_bytes')!=0 or proof.get('model_forward_count')!=0
            or any(proof.get(k) is not False for k in
                ('author_protocol_equivalence_verified','formal_Table1_result','full_paper_completed'))
            or summary.get('request_count')!=REQUESTS or summary.get('model_forward_count')!=0
            or any(summary.get(k) is not False for k in
                ('author_protocol_equivalence_verified','formal_Table1_result','full_paper_completed'))):
        raise ValueError('Input proof incomplete, missing strict flags or promoted as paper score')
    expected={f'request_{i:03d}.json' for i in range(REQUESTS)}
    if (set(proof.get('request_sha256',{}))!=expected
            or proof['request_sha256']!=summary.get('request_sha256')):
        raise ValueError('Full1000 input request set differs from preparation/proof')


def readiness_snapshot():
    names=sorted(p.name for p in PREP_ROOT.glob('request_*.json'))
    if names!=[f'request_{i:03d}.json' for i in range(len(names))] or len(names)>REQUESTS:
        raise ValueError('Actual preparation journal filenames are not a contiguous prefix')
    complete=PREP_SUMMARY.is_file() and INPUT_PROOF.is_file()
    if complete:
        proof=json.loads(INPUT_PROOF.read_text());summary=json.loads(PREP_SUMMARY.read_text())
        assert_full_input_claims(proof,summary)
    return {'status':'full_input_proof_present_needs_fresh_replay' if complete else 'waiting_for_complete1000_input_preparation_and_replay',
        'paper_mapping':'section4.2/Table1 source51 public-HF candidate adapter only',
        'observed_committed_request_file_count':len(names),'expected_request_count':REQUESTS,
        'observed_prefix_count_is_not_independent_input_replay':True,
        'complete_preparation_summary_present':PREP_SUMMARY.is_file(),
        'complete_independent_input_proof_present':INPUT_PROOF.is_file(),
        'model_loaded':False,'model_forward_count':0,'network_bytes':0,
        'formal_Table1_result':False,'full_paper_completed':False,
        'runner_sha256':sha(__file__)}


def checked_inputs():
    if not PREP_SUMMARY.is_file() or not INPUT_PROOF.is_file():
        raise PreparationPending('Full1000 prepared summary and independent input replay proof required before model load')
    proof=json.loads(INPUT_PROOF.read_text());summary=json.loads(PREP_SUMMARY.read_text())
    assert_full_input_claims(proof,summary)
    if proof.get('verifier_sha256')!=sha('scripts/verify_co3d51_source_inputs_v1.py'):
        raise ValueError('Independent input verifier changed')
    replay=verify_input_replay(REQUESTS)  # Fresh offline replay, no model/download/preprocessing writes.
    if replay!=proof:raise ValueError('Inputs changed since independent full replay')
    initial_path=PREP_ROOT/'initial.json';initial=json.loads(initial_path.read_text())
    envelopes,final=read_prefix(PREP_ROOT,initial,validate_result)
    if (len(envelopes)!=REQUESTS or final['next_request']!=REQUESTS
            or sha(initial_path)!=proof['initial_sha256'] or digest(final)!=proof['final_state_sha256']):
        raise ValueError('Prepared initial/final state changed')
    return initial,envelopes,proof


def protocol(args,proof):
    audit=json.loads(Path('results/protocol_audit_20261001.json').read_text())['checkpoint']
    weight_sha=sha(args.checkpoint_dir/'model.safetensors');config_sha=sha(args.checkpoint_dir/'config.json')
    if (weight_sha!=audit['local_weight_sha256'] or config_sha!=audit['local_config_sha256']):
        raise ValueError('Pinned public weights/config changed')
    files=[__file__,'scripts/verify_co3d51_source_inputs_v1.py','scripts/prepare_co3d51_source_v1.py',
        'scripts/co3d51_source_lazy_v1.py','scripts/co3d_lazy_dataset_v4.py','scripts/co3d_lazy_dataset_v3.py',
        'scripts/co3d_sampling_state.py','scripts/co3d_request_journal.py',
        'scripts/fast3r_hf_co3d_100_pose_eval.py','scripts/fast3r_hf_co3d_pose_smoke.py',
        'scripts/fast3r_hf_re10k_pose_eval.py','fast3r/models/fast3r.py',
        'fast3r/models/multiview_dust3r_module.py','fast3r/dust3r/inference_multiview.py',
        'fast3r/dust3r/cloud_opt/init_im_poses.py','fast3r/eval/cam_pose_metric.py',
        'fast3r/utils/so3_utils.py','fast3r/dust3r/datasets/co3d_multiview.py',
        'fast3r/dust3r/datasets/base/base_stereo_view_dataset.py','fast3r/dust3r/datasets/base/easy_dataset.py']
    return {'dataset':'CO3D','scope':'full1000_source51_landscape_publicHF_candidate_not_formal_Table1',
        'candidate_sha256':sha(MANIFEST),'preparation_root':str(PREP_ROOT),
        'complete_preparation_summary_sha256':sha(PREP_SUMMARY),'full_input_proof_sha256':sha(INPUT_PROOF),
        'full_input_final_state_sha256':proof['final_state_sha256'],
        'request_count':REQUESTS,'views_per_request':10,'pairs_per_request':45,
        'crop':'declared fixed landscape512x384 source51 candidate; exact author processed JSON unknown',
        'aggregate':'arithmetic mean of1000 request-level metrics; all45000 pairs retained including duplicates',
        'network_input_keys':['img','true_shape'],
        'GT_pose_intrinsics_depth_mask_used_by_network_or_focal_PnP':False,
        'focal':'first-view global points/confidence percentile10, no GT K',
        'pnp':'released fast_pnp confidence>1,100 iterations, per-view OpenCV seed, zero focal passed through',
        'failed_pnp':'identity fallback retained and counted; nonfinite/IO failures hard-stop, no skipping',
        'orientation':'released correction once; prepared tensor and true_shape both384x512',
        'inference_seed_scheme':'seed+request; sampler RNG independently restored from immutable prep envelope',
        'seed':args.seed,'head_chunk_size':args.head_chunk_size,'precision':'16-mixed',
        'checkpoint_weight_sha256':weight_sha,'checkpoint_config_sha256':config_sha,
        'source_code_sha256':{p:sha(p) for p in files},'torch':torch.__version__,'cuda':torch.version.cuda,
        'author_protocol_equivalence_verified':False,'paper_checkpoint_mapping_verified':False,
        'formal_Table1_result':False,'full_paper_completed':False}


def validate_pose_row(row,i,proto,proof,prep):
    validate_eval(row,i,signature(proto),proof['request_sha256'][f'request_{i:03d}.json'])
    views=prep['returned_views'];gt=np.asarray(row['gt_c2w'],np.float32)
    if (row.get('formal_Table1_result') is not False or row['request_seed']!=(proto['seed']+i)%(2**32)
            or row['base_index']!=prep['base_index'] or row['scene']!=views[0]['label']
            or row['input_tensor_sha256']!=[v['img_tensor_sha256'] for v in views]
            or row['input_shape']!=[[1]+v['img_shape'] for v in views]
            or row['true_shape']!=[[v['true_shape_hw']] for v in views]
            or not np.array_equal(gt,np.asarray([v['camera_pose'] for v in views],np.float32))):
        raise ValueError('Saved pose row differs from actual input/GT/seed or paper scope')
    duplicates=[[a,b] for a in range(10) for b in range(a+1,10)
        if (views[a]['label'],views[a]['instance'])==(views[b]['label'],views[b]['instance'])]
    if row['duplicate_camera_pairs']!=duplicates:raise ValueError('Duplicate pair count changed')
    predicted=np.asarray(row['predicted_c2w'],np.float32)
    for j in row['pnp_failed_view_indices']:
        if not np.array_equal(predicted[j],np.eye(4,dtype=np.float32)):
            raise ValueError('Failed PnP identity fallback dropped/changed')


def read_pose_prefix(root,proto,proof,prep_rows):
    rows=[]
    for i,path in enumerate(sorted(root.glob('request_*.json'))):
        if i>=REQUESTS or path.name!=f'request_{i:03d}.json':raise ValueError('Noncontiguous pose prefix')
        row=json.loads(path.read_text());validate_pose_row(row,i,proto,proof,prep_rows[i]['result']);rows.append(row)
    return rows


def pose_budget(additional=0):
    used=sum(p.stat().st_size for p in PROGRESS.rglob('*') if p.is_file())
    if used+additional>MAX_POSE_JOURNAL or shutil.disk_usage('.').free<RESERVE+additional:
        raise RuntimeError('Pose journal512MiB/1GiB reserve; no deletion or skipped requests')


def aggregate(rows,expected=REQUESTS):
    if len(rows)!=expected or [r['request'] for r in rows]!=list(range(expected)):
        raise ValueError('Cannot aggregate an incomplete/duplicate request set')
    if any(len(row['relative_errors'][k])!=45 for row in rows for k in ('rotation_deg','translation_deg')):
        raise ValueError('Every request must retain exactly45 pairs')
    r=torch.tensor([x for row in rows for x in row['relative_errors']['rotation_deg']],dtype=torch.float32)
    t=torch.tensor([x for row in rows for x in row['relative_errors']['translation_deg']],dtype=torch.float32)
    if len(r)!=45*expected or len(t)!=len(r) or not torch.isfinite(r).all() or not torch.isfinite(t).all():
        raise ValueError('All45 camera pairs per request must be retained')
    pooled={f'RRA_at_{k}':float((r<k).float().mean()) for k in (5,15,30)}
    pooled.update({f'RTA_at_{k}':float((t<k).float().mean()) for k in (5,15,30)})
    pooled['mAA_30']=float(calculate_auc(r,t,max_threshold=30))
    return {'macro_mean_request_metrics':{k:statistics.mean(row['metrics'][k] for row in rows) for k in METRICS},
        'pooled_all_pair_metrics':pooled,'request_count':len(rows),'total_pair_count':len(r),
        'pnp_failed_view_count':sum(len(row['pnp_failed_view_indices']) for row in rows),
        'duplicate_pair_count':sum(len(row['duplicate_camera_pairs']) for row in rows),
        'zero_focal_request_count':sum(row['estimated_focal']==0 for row in rows)}


def final_report(rows,proto):
    return {'status':'full1000_source51_landscape_candidate_pose_evaluation_complete_not_formal_Table1',
        'protocol':proto,'protocol_sha256':signature(proto),**aggregate(rows),
        'request_checkpoint_sha256':{f'request_{i:03d}.json':sha(PROGRESS/f'request_{i:03d}.json') for i in range(REQUESTS)},
        'author_protocol_equivalence_verified':False,'paper_checkpoint_mapping_verified':False,
        'formal_Table1_result':False,'full_paper_completed':False,
        'saved_predictions_independently_reinferred':False,
        'note':'All actual input GT and saved poses/pair metrics rechecked; prediction code/hashes are provenance, not an independent second model inference.'}


def run(args):
    torch.set_num_threads(2)
    if args.dry_run and (not PREP_SUMMARY.is_file() or not INPUT_PROOF.is_file()):
        value=readiness_snapshot()
        if args.save_gate_snapshot:atomic_new(GATE_SNAPSHOT,value)
        print(json.dumps(value,indent=2),flush=True);return
    initial,prep_rows,proof=checked_inputs()  # MUST precede any model-loading path.
    proto=protocol(args,proof);proto_sha=signature(proto)
    identity={'protocol':proto,'protocol_sha256':proto_sha,'preparation_initial_sha256':sha(PREP_ROOT/'initial.json'),
        'preparation_request_sha256':proof['request_sha256'],'scope':'source51_candidate_not_formal_Table1'}
    if (PROGRESS/'initial.json').exists() and json.loads((PROGRESS/'initial.json').read_text())!=identity:
        raise ValueError('Evaluation identity changed; never rewrite checkpoint')
    rows=read_pose_prefix(PROGRESS,proto,proof,prep_rows)
    if args.verify_only:
        if OUTPUT.exists():
            if json.loads(OUTPUT.read_text())!=final_report(rows,proto):raise ValueError('Final compact report differs')
        print(f'VERIFIED {len(rows)}/{REQUESTS} saved pose requests; no model',flush=True);return
    if args.dry_run:
        print(json.dumps({'status':'full1000_actual_inputs_fresh_replay_verified_ready_for_GPU',
            'existing_pose_requests':len(rows),'model_loaded':False,'model_forward_count':0,
            'formal_Table1_result':False},indent=2),flush=True);return
    if OUTPUT.exists():raise FileExistsError('Final report exists; use verify-only, no model rerun')
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    if free<10240:raise RuntimeError('Wait for>=10240MiB free GPU, do not compete/stop other tasks')
    if not torch.cuda.is_available():raise RuntimeError('Host CUDA unavailable')
    pose_budget(MAX_FINAL_BYTES);PROGRESS.mkdir(parents=True,exist_ok=True)
    if not (PROGRESS/'initial.json').exists():atomic_new(PROGRESS/'initial.json',identity)
    dataset=Source51Landscape(json.loads(MANIFEST.read_text()),ReadOnlyInputs(prep_rows,initial['identity']['bindings']['preparer']),42,777)
    wrapper=ResizedDataset(REQUESTS,dataset);wrapper.set_epoch(0);mapping=[int(x) for x in wrapper._idxs_mapping]
    bindings=initial['identity']['bindings']
    from fast3r.models.fast3r import Fast3R
    from fast3r.dust3r.inference_multiview import inference
    model=Fast3R.from_pretrained(str(args.checkpoint_dir)).cuda().eval()
    model.set_max_parallel_views_for_head(args.head_chunk_size)
    for i in range(len(rows),REQUESTS):
        pose_budget(MAX_FINAL_BYTES)
        before=initial if i==0 else prep_rows[i-1]['after_state'];prep=prep_rows[i]['result']
        if restore(dataset,mapping,before,bindings)!=i or digest(capture(dataset,mapping,i,bindings))!=prep_rows[i]['before_state_sha256']:
            raise ValueError('Actual sampler before-state differs')
        dataset.trace=[];dataset.pool_attempts=[];dataset.active_request=i;dataset.preparer.prepared_records={}
        views=wrapper[i]
        if (inputs(views)!=prep['returned_views'] or dataset.trace!=prep['load_trace']
                or dataset.pool_attempts!=prep['pool_attempts']
                or capture(dataset,mapping,i+1,bindings)!=prep_rows[i]['after_state']):
            raise ValueError('Actual inputs or shared after-state diverged before inference')
        gt=np.stack([v['camera_pose'] for v in views]).astype(np.float32);network_views=model_inputs(views)
        seed=(args.seed+i)%(2**32)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            output=inference(network_views,model,torch.device('cuda'),dtype='16-mixed',verbose=False)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            corrected=correct_orientation(output['preds'],network_views)
            predicted,focal,failures=predict_poses_reference_fallback(corrected,seed,niter=100)
            metrics,errors=pose_metrics(predicted,gt)
        row={'request':i,'protocol_sha256':proto_sha,'preparation_request_sha256':proof['request_sha256'][f'request_{i:03d}.json'],
            'base_index':prep['base_index'],'scene':views[0]['label'],'request_seed':seed,
            'input_tensor_sha256':[hashlib.sha256(v['img'].contiguous().numpy().tobytes()).hexdigest() for v in views],
            'input_shape':[list(v['img'].shape) for v in network_views],
            'true_shape':[v['true_shape'].tolist() for v in network_views],
            'gt_c2w':gt.tolist(),'predicted_c2w':predicted.tolist(),'metrics':metrics,'relative_errors':errors,
            'pair_count':45,'estimated_focal':float(focal),'pnp_failed_view_indices':failures,
            'duplicate_camera_pairs':[[a,b] for a in range(10) for b in range(a+1,10)
                if (views[a]['label'],views[a]['instance'])==(views[b]['label'],views[b]['instance'])],
            'model_inference_completed':True,'formal_Table1_result':False,
            'runtime':{'forward_seconds':seconds,'cuda_max_memory_allocated_bytes':torch.cuda.max_memory_allocated(),
                'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda}}
        validate_pose_row(row,i,proto,proof,prep)
        atomic_new(PROGRESS/f'request_{i:03d}.json',row);rows.append(row)
        print(json.dumps({'completed':i+1,'expected':REQUESTS,'scene':row['scene'],
            'pnp_failed_views':len(failures),'mAA_30':metrics['mAA_30'],'formal_Table1_result':False}),flush=True)
        del views,network_views,output,corrected
    rows=read_pose_prefix(PROGRESS,proto,proof,prep_rows)
    result=final_report(rows,proto);payload=json.dumps(result,allow_nan=False).encode()
    if len(payload)>MAX_FINAL_BYTES:raise ValueError('Final compact report exceeded bound')
    pose_budget(MAX_FINAL_BYTES);atomic_new(OUTPUT,result)
    print('ALL1000 candidate pose requests committed; not confirmed author Table1',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir',type=Path,default=CHECKPOINT);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--head-chunk-size',type=int,default=2);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--verify-only',action='store_true');p.add_argument('--save-gate-snapshot',action='store_true')
    args=p.parse_args()
    if args.seed!=42 or args.head_chunk_size!=2:p.error('This version binds seed42/head chunk2; use a new version for other protocols')
    if args.save_gate_snapshot and not args.dry_run:p.error('save-gate-snapshot requires dry-run')
    run(args)
