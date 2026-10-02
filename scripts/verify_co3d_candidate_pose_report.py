#!/usr/bin/env python3
"""Read-only audit of all saved candidate poses; emit a compact result via apply_patch.

No model, GPU, downloads or preparation replay. The complete input replay has its
own immutable proof. This audit links every final row to that proof and to its
checkpoint, recomputes the released metrics, and counts every identity fallback.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from fast3r.eval.cam_pose_metric import camera_to_rel_deg, calculate_auc

REPORT = Path('results/co3d_pose_100_seed42_adaptation_v3.json')
SUMMARY = Path('results/co3d_pose_100_seed42_verified_summary_20261003.json')
EVAL_ROOT = Path('results/co3d_pose_100_seed42_progress_v3')
PROOF = Path('results/co3d_v4_prefix_full_20261003.json')
KEYS = tuple([f'RRA_at_{k}' for k in (5,15,30)] +
             [f'RTA_at_{k}' for k in (5,15,30)] + ['mAA_30'])


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def metrics(r,t):
    result={f'RRA_at_{k}':float((r<k).float().mean()) for k in (5,15,30)}
    result.update({f'RTA_at_{k}':float((t<k).float().mean()) for k in (5,15,30)})
    result['mAA_30']=float(calculate_auc(r,t,max_threshold=30))
    return result


def require_equal_metrics(actual,expected):
    if set(actual)!=set(KEYS) or set(expected)!=set(KEYS):
        raise ValueError('Wrong metric key set')
    for k in KEYS:
        if not np.isfinite(expected[k]) or not 0<=expected[k]<=1:
            raise ValueError('Nonfinite or out-of-range metric: '+k)
        if abs(actual[k]-expected[k])>1e-7:
            raise ValueError('Metric mismatch: '+k)


def verify():
    torch.set_num_threads(2)
    report=json.loads(REPORT.read_text()); proof=json.loads(PROOF.read_text())
    proto=report['protocol']; proto_sha=hashlib.sha256(json.dumps(proto,sort_keys=True).encode()).hexdigest()
    if report['protocol_sha256']!=proto_sha: raise ValueError('Protocol digest mismatch')
    if (report['request_count']!=100 or report['total_pair_count']!=4500 or
        report['formal_Table1_result'] is not False or report['full_paper_completed'] is not False or
        report['author_split_equivalence_verified'] is not False or
        report['paper_checkpoint_mapping_verified'] is not False):
        raise ValueError('Coverage or paper scope mismatch')
    if (proto['network_input_keys']!=['img','true_shape'] or
        proto['GT_pose_intrinsics_depth_mask_used_by_network_or_focal_PnP'] is not False):
        raise ValueError('Network/GT separation mismatch')
    if (sha(PROOF)!=proto['full_prefix_proof_sha256'] or
        proof['verified_request_count']!=100 or proof['all_100_requests_prepared'] is not True or
        proof['input_tensors_GT_rng_and_shared_state_exact_equal'] is not True or
        proof['network_bytes']!=0 or proof['model_forward_count']!=0 or
        proof['final_state_sha256']!=proto['full_prefix_final_state_sha256']):
        raise ValueError('Input preparation proof mismatch')
    for p,h in proto['source_code_sha256'].items():
        if sha(p)!=h: raise ValueError('Bound source changed: '+p)
    checkpoint=Path('checkpoints/Fast3R_ViT_Large_512')
    for name,key in [('model.safetensors','checkpoint_weight_sha256'),('config.json','checkpoint_config_sha256')]:
        if sha(checkpoint/name)!=proto[key]: raise ValueError('Checkpoint bytes changed')
    prep_root=Path(proto['preparation_root'])
    initial=json.loads((EVAL_ROOT/'initial.json').read_text())
    if initial['protocol']!=proto or initial['preparation_request_sha256']!=proof['request_sha256']:
        raise ValueError('Evaluation initial identity mismatch')
    if {p.name for p in EVAL_ROOT.glob('request_*.json')}!={f'request_{i:03d}.json' for i in range(100)}:
        raise ValueError('Evaluation checkpoint set incomplete')
    rows=report['requests']
    if len(rows)!=100 or [x['request'] for x in rows]!=list(range(100)):
        raise ValueError('Final request ordering/uniqueness mismatch')
    compact=[]; rr=[]; tt=[]; total_failures=0; unique_frames=set()
    for i,row in enumerate(rows):
        name=f'request_{i:03d}.json'; prep_path=prep_root/name
        if sha(prep_path)!=proof['request_sha256'][name]: raise ValueError('Prepared request changed')
        prep=json.loads(prep_path.read_text())['result']; views=prep['returned_views']
        if row!=json.loads((EVAL_ROOT/name).read_text()): raise ValueError('Final row differs from checkpoint')
        if (row['protocol_sha256']!=proto_sha or row['preparation_request_sha256']!=proof['request_sha256'][name] or
            row['base_index']!=prep['base_index'] or row['scene']!=views[0]['label'] or
            row['request_seed']!=proto['seed']+i or row['model_inference_completed'] is not True or
            row['pair_count']!=45 or len(views)!=10): raise ValueError('Row provenance mismatch')
        if row['input_tensor_sha256']!=[v['img_tensor_sha256'] for v in views]:
            raise ValueError('Model RGB input digest mismatch')
        if (row['input_shape']!=[[1]+v['img_shape'] for v in views] or
            row['true_shape']!=[[v['true_shape_hw']] for v in views]):
            raise ValueError('Model input shape mismatch')
        gt=np.asarray(row['gt_c2w'],dtype=np.float32); pred=np.asarray(row['predicted_c2w'],dtype=np.float32)
        if not np.array_equal(gt,np.asarray([v['camera_pose'] for v in views],dtype=np.float32)):
            raise ValueError('GT differs from prepared input')
        if pred.shape!=(10,4,4) or gt.shape!=(10,4,4) or not np.isfinite(pred).all() or not np.isfinite(gt).all():
            raise ValueError('Nonfinite/invalid pose matrices')
        failures=row['pnp_failed_view_indices']
        if len(set(failures))!=len(failures) or any(type(j) is not int or not 0<=j<10 for j in failures):
            raise ValueError('Invalid PnP failure indices')
        for j in failures:
            if not np.array_equal(pred[j],np.eye(4,dtype=np.float32)):
                raise ValueError('PnP fallback was not retained as identity')
        focal=row['estimated_focal']
        if not np.isfinite(focal) or focal<0: raise ValueError('Invalid estimated focal')
        duplicate=[[a,b] for a in range(10) for b in range(a+1,10)
                   if (views[a]['label'],views[a]['instance'])==(views[b]['label'],views[b]['instance'])]
        if row['duplicate_camera_pairs']!=duplicate: raise ValueError('Duplicate-pair accounting mismatch')
        r,t=camera_to_rel_deg(torch.tensor(pred),torch.tensor(gt),'cpu',10)
        if len(r)!=45 or len(t)!=45 or not torch.isfinite(r).all() or not torch.isfinite(t).all():
            raise ValueError('Invalid relative-pose errors')
        require_equal_metrics(metrics(r,t),row['metrics'])
        for k,values in [('rotation_deg',r),('translation_deg',t)]:
            saved=np.asarray(row['relative_errors'][k])
            if saved.shape!=(45,) or not np.allclose(values.numpy(),saved,atol=1e-7,rtol=0):
                raise ValueError('Saved pair errors disagree')
        rr.append(r);tt.append(t);total_failures+=len(failures)
        frames={(v['label'],v['instance']) for v in views};unique_frames.update(frames)
        compact.append({'request':i,'scene':row['scene'],'unique_view_count':len(frames),
            'duplicate_pair_count':len(duplicate),'estimated_focal':focal,
            'pnp_failed_view_indices':failures,'metrics':row['metrics'],
            'evaluation_checkpoint_sha256':sha(EVAL_ROOT/name)})
    macro={k:statistics.mean(x['metrics'][k] for x in compact) for k in KEYS}
    pooled=metrics(torch.cat(rr),torch.cat(tt))
    require_equal_metrics(macro,report['macro_mean_request_metrics'])
    require_equal_metrics(pooled,report['pooled_all_pair_metrics'])
    if total_failures!=report['pnp_failed_view_count']: raise ValueError('Total PnP failures disagree')
    paper={'RRA_at_5':0.902,'RRA_at_15':0.962,'RTA_at_5':0.682,'RTA_at_15':0.816,'mAA_30':0.750}
    return {'status':'all_100_candidate_pose_requests_independently_verified',
        'complete_report_sha256':sha(REPORT),'complete_report_path':str(REPORT),
        'verifier_sha256':sha(__file__),'protocol_sha256':proto_sha,
        'input_prefix_proof_sha256':sha(PROOF),'request_count':100,'pair_count':4500,
        'unique_trajectories':len({x['scene'] for x in compact}),
        'unique_categories':len({x['scene'].split('/')[0] for x in compact}),
        'unique_returned_frames':len(unique_frames),
        'requests_with_duplicate_views':sum(x['unique_view_count']<10 for x in compact),
        'duplicate_pair_count':sum(x['duplicate_pair_count'] for x in compact),
        'zero_focal_request_count':sum(x['estimated_focal']==0 for x in compact),
        'requests_with_pnp_failures':sum(bool(x['pnp_failed_view_indices']) for x in compact),
        'pnp_failed_view_count':total_failures,'pnp_attempted_view_count':1000,
        'all_checkpoint_final_rows_identical':True,'all_GT_and_input_hashes_match_preparation':True,
        'all_saved_poses_pair_errors_and_metrics_recomputed':True,'all_fallback_identity_poses_retained':True,
        'macro_mean_request_metrics':macro,'pooled_all_pair_metrics':pooled,
        'paper_Table1_Fast3R_reference_only':paper,
        'paper_source':'https://arxiv.org/html/2501.13928v2#S4.SS2',
        'reference_minus_candidate_percentage_points':{k:100*(paper[k]-macro[k]) for k in paper},
        'formal_Table1_result':False,'author_split_equivalence_verified':False,
        'paper_checkpoint_mapping_verified':False,'full_paper_completed':False,
        'network_bytes':0,'model_forward_count':0,'requests':compact}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emit-summary-patch',action='store_true')
    args=parser.parse_args(); result=verify()
    if args.emit_summary_patch:
        if SUMMARY.exists(): raise FileExistsError('Preserve existing summary')
        content=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        print('*** Begin Patch\n*** Add File: '+str(SUMMARY))
        for line in content.splitlines(): print('+'+line)
        print('*** End Patch')
    else:
        print(json.dumps({k:v for k,v in result.items() if k!='requests'},indent=2,allow_nan=False))
