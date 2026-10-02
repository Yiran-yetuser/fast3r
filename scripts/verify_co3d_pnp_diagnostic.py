#!/usr/bin/env python3
"""Independently recompute saved diagnostic poses/pair metrics, no GPU or network."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from fast3r.eval.cam_pose_metric import camera_to_rel_deg, calculate_auc

REPORT=Path('results/co3d_candidate_pnp_diagnostic_v1_20261003.json')
PROOF=Path('results/co3d_candidate_pnp_diagnostic_verified_20261003.json')
BASELINE=Path('results/co3d_pose_100_seed42_adaptation_v3.json')
PREP=Path('results/co3d_continuous_prepare_v4_20261002')
PROGRESS=Path('results/co3d_candidate_pnp_diagnostic_v1_progress')
REQUESTS=(0,2,3)
BRANCHES=('released','top15_only','search_only','search_top15')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def verify():
    torch.set_num_threads(2)
    report=json.loads(REPORT.read_text());baseline=json.loads(BASELINE.read_text())
    p=report['protocol'];p_sha=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    if (report['protocol_sha256']!=p_sha or report['formal_Table1_result'] is not False
        or report['full_paper_completed'] is not False or report['model_forward_count']!=3
        or report['request_count']!=3 or report['branches_per_request']!=4 or report['network_bytes']!=0
        or report['no_diagnostic_average_reported_because_probes_selected_using_archived_performance'] is not True
        or p['GT_used_by_network_focal_or_PnP'] is not False
        or p['search_is_paper_random_guess_replication'] is not False):
        raise ValueError('Diagnostic scope/provenance changed')
    if sha(BASELINE)!=p['baseline_report_sha256'] or baseline['protocol_sha256']!=p['baseline_protocol_sha256']:
        raise ValueError('Archived baseline changed')
    for path,expected in p['source_sha256'].items():
        if sha(path)!=expected:raise ValueError('Diagnostic bound code changed')
    if [r['request'] for r in report['requests']]!=list(REQUESTS):raise ValueError('Request set/order changed')
    compact=[]
    for row in report['requests']:
        i=row['request'];archived=baseline['requests'][i];name=f'request_{i:03d}.json'
        if (row!=json.loads((PROGRESS/name).read_text()) or row['protocol_sha256']!=p_sha
            or sha(PREP/name)!=p['preparation_request_sha256'][str(i)]
            or row['input_tensor_sha256']!=archived['input_tensor_sha256']
            or row['gt_c2w']!=archived['gt_c2w'] or row['scene']!=archived['scene']
            or row['duplicate_camera_pairs']!=archived['duplicate_camera_pairs']
            or row['paired_same_forward'] is not True or row['forward_count']!=1
            or row['prediction_before']!=row['prediction_after'] or len(row['prediction_before'])!=10
            or set(row['branches'])!=set(BRANCHES)):
            raise ValueError('Fixed input, same-forward or checkpoint link mismatch')
        gt=torch.tensor(row['gt_c2w'],dtype=torch.float32)
        for branch in BRANCHES:
            b=row['branches'][branch];pred=torch.tensor(b['predicted_c2w'],dtype=torch.float32)
            failures=b['pnp_failed_view_indices']
            if (pred.shape!=(10,4,4) or not torch.isfinite(pred).all() or b['pair_count']!=45
                or len(failures)!=len(set(failures)) or any(type(j) is not int or not 0<=j<10 for j in failures)):
                raise ValueError('Invalid matrices/failure indices')
            if any(not torch.equal(pred[j],torch.eye(4)) for j in failures):raise ValueError('Lost identity fallback')
            r,t=camera_to_rel_deg(pred,gt,'cpu',10)
            metrics={f'RRA_at_{k}':float((r<k).float().mean()) for k in (5,15,30)}
            metrics.update({f'RTA_at_{k}':float((t<k).float().mean()) for k in (5,15,30)})
            metrics['mAA_30']=float(calculate_auc(r,t,max_threshold=30))
            if set(b['metrics'])!=set(metrics) or any(not np.isfinite(b['metrics'][k]) or abs(metrics[k]-b['metrics'][k])>1e-7 for k in metrics):
                raise ValueError('Metric recomputation failed')
            for key,values in [('rotation_deg',r),('translation_deg',t)]:
                if np.asarray(b['relative_errors'][key]).shape!=(45,) or not np.allclose(values.numpy(),b['relative_errors'][key],rtol=0,atol=1e-7):
                    raise ValueError('Pairwise error recomputation failed')
            if branch=='released':
                if (b['estimated_focal']!=archived['estimated_focal'] or failures!=archived['pnp_failed_view_indices']
                    or not np.allclose(pred.numpy(),archived['predicted_c2w'],rtol=0,atol=1e-5)
                    or any(abs(metrics[k]-archived['metrics'][k])>1e-7 for k in metrics)):
                    raise ValueError('Archived baseline failed to reproduce')
            stats=b['confidence_masks']
            if len(stats)!=10:raise ValueError('Mask count mismatch')
            for s in stats:
                if not 0<=s['retained_points']<=s['total_points'] or not np.isfinite(s['threshold']):raise ValueError('Invalid mask accounting')
            compact.append({'request':i,'scene':row['scene'],'branch':branch,'metrics':metrics,
                'estimated_focal':b['estimated_focal'],'pnp_failed_view_count':len(failures),
                'retained_mask_fraction_min':min(s['retained_points']/s['total_points'] for s in stats),
                'retained_mask_fraction_max':max(s['retained_points']/s['total_points'] for s in stats),
                'baseline_pose_max_abs_difference':row['baseline_pose_max_abs_difference']})
    return {'status':'three_selected_diagnostic_requests_and_twelve_branches_independently_verified',
        'report_path':str(REPORT),'report_sha256':sha(REPORT),'verifier_sha256':sha(__file__),
        'baseline_report_sha256':sha(BASELINE),'request_count':3,'branch_count':12,
        'all_pair_errors_and_metrics_recomputed':True,'all_identity_fallbacks_retained':True,
        'baseline_poses_and_metrics_reproduced':True,'network_bytes':0,'model_forward_count':0,
        'verification_limit':'Prediction/mask hashes are provenance assertions checked against frozen generating code; raw predicted maps are not stored and are not independently recomputed here.',
        'formal_Table1_result':False,'full_paper_completed':False,'rows':compact}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emit-proof-patch',action='store_true');args=parser.parse_args();result=verify()
    if args.emit_proof_patch:
        if PROOF.exists():raise FileExistsError('Preserve historical proof')
        print('*** Begin Patch\n*** Add File: '+str(PROOF))
        for line in (json.dumps(result,indent=2,allow_nan=False)+'\n').splitlines():print('+'+line)
        print('*** End Patch')
    else:print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
