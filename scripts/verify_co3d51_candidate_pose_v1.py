"""Independent full1000 saved-pose/input/aggregate audit, no model/download.

The network predictions are not independently reinferred. Every saved pose,
GT, pair error and arithmetic aggregate is independently recomputed instead.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from co3d_request_journal import atomic_new
from fast3r.eval.cam_pose_metric import camera_to_rel_deg,calculate_auc
from co3d51_source_lazy_v1 import MANIFEST
from fast3r_hf_re10k_pose_eval import sha
from verify_co3d_candidate_pose_report import KEYS,metrics,require_equal_metrics

REPORT=Path('results/co3d51_pose_seed42_candidate_v1.json')
SUMMARY=Path('results/co3d51_pose_seed42_verified_20261003.json')
PROGRESS=Path('results/co3d51_pose_seed42_progress_v1')
PROOF=Path('results/co3d51_source_inputs_full_verified_20261003.json')
PREP_ROOT=Path('results/co3d51_source_prepare_v1_progress')
PREP_SUMMARY=Path('results/co3d51_source_prepare_v1_summary_20261003.json')
REQUESTS=1000


def check_bound_source(path,h):
    p=Path(path);root=Path.cwd().resolve()
    if '..' in p.parts:raise ValueError('Bound source identity/path changed')
    try:relative=p.resolve().relative_to(root)
    except ValueError as error:raise ValueError('Bound source escaped project root') from error
    if not relative.parts or relative.parts[0] not in ('scripts','fast3r') or sha(p)!=h:
        raise ValueError('Bound source identity/path changed')


def strict_header(report,proof,source_summary):
    expected={f'request_{i:03d}.json' for i in range(REQUESTS)}
    if (report.get('request_count')!=REQUESTS or report.get('total_pair_count')!=REQUESTS*45
            or any(report.get(k) is not False for k in ('formal_Table1_result','full_paper_completed',
                'author_protocol_equivalence_verified','paper_checkpoint_mapping_verified','saved_predictions_independently_reinferred'))
            or set(report.get('request_checkpoint_sha256',{}))!=expected
            or proof.get('verified_request_count')!=REQUESTS
            or proof.get('expected_full_request_count')!=REQUESTS
            or proof.get('all_1000_requests_prepared_and_replayed') is not True
            or proof.get('input_tensor_GT_rng_trace_and_shared_state_exact_equal') is not True
            or proof.get('raw_SHA_CRC_processed_SHA_and_receipts_checked') is not True
            or proof.get('model_forward_count')!=0 or proof.get('network_bytes')!=0
            or any(proof.get(k) is not False for k in ('formal_Table1_result','full_paper_completed','author_protocol_equivalence_verified'))
            or source_summary.get('request_count')!=REQUESTS
            or source_summary.get('model_forward_count')!=0
            or any(source_summary.get(k) is not False for k in
                ('formal_Table1_result','full_paper_completed','author_protocol_equivalence_verified'))
            or proof.get('request_sha256')!=source_summary.get('request_sha256')
            or set(proof.get('request_sha256',{}))!=expected):
        raise ValueError('Incomplete1000 coverage or dishonest paper/input/model scope')


def inspect_pose(row,prep,i,protocol_sha,source_sha,seed):
    views=prep['returned_views']
    if (len(views)!=10 or row.get('request')!=i or row.get('protocol_sha256')!=protocol_sha
            or row.get('preparation_request_sha256')!=source_sha
            or row.get('pair_count')!=45 or row.get('model_inference_completed') is not True
            or row.get('formal_Table1_result') is not False or row.get('request_seed')!=(seed+i)%(2**32)
            or row['base_index']!=prep['base_index'] or row['scene']!=views[0]['label']
            or row['input_tensor_sha256']!=[v['img_tensor_sha256'] for v in views]
            or row['input_shape']!=[[1]+v['img_shape'] for v in views]
            or row['true_shape']!=[[v['true_shape_hw']] for v in views]):
        raise ValueError('Pose input/provenance differs from actual prepared request')
    gt=np.asarray(row['gt_c2w'],np.float32);pred=np.asarray(row['predicted_c2w'],np.float32)
    if (gt.shape!=(10,4,4) or pred.shape!=gt.shape or not np.isfinite(gt).all() or not np.isfinite(pred).all()
            or not np.array_equal(gt,np.asarray([v['camera_pose'] for v in views],np.float32))):
        raise ValueError('Saved GT/prediction invalid or changed')
    failures=row['pnp_failed_view_indices']
    if len(failures)!=len(set(failures)) or any(type(j) is not int or not 0<=j<10 for j in failures):
        raise ValueError('PnP fallback index changed')
    for j in failures:
        if not np.array_equal(pred[j],np.eye(4,dtype=np.float32)):raise ValueError('Identity fallback dropped')
    focal=row['estimated_focal']
    if not np.isfinite(focal) or focal<0:raise ValueError('Saved estimated focal invalid')
    duplicates=[[a,b] for a in range(10) for b in range(a+1,10)
        if (views[a]['label'],views[a]['instance'])==(views[b]['label'],views[b]['instance'])]
    if row['duplicate_camera_pairs']!=duplicates:raise ValueError('Duplicate camera pairs changed')
    r,t=camera_to_rel_deg(torch.tensor(pred),torch.tensor(gt),'cpu',10)
    if len(r)!=45 or len(t)!=45 or not torch.isfinite(r).all() or not torch.isfinite(t).all():
        raise ValueError('Nonfinite or missing pair errors')
    actual=metrics(r,t);require_equal_metrics(actual,row['metrics'])
    for key,value in [('rotation_deg',r),('translation_deg',t)]:
        saved=np.asarray(row['relative_errors'][key])
        if saved.shape!=(45,) or not np.allclose(value.numpy(),saved,atol=1e-7,rtol=0):
            raise ValueError('Independently recomputed pair errors differ')
    return actual,r,t,duplicates


def verify():
    torch.set_num_threads(2)
    report=json.loads(REPORT.read_text());proof=json.loads(PROOF.read_text())
    source_summary=json.loads(PREP_SUMMARY.read_text());strict_header(report,proof,source_summary)
    proto=report['protocol'];proto_sha=hashlib.sha256(json.dumps(proto,sort_keys=True).encode()).hexdigest()
    if (report['protocol_sha256']!=proto_sha or proto['request_count']!=REQUESTS or proto['seed']!=42
            or proto['head_chunk_size']!=2 or proto['precision']!='16-mixed'
            or proto['network_input_keys']!=['img','true_shape']
            or proto['GT_pose_intrinsics_depth_mask_used_by_network_or_focal_PnP'] is not False
            or any(proto[k] is not False for k in ('formal_Table1_result','full_paper_completed',
                'author_protocol_equivalence_verified','paper_checkpoint_mapping_verified'))
            or proto['full_input_proof_sha256']!=sha(PROOF)
            or proto['complete_preparation_summary_sha256']!=sha(PREP_SUMMARY)
            or proto['full_input_final_state_sha256']!=proof['final_state_sha256']
            or proto['candidate_sha256']!=sha(MANIFEST)
            or proof['verifier_sha256']!=sha('scripts/verify_co3d51_source_inputs_v1.py')):
        raise ValueError('Protocol or full input proof changed/promoted')
    for path,h in proto['source_code_sha256'].items():
        check_bound_source(path,h)
    for name,key in [('model.safetensors','checkpoint_weight_sha256'),('config.json','checkpoint_config_sha256')]:
        if sha(Path('checkpoints/Fast3R_ViT_Large_512')/name)!=proto[key]:raise ValueError('Weight/config changed')
    initial=json.loads((PROGRESS/'initial.json').read_text())
    if (initial['protocol']!=proto or initial['protocol_sha256']!=proto_sha
            or initial['preparation_request_sha256']!=proof['request_sha256']
            or initial['preparation_initial_sha256']!=proof['initial_sha256']
            or sha(PREP_ROOT/'initial.json')!=proof['initial_sha256']):
        raise ValueError('Evaluation initial identity changed')
    if {p.name for p in PROGRESS.glob('request_*.json')}!=set(report['request_checkpoint_sha256']):
        raise ValueError('Saved pose request set incomplete')
    compact=[];rr=[];tt=[];frames=set();trajectories=set()
    for i in range(REQUESTS):
        name=f'request_{i:03d}.json';pose_path=PROGRESS/name;prep_path=PREP_ROOT/name
        if (sha(pose_path)!=report['request_checkpoint_sha256'][name]
                or sha(prep_path)!=proof['request_sha256'][name]):raise ValueError('Request SHA changed')
        row=json.loads(pose_path.read_text());prep=json.loads(prep_path.read_text())['result']
        values,r,t,duplicate=inspect_pose(row,prep,i,proto_sha,proof['request_sha256'][name],42)
        rr.append(r);tt.append(t);trajectories.add(row['scene'])
        unique={(v['label'],v['instance']) for v in prep['returned_views']};frames.update(unique)
        compact.append({'request':i,'scene':row['scene'],'unique_view_count':len(unique),
            'duplicate_pair_count':len(duplicate),'estimated_focal':row['estimated_focal'],
            'pnp_failed_view_indices':row['pnp_failed_view_indices'],'metrics':values,
            'evaluation_checkpoint_sha256':sha(pose_path)})
    macro={k:statistics.mean(row['metrics'][k] for row in compact) for k in KEYS}
    pooled=metrics(torch.cat(rr),torch.cat(tt))
    require_equal_metrics(macro,report['macro_mean_request_metrics'])
    require_equal_metrics(pooled,report['pooled_all_pair_metrics'])
    counts={'pnp_failed_view_count':sum(len(row['pnp_failed_view_indices']) for row in compact),
        'duplicate_pair_count':sum(row['duplicate_pair_count'] for row in compact),
        'zero_focal_request_count':sum(row['estimated_focal']==0 for row in compact)}
    if any(report[k]!=v for k,v in counts.items()):raise ValueError('Failure/duplicate aggregate differs')
    return {'status':'all1000_source51_landscape_candidate_saved_poses_independently_verified_not_Table1',
        'paper_mapping':'section4.2/Table1 public-HF candidate; author equivalence not confirmed',
        'complete_report_path':str(REPORT),'complete_report_sha256':sha(REPORT),
        'full_input_proof_sha256':sha(PROOF),'verifier_sha256':sha(__file__),
        'request_count':REQUESTS,'pair_count':45000,'unique_trajectories':len(trajectories),
        'unique_categories':len({s.split('/')[0] for s in trajectories}),'unique_returned_frames':len(frames),
        'requests_with_duplicate_views':sum(row['unique_view_count']<10 for row in compact),**counts,
        'macro_mean_request_metrics':macro,'pooled_all_pair_metrics':pooled,
        'all_input_GT_saved_pose_errors_and_aggregates_recomputed':True,
        'all_identity_fallbacks_and_duplicate_pairs_retained':True,
        'saved_predictions_independently_reinferred':False,'network_bytes':0,'model_forward_count':0,
        'formal_Table1_result':False,'author_protocol_equivalence_verified':False,
        'paper_checkpoint_mapping_verified':False,'full_paper_completed':False,'requests':compact}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',action='store_true')
    args=p.parse_args();result=verify()
    if args.output:atomic_new(SUMMARY,result)
    elif SUMMARY.exists() and json.loads(SUMMARY.read_text())!=result:
        raise ValueError('Saved independent proof changed; never overwrite')
    print(json.dumps({k:v for k,v in result.items() if k!='requests'},indent=2))
