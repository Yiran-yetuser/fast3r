#!/usr/bin/env python3
"""Independent saved-matrix/input-byte verification, no GPU or network."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from fast3r.eval.cam_pose_metric import camera_to_rel_deg, calculate_auc
from fast3r_hf_re10k_pose_eval import sha
from fast3r_hf_co3d_pose_smoke import signature
from audit_co3d_portrait_geometry import load_context, fixed_inputs, REQUESTS
from diagnose_co3d_landscape_inputs import OUTPUT, PROGRESS, BRANCHES, expected_inputs

PROOF = Path('results/co3d_landscape_input_diagnostic_verified_20261003.json')


def verify():
    torch.set_num_threads(2); report = json.loads(OUTPUT.read_text()); p = report['protocol']
    if (report['protocol_sha256'] != signature(p) or report['formal_Table1_result'] is not False
            or report['full_paper_completed'] is not False or report['network_bytes'] != 0
            or report['no_diagnostic_average_reported_because_performance_selected'] is not True
            or p['GT_used_by_network_focal_or_PnP'] is not False
            or p['paired_same_forward_across_different_inputs'] is not False
            or p['archived_original_model_forward_rerun'] is not False):
        raise ValueError('Input diagnostic scope/provenance mismatch')
    for name, h in p['source_sha256'].items():
        if sha(name) != h: raise ValueError('Bound diagnostic code changed')
    context = load_context(); baseline = context[0]
    if [r['request'] for r in report['requests']] != list(REQUESTS): raise ValueError('Request order/set changed')
    compact = []; total_forwards = 0
    for row in report['requests']:
        i = row['request']; old = baseline['requests'][i]
        if (row != json.loads((PROGRESS/f'request_{i:03d}.json').read_text())
                or row['gt_c2w'] != old['gt_c2w'] or row['duplicate_camera_pairs'] != old['duplicate_camera_pairs']
                or row['original_input_tensor_sha256'] != old['input_tensor_sha256']
                or row['protocol_sha256'] != report['protocol_sha256']
                or row['paired_same_forward_across_different_inputs'] is not False
                or row['formal_Table1_result'] is not False or set(row['branches']) != set(BRANCHES)):
            raise ValueError('Lost fixed GT/checkpoint/fallback scope')
        native, forced = fixed_inputs(i, context); expected = expected_inputs(native, forced)
        if (row['forced_crop_intrinsics'] != [v['camera_intrinsics'].tolist() for v in forced]
                or row['forced_crop_rng_markers'] != [v['rng'] for v in forced]
                or any(np.linalg.det(K) <= 0 for K in row['forced_crop_intrinsics'])):
            raise ValueError('Forced-crop intrinsics/RNG/input replay mismatch')
        seen = {}; row_forwards = 0; gt = torch.tensor(row['gt_c2w'], dtype=torch.float32)
        for name in BRANCHES:
            b = row['branches'][name]; key = signature(expected[name]); repeat = key in seen
            if (b['input_signature'] != expected[name] or b['new_forward_count'] != int(not repeat)
                    or b['same_forward_as_identical_other_new_input'] != repeat or b['pair_count'] != 45):
                raise ValueError('Input/forward/pair accounting mismatch')
            pred = torch.tensor(b['predicted_c2w'], dtype=torch.float32); failed = b['pnp_failed_view_indices']
            if (pred.shape != (10,4,4) or not torch.isfinite(pred).all()
                    or len(failed) != len(set(failed)) or any(type(j) is not int or not 0<=j<10 for j in failed)
                    or any(not torch.equal(pred[j], torch.eye(4)) for j in failed)
                    or not np.isfinite(b['estimated_focal']) or b['estimated_focal'] < 0):
                raise ValueError('Invalid matrix/failure/focal')
            r, t = camera_to_rel_deg(pred, gt, 'cpu', 10)
            metrics = {f'RRA_at_{k}':float((r<k).float().mean()) for k in (5,15,30)}
            metrics.update({f'RTA_at_{k}':float((t<k).float().mean()) for k in (5,15,30)})
            metrics['mAA_30'] = float(calculate_auc(r,t,max_threshold=30))
            if set(b['metrics']) != set(metrics) or any(abs(metrics[k]-b['metrics'][k])>1e-7 for k in metrics):
                raise ValueError('Independent metric recomputation failed')
            for k, values in [('rotation_deg',r),('translation_deg',t)]:
                if np.asarray(b['relative_errors'][k]).shape!=(45,) or not np.allclose(values.numpy(),b['relative_errors'][k],atol=1e-7,rtol=0):
                    raise ValueError('Independent pair error recomputation failed')
            if repeat:
                if any(b[k] != seen[key][k] for k in ('predicted_c2w','metrics','prediction_signature','estimated_focal','pnp_failed_view_indices')):
                    raise ValueError('Identical input did not share immutable prediction')
            if i == 2 and (not np.allclose(pred.numpy(),old['predicted_c2w'],atol=1e-5,rtol=0)
                    or b['estimated_focal'] != old['estimated_focal'] or failed != old['pnp_failed_view_indices']
                    or any(abs(metrics[k]-old['metrics'][k])>1e-7 for k in metrics)):
                raise ValueError('Landscape control baseline changed')
            seen[key]=b; row_forwards += b['new_forward_count']
            compact.append({'request':i,'scene':row['scene'],'branch':name,'metrics':metrics,
                'estimated_focal':b['estimated_focal'],'pnp_failed_view_count':len(failed),
                'original_mAA_30':old['metrics']['mAA_30'],
                'changed_RGB_view_count':sum(x['img_tensor_sha256']!=h for x,h in zip(expected[name],old['input_tensor_sha256']))})
        if row_forwards != row['model_forward_count']: raise ValueError('Request forward count mismatch')
        total_forwards += row_forwards
    if total_forwards != report['model_forward_count'] or total_forwards != 5:
        raise ValueError('Distinct new input forward total mismatch')
    return {'status':'three_fixed_frame_requests_six_new_branches_independently_verified',
        'report_sha256':sha(OUTPUT),'verifier_sha256':sha(__file__),'baseline_report_sha256':p['baseline_report_sha256'],
        'all_fixed_RGB_tensors_GT_and_crop_intrinsics_replayed':True,
        'all_270_new_branch_pairs_recomputed':True, 'all_identity_fallbacks_retained':True,
        'landscape_control_matches_archived_baseline':True, 'distinct_new_forward_count':5,
        'verification_limit':'No trained network re-execution: saved prediction hashes remain provenance assertions. Different input conditions are not same-forward and selected probes have no unbiased average.',
        'formal_Table1_result':False,'full_paper_completed':False,'rows':compact}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--emit-proof-patch',action='store_true')
    args=p.parse_args(); result=verify()
    if args.emit_proof_patch:
        if PROOF.exists(): raise FileExistsError('Preserve historical proof')
        print('*** Begin Patch\n*** Add File: '+str(PROOF))
        for line in (json.dumps(result,indent=2,allow_nan=False)+'\n').splitlines(): print('+'+line)
        print('*** End Patch')
    else: print(json.dumps(result,indent=2,allow_nan=False))
