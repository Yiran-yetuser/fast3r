#!/usr/bin/env python3
"""Same-forward two-orientation diagnostic, not Table1 or a protocol replacement."""
import argparse
import hashlib
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from fast3r_hf_co3d_pose_smoke import (
    load_archived_draw, model_inputs, correct_orientation, protocol, signature,
    predict_poses, pose_metrics, validate_row, save_new, sha,
)

OUTPUT = Path('results/co3d_orientation_diagnostic_20261002.json')
BASELINE = Path('results/co3d_pose_draw0_seed42_20261002.json')
BRANCHES = ('released_landscape_correction', 'raw_portrait_no_correction')


def evaluate_branches(preds, views, seed, gt):
    # Shallow correction copies dictionaries, never mutates original tensors.
    variants = [correct_orientation(preds, views),
                [{k:p[k] for k in ('pts3d_in_other_view','conf')} for p in preds]]
    result = {}
    for name, values in zip(BRANCHES, variants):
        poses, focal, failures = predict_poses(values, seed, niter=100)
        metrics, errors = pose_metrics(poses, gt)
        result[name] = {'predicted_c2w':poses.tolist(), 'gt_c2w':gt.tolist(),
            'metrics':metrics, 'relative_errors':errors, 'pair_count':45,
            'estimated_focal':float(focal), 'pnp_failed_view_indices':failures,
            'confidence_shapes':[list(p['conf'].shape) for p in values]}
    return result


def validate(report, proto, inputs, gt):
    if (report['protocol'] != proto or report['protocol_sha256'] != signature(proto)
        or not report['paired_same_forward'] or report['forward_count'] != 1
        or report['formal_pose_metrics_available'] or report['full_paper_completed']):
        raise ValueError('Invalid diagnostic provenance or formal claim')
    if set(report['branches']) != set(BRANCHES): raise ValueError('Branch set changed')
    for name, row in report['branches'].items():
        validate_row(row, inputs['frames'][0]['label'],signature(proto),inputs)
        if not np.array_equal(np.asarray(row['gt_c2w'],np.float32),gt):
            raise ValueError('Saved GT changed')
        expected = [1,384,512] if name == BRANCHES[0] else [1,512,384]
        if row['confidence_shapes'] != [expected]*10: raise ValueError('Shape contract changed')
        if not np.isfinite(row['estimated_focal']) or row['estimated_focal'] <= 0:
            raise ValueError('Invalid predicted focal')


def run(args):
    torch.set_num_threads(2)
    archived, views = load_archived_draw()
    inputs = {'frames':archived['returned_views']}
    gt = np.stack([v['camera_pose'] for v in views]).astype(np.float32)
    seed = 42 + int(archived['scope']['base_index'])
    proto = protocol(Path('checkpoints/Fast3R_ViT_Large_512'),seed,2)
    proto.update({'diagnostic_runner_sha256':sha(__file__),
        'historical_baseline_sha256':sha(BASELINE),
        'benchmark_scope':'same_forward_one_draw_orientation_diagnostic_not_Table1',
        'changed_variable':'released landscape correction vs raw portrait only; focal is recomputed from each branch',
        'branches':list(BRANCHES),'GT_for_focal_or_PnP':False})
    if args.verify_only:
        validate(json.loads(OUTPUT.read_text()),proto,inputs,gt)
        print('SAME-FORWARD ORIENTATION DIAGNOSTIC VERIFIED; no inference rerun');return
    if OUTPUT.exists(): raise FileExistsError('Historical output preserved; use verify-only')
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True).splitlines()[0])
    if free < 10240: raise RuntimeError('GPU busy; queue on next heartbeat, do not compete')
    if shutil.disk_usage('results').free < 1024**3+1024**2: raise RuntimeError('Preserve1GiB')
    from fast3r.models.fast3r import Fast3R
    from fast3r.dust3r.inference_multiview import inference
    if not torch.cuda.is_available(): raise RuntimeError('Host CUDA unavailable')
    model=Fast3R.from_pretrained('checkpoints/Fast3R_ViT_Large_512').cuda().eval()
    model.set_max_parallel_views_for_head(2)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    model_views = model_inputs(views)
    with torch.inference_mode():
        output=inference(model_views,model,torch.device('cuda'),dtype='16-mixed',verbose=False)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        preds=output['preds']
        hashes=[{k:hashlib.sha256(p[k].contiguous().numpy().tobytes()).hexdigest()
                 for k in ('pts3d_in_other_view','conf')} for p in preds]
        branches=evaluate_branches(preds,model_views,seed,gt)
        assert hashes == [{k:hashlib.sha256(p[k].contiguous().numpy().tobytes()).hexdigest()
                 for k in ('pts3d_in_other_view','conf')} for p in preds]
    for row in branches.values():
        row.update({'scene':inputs['frames'][0]['label'],'inputs':inputs,'protocol_sha256':signature(proto)})
    old=json.loads(BASELINE.read_text())
    result={'status':'same_forward_orientation_diagnostic_complete_not_Table1',
        'protocol':proto,'protocol_sha256':signature(proto),'branches':branches,
        'paired_same_forward':True,'forward_count':1,'global_tensor_sha256':hashes,
        'historical_baseline_metrics_exact_match':branches[BRANCHES[0]]['metrics']==old['metrics'],
        'historical_baseline_metric_deltas':{k:branches[BRANCHES[0]]['metrics'][k]-old['metrics'][k]
            for k in old['metrics']},
        'formal_pose_metrics_available':False,'full_paper_completed':False,
        'runtime':{'forward_seconds':seconds,'gpu':torch.cuda.get_device_name(),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated()},
        'note':'One fixed draw, same forward/points/GT/seed/PnP/metric. Direction change also changes focal pixel axes, so it is not an isolated fixed-focal experiment. Raw branch is diagnostic only; published baseline is retained. Does not establish paper equivalence or all-scene causality.'}
    validate(result,proto,inputs,gt);json.dumps(result,allow_nan=False);save_new(OUTPUT,result)
    print(json.dumps({name:{k:r[k] for k in ('metrics','estimated_focal','pnp_failed_view_indices')}
        for name,r in branches.items()},indent=2))
    print('DIAGNOSTIC COMPLETE: one forward, no full Table1 claim',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-only',action='store_true')
    run(p.parse_args())
