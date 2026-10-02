#!/usr/bin/env python3
"""Reproduce one candidate request and record the released focal estimator output."""
import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from co3d_sampling_state import restore
from fast3r.dust3r.datasets.base.easy_dataset import ResizedDataset
from fast3r_hf_co3d_100_pose_eval import (CHECKPOINT, PREP_ROOT, build_dataset,
    checked_prefix, model_inputs, signature, tensor_sha)
from fast3r_hf_co3d_pose_smoke import correct_orientation
from prepare_co3d_continuous_v4 import inputs
from fast3r_hf_re10k_pose_eval import sha


def finite_stats(tensor):
    flat=tensor.detach().float().reshape(-1)
    finite=flat[torch.isfinite(flat)]
    if finite.numel()==0:
        return {'count':int(flat.numel()),'finite_count':0}
    qs=torch.quantile(finite,torch.tensor([0.,.01,.1,.5,.9,.99,1.],device=finite.device))
    return {'count':int(flat.numel()),'finite_count':int(finite.numel()),
        'nan_count':int(torch.isnan(flat).sum()),'posinf_count':int(torch.isposinf(flat).sum()),
        'neginf_count':int(torch.isneginf(flat).sum()),'quantiles_0_1_10_50_90_99_100':qs.cpu().tolist()}


def run(index, output_path):
    torch.set_num_threads(2)
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    if free<10240:raise RuntimeError('GPU free memory below 10240 MiB; do not compete')
    initial,envelopes,proof=checked_prefix()
    dataset,wrapper,mapping,bindings=build_dataset(initial,envelopes)
    before=initial if index==0 else json.loads((PREP_ROOT/f'request_{index-1:03d}.json').read_text())['after_state']
    source=json.loads((PREP_ROOT/f'request_{index:03d}.json').read_text())
    restore(dataset,mapping,before,bindings)
    if index!=before['next_request']:raise ValueError('Request cursor mismatch')
    dataset.trace=[];dataset.pool_attempts=[];dataset.active_request=index
    views=wrapper[index]
    if inputs(views)!=source['result']['returned_views'] or dataset.trace!=source['result']['load_trace']:
        raise ValueError('Diagnostic request differs from verified fixed input')
    network_views=model_inputs(views)
    request_seed=(42+index)%(2**32)
    from fast3r.models.fast3r import Fast3R
    from fast3r.dust3r.inference_multiview import inference
    from fast3r.models.multiview_dust3r_module import estimate_focal
    model=Fast3R.from_pretrained(str(CHECKPOINT)).cuda().eval()
    model.set_max_parallel_views_for_head(2)
    random.seed(request_seed);np.random.seed(request_seed);torch.manual_seed(request_seed);torch.cuda.manual_seed_all(request_seed)
    torch.cuda.synchronize();started=time.perf_counter()
    with torch.inference_mode():
        output=inference(network_views,model,torch.device('cuda'),dtype='16-mixed',verbose=False)
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        corrected=correct_orientation(output['preds'],network_views)
        points=corrected[0]['pts3d_in_other_view'].float()
        confidence=corrected[0]['conf'].float()
        focal=estimate_focal(points,confidence,min_conf_thr_percentile=10)
        conf_thr=torch.quantile(confidence.reshape(-1),.10)
        result={'status':'candidate_request_focal_diagnostic_not_pose_metrics',
            'request_index':index,'base_index':source['result']['base_index'],
            'source_request_sha256':sha(PREP_ROOT/f'request_{index:03d}.json'),
            'prefix_proof_sha256':sha('results/co3d_v4_prefix_full_20261003.json'),
            'request_seed':request_seed,'input_tensor_sha256':[tensor_sha(v['img']) for v in views],
            'input_labels':[[v['label'],v['instance'],v['rng']] for v in views],
            'input_shapes':[list(v['img'].shape) for v in views],
            'raw_prediction_shape':list(points.shape),'confidence_shape':list(confidence.shape),
            'points_stats':finite_stats(points),'confidence_stats':finite_stats(confidence),
            'focal_confidence_threshold_p10':float(conf_thr),'focal_value':float(focal) if np.isfinite(focal) else str(focal),
            'focal_finite_positive':bool(np.isfinite(focal) and focal>0),
            'network_input_keys':['img','true_shape'],'GT_used_by_network_or_focal':False,
            'forward_seconds':seconds,'model_forward_count':1,'network_bytes':0,
            'checkpoint_weight_sha256':sha(CHECKPOINT/'model.safetensors'),
            'checkpoint_config_sha256':sha(CHECKPOINT/'config.json'),
            'protocol_signature':signature({'candidate_sha256':sha('data/co3d_test_metadata/selected_seqs_test_seen41_candidate.json'),
                'request_seed':request_seed,'head_chunk_size':2,'precision':'16-mixed'})}
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','request_index','focal_value','focal_finite_positive','points_stats','confidence_stats','forward_seconds')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request-index',type=int,default=3)
    p.add_argument('--output',type=Path,default=Path('results/co3d_candidate_request003_focal_diagnostic_20261003.json'))
    a=p.parse_args()
    if not 0<=a.request_index<100:p.error('request index must be 0..99')
    run(a.request_index,a.output)
