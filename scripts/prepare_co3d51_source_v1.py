#!/usr/bin/env python3
"""Real source51 1000@ data preparation, no model/PnP or Table1 score.

Original candidate frame pools; source jitter/oversampling/5-scene retry;
single-worker shared-state immutable request commits; new declared landscape
crop. Historical100 inputs and old cache ceilings remain unchanged.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from co3d51_source_lazy_v1 import (Source51Preparer,Source51Landscape,MANIFEST,
    RAW_ROOT,PROCESSED_ROOT,RAW_LIMIT,PROCESSED_LIMIT,RESERVE,file_bytes)
from co3d_request_journal import atomic_new,read_prefix,commit
from co3d_sampling_state import capture,restore,digest
from fast3r.dust3r.datasets.base.easy_dataset import ResizedDataset
from prepare_co3d_continuous_v4 import inputs
from prepare_re10k_rgb_from_archive import sha,save_identical

ROOT=Path('results/co3d51_source_prepare_v1_progress')
SUMMARY=Path('results/co3d51_source_prepare_v1_summary_20261003.json')
JOURNAL_LIMIT=4*1024**3
REQUESTS=1000


def journal_budget(additional=0):
    if file_bytes(ROOT)+additional>JOURNAL_LIMIT or shutil.disk_usage('.').free<RESERVE+additional:
        raise RuntimeError('New request journal4GiB or1GiB reserve; preserve completed requests')


def validate_result(row, request):
    if (row['request']!=request or len(row['returned_views'])!=10 or row['model_forward_count']!=0
            or row['formal_Table1_result'] or any(t['status']=='hard_error' for t in row['load_trace'])):
        raise ValueError('Invalid actual data transaction/scope')
    for v in row['returned_views']:
        if (v['img_shape']!=[3,384,512] or v['true_shape_hw']!=[384,512]
                or np.asarray(v['camera_pose']).shape!=(4,4)
                or not np.isfinite(np.asarray(v['camera_pose'])).all()
                or np.asarray(v['camera_intrinsics']).shape!=(3,3)
                or not np.isfinite(np.asarray(v['camera_intrinsics'])).all()
                or np.linalg.det(v['camera_intrinsics'])<=0 or len(v['img_tensor_sha256'])!=64):
            raise ValueError('Actual input landscape/GT/hash invalid')
    for frame in row['prepared_frames']:
        for name,h in frame['processed_sha256'].items():
            if sha(PROCESSED_ROOT/name)!=h:raise ValueError('Prepared frame bytes changed')


def make_context():
    torch.set_num_threads(2)
    selected=json.loads(MANIFEST.read_text());preparer=Source51Preparer()
    dataset=Source51Landscape(selected,preparer,42,777)
    wrapper=ResizedDataset(REQUESTS,dataset);wrapper.set_epoch(0)
    mapping=[int(x) for x in wrapper._idxs_mapping]
    plan_path=Path('data/co3d_test_metadata/released_nominal_51_1000_seed777_v1.json')
    plan=json.loads(plan_path.read_text())
    if mapping!=plan['mapping'] or list(dataset.combinations[0])!=plan['combination_actually_used']:
        raise ValueError('Released source mapping/combinations differ from verified nominal plan')
    files=[__file__,'scripts/co3d51_source_lazy_v1.py','scripts/co3d_sampling_state.py',
        'scripts/co3d_request_journal.py','scripts/audit_co3d_portrait_geometry.py',
        'scripts/co3d_lazy_dataset_v4.py','scripts/co3d_lazy_dataset_v3.py',
        'scripts/probe_co3d_preprocess.py','scripts/co3d_range_cache.py',
        'fast3r/dust3r/datasets/co3d_multiview.py',
        'fast3r/dust3r/datasets/base/easy_dataset.py',
        'fast3r/dust3r/datasets/base/base_stereo_view_dataset.py']
    bindings={'scope':'actual_source51_1000_landscape_candidate_preparation_not_Table1',
        'candidate_sha256':sha(MANIFEST),'nominal_plan_sha256':sha(plan_path),
        'preparer':preparer.fingerprint,'code_sha256':{p:sha(p) for p in files},
        'workers':0,'epoch':0,'combination_seed':42,'dataset_seed':777,
        'initial_python_seed':42+mapping[0],'raw_cache_limit_bytes':RAW_LIMIT,
        'processed_limit_bytes':PROCESSED_LIMIT,'journal_limit_bytes':JOURNAL_LIMIT,
        'reserve_bytes':RESERVE,'author_rng_recovered':False,'formal_Table1_result':False,
        'crop':'AST guarded fixed landscape512x384; source portrait/square reversal removed',
        'IO_error_policy':'hard failure, no silent resampling; real zero masked depth keeps source invalidation/oversampling'}
    random.seed(bindings['initial_python_seed'])
    return dataset,wrapper,mapping,bindings


def run(args):
    ROOT.mkdir(parents=True,exist_ok=True)
    d,w,mapping,bindings=make_context()
    initial=capture(d,mapping,0,bindings);initial_path=ROOT/'initial.json'
    if args.dry_run:
        remaining=max(0,RAW_LIMIT-file_bytes(RAW_ROOT))+max(0,PROCESSED_LIMIT-file_bytes(PROCESSED_ROOT))
        remaining+=max(0,JOURNAL_LIMIT-file_bytes(ROOT))+RESERVE
        free=shutil.disk_usage('.').free
        if free<remaining:raise RuntimeError('Declared new-version full allocation envelope does not fit; no downloads')
        print(json.dumps({'status':'source51_1000_preparation_dry_run_not_data_ready',
            'raw_limit_bytes':RAW_LIMIT,'processed_limit_bytes':PROCESSED_LIMIT,
            'journal_limit_bytes':JOURNAL_LIMIT,'free_bytes':free,
            'remaining_declared_allocation_plus_reserve_bytes':remaining,
            'historical_caches_frozen':True,'model_forward_count':0,'formal_Table1_result':False},indent=2))
        return
    if initial_path.exists():
        if json.loads(initial_path.read_text())!=initial:raise ValueError('Initial state identity changed')
    elif args.verify_only:raise FileNotFoundError(initial_path)
    else:journal_budget(16*1024**2);atomic_new(initial_path,initial)
    rows,state=read_prefix(ROOT,initial,validate_result)
    cursor=restore(d,mapping,state,bindings)
    if args.verify_only:
        print(f'VERIFIED {cursor}/{REQUESTS} actual prepared transactions; no download/model');return
    for request in range(cursor,min(args.stop_after,REQUESTS)):
        journal_budget(16*1024**2)
        before=capture(d,mapping,request,bindings)
        d.trace=[];d.pool_attempts=[];d.preparer.prepared_records={};d.active_request=request
        views=w[request]
        row={'request':request,'base_index':mapping[request],'returned_views':inputs(views),
            'load_trace':d.trace,'pool_attempts':d.pool_attempts,
            'prepared_frames':list(d.preparer.prepared_records.values()),
            'model_forward_count':0,'formal_Table1_result':False}
        validate_result(row,request)
        after=capture(d,mapping,request+1,bindings)
        journal_budget(len(json.dumps(after).encode())*2+len(json.dumps(row).encode())*2)
        commit(ROOT,request,before,row,after)
        print(f'COMMITTED source51 real request {request+1}/{REQUESTS} '+
              f"scene={views[0]['label']} unique={len({v['instance'] for v in views})}",flush=True)
        # Actual files remain; only refresh in-memory accounting at boundaries.
        if (request+1)%25==0:
            d.preparer.used_bytes=file_bytes(PROCESSED_ROOT)
            d.preparer.cache.used_bytes=file_bytes(RAW_ROOT)
            d.preparer.budget();d.preparer.cache.budget()
    if min(args.stop_after,REQUESTS)==REQUESTS:
        complete,final=read_prefix(ROOT,initial,validate_result)
        if len(complete)!=REQUESTS:raise ValueError('Incomplete full preparation')
        summary={'status':'actual_source51_1000_landscape_candidate_prepared_not_pose_scores',
            'paper_mapping':'section4.2/Table1 public-HF candidate input prerequisites only',
            'request_count':len(complete),'initial_sha256':sha(initial_path),
            'final_sampling_state_sha256':digest(final),'bindings_sha256':digest(bindings),
            'request_sha256':{f'request_{i:03d}.json':sha(ROOT/f'request_{i:03d}.json') for i in range(REQUESTS)},
            'raw_new_version_used_bytes':file_bytes(RAW_ROOT),
            'processed_new_version_used_bytes':file_bytes(PROCESSED_ROOT),
            'request_journal_used_bytes':file_bytes(ROOT),'model_forward_count':0,
            'GT_not_used_by_model':True,'author_protocol_equivalence_verified':False,
            'readonly_input_state_replay_still_required_before_GPU':True,
            'formal_Table1_result':False,'full_paper_completed':False}
        save_identical(SUMMARY,summary)
        print('ALL1000 actual candidate requests prepared; independent read-only replay required, no pose score',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stop-after',type=int,default=REQUESTS)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    if not 1<=args.stop_after<=REQUESTS:p.error('stop-after must be1..1000')
    run(args)
