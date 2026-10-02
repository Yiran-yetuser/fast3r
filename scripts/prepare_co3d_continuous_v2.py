"""Real candidate 100@ sequential preparation, not Table1 inference/scores.

Single worker, original pool and shared invalidity/oversampling states. Existing
lazy cache bounds stay unchanged. Error leaves only earlier completed commits.
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from co3d_lazy_dataset_v2 import LazyPreparer, StrictLazyCo3d
from co3d_sampling_state import capture, restore
from co3d_request_journal import atomic_new, read_prefix, commit
from fast3r.dust3r.datasets.base.easy_dataset import ResizedDataset
from fast3r_hf_re10k_pose_eval import sha

ROOT=Path('results/co3d_continuous_prepare_v2_20261002')
CANDIDATE=Path('data/co3d_test_metadata/selected_seqs_test_seen41_candidate.json')


def inputs(views):
    return [{'label':v['label'],'instance':v['instance'],
        'img_shape':list(v['img'].shape),'true_shape_hw':v['true_shape'].tolist(),
        'rng':v['rng'],'camera_pose':v['camera_pose'].tolist(),
        'camera_intrinsics':v['camera_intrinsics'].tolist(),
        'img_tensor_sha256':__import__('hashlib').sha256(v['img'].contiguous().numpy().tobytes()).hexdigest()}
        for v in views]


def validate_result(row, request):
    if row['request']!=request or len(row['returned_views'])!=10 or row['model_forward_count']!=0:
        raise ValueError('Invalid preparation result')
    for v in row['returned_views']:
        if (np.asarray(v['camera_pose']).shape!=(4,4)
            or not np.isfinite(np.asarray(v['camera_pose'])).all()
            or len(v['img_tensor_sha256'])!=64):raise ValueError('Invalid input geometry/hash')
    if any(t['status']=='hard_error' for t in row['load_trace']):raise ValueError('Hard error committed')
    for frame in row['prepared_frames']:
        for name,h in frame['processed_sha256'].items():
            if sha(Path('data/co3d_lazy_v2_processed')/name)!=h:raise ValueError('Prepared bytes changed')


def verify_v1_prefix(row, after, request):
    # Independent v2 replay, not identity rewriting of frozen v1 transactions.
    old_root = Path('results/co3d_continuous_prepare_20261002')
    old_path = old_root/f'request_{request:03d}.json'
    if request >= 3: return
    evidence = json.loads(Path('results/co3d_continuous_prefix3_verified_20261002.json').read_text())
    if sha(old_path) != evidence['request_sha256'][old_path.name]:
        raise ValueError('Verified frozen V1 transaction changed')
    old = json.loads(old_path.read_text())
    for key in ('request', 'base_index', 'returned_views', 'load_trace', 'pool_attempts', 'model_forward_count'):
        if old['result'][key] != row[key]: raise ValueError('V1 prefix inputs/trace differ: '+key)
    for key in after:
        if key != 'identity' and old['after_state'][key] != after[key]:
            raise ValueError('V1 prefix shared sampling state differs: '+key)
    # Metadata ZIP timestamps can change NPZ bytes; compare actual arrays.
    for frame in row['prepared_frames']:
        previous = next(f for f in old['result']['prepared_frames'] if f['image_path'] == frame['image_path'])
        if previous['raw_members'] != frame['raw_members']: raise ValueError('Raw identities differ')
        for name in frame['processed_sha256']:
            left, right = Path('data/co3d_lazy_processed')/name, Path('data/co3d_lazy_v2_processed')/name
            if name.endswith('.npz'):
                with np.load(left,allow_pickle=False) as a, np.load(right,allow_pickle=False) as b:
                    if set(a.files)!=set(b.files) or any(a[k].dtype!=b[k].dtype or not np.array_equal(a[k],b[k]) for k in a.files):
                        raise ValueError('V1 prefix NPZ differs')
            elif sha(left)!=sha(right): raise ValueError('V1 prefix processed modality differs')


def run(args):
    torch.set_num_threads(2)
    selected=json.loads(CANDIDATE.read_text());preparer=LazyPreparer()
    proof_path = Path('results/co3d_inf_reference_v2_offline_verified_20261002.json')
    proof = json.loads(proof_path.read_text())
    if (proof['v2_fingerprint'] != preparer.fingerprint
        or proof['verification_code_sha256'] != sha('scripts/verify_co3d_inf_reference_v2.py')
        or not all(proof[k] for k in ('image_depth_mask_bytes_equal', 'npz_arrays_and_dtypes_equal',
             'original_loader_returns_none_and_invalidates', 'strict_loader_returns_none_and_invalidates'))):
        raise ValueError('Real Inf-frame proof is absent or stale')
    dataset=StrictLazyCo3d(selected,preparer,42,777)
    wrapper=ResizedDataset(100,dataset);wrapper.set_epoch(0)
    mapping=[int(x) for x in wrapper._idxs_mapping]
    bindings={'candidate_sha256':sha(CANDIDATE),'preparer':preparer.fingerprint,
        'code_sha256':{p:sha(p) for p in (__file__,'scripts/co3d_sampling_state.py',
            'scripts/co3d_request_journal.py','fast3r/dust3r/datasets/base/easy_dataset.py',
            'fast3r/dust3r/datasets/base/base_stereo_view_dataset.py')},
        'workers':0,'epoch':0,'combination_seed':42,'dataset_seed':777,
        'initial_python_seed':42+mapping[0], 'raw_cache_bound_bytes':2*1024**3,
        'processed_cache_bound_bytes':512*1024**2,'reserve_bytes':1024**3,
        'scope':'actual_100_candidate_requests_preparation_not_author_split_or_Table1'}
    bindings['inf_reference_proof_sha256'] = sha(proof_path)
    bindings['v1_prefix_proof_sha256'] = sha('results/co3d_continuous_prefix3_verified_20261002.json')
    random.seed(bindings['initial_python_seed']);initial=capture(dataset,mapping,0,bindings)
    initial_path=ROOT/'initial.json'
    if initial_path.exists():
        if initial!=json.loads(initial_path.read_text()):raise ValueError('Initial identity changed')
    elif args.verify_only:raise FileNotFoundError(initial_path)
    else:atomic_new(initial_path,initial)
    rows,state=read_prefix(ROOT,initial,validate_result)
    for envelope in rows:
        verify_v1_prefix(envelope['result'], envelope['after_state'], envelope['request'])
    cursor=restore(dataset,mapping,state,bindings)
    if args.verify_only:
        print(f'VERIFIED {cursor}/100 actual prepared requests; no download/inference');return
    for request in range(cursor,min(args.stop_after,100)):
        before=capture(dataset,mapping,request,bindings)
        dataset.trace=[];dataset.pool_attempts=[];preparer.prepared_records={}
        dataset.active_request=request
        views=wrapper[request]
        row={'request':request,'base_index':mapping[request],'returned_views':inputs(views),
            'load_trace':dataset.trace,'pool_attempts':dataset.pool_attempts,
            'prepared_frames':list(preparer.prepared_records.values()),'model_forward_count':0}
        validate_result(row,request)
        after=capture(dataset,mapping,request+1,bindings)
        verify_v1_prefix(row, after, request)
        commit(ROOT,request,before,row,after)
        print(f'COMMITTED real request {request+1}/100 base={mapping[request]} '+
              f"scene={views[0]['label']} unique={len({v['instance'] for v in views})}",flush=True)
    print(f'PREPARATION boundary {min(args.stop_after,100)}/100; NOT pose scores',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stop-after',type=int,default=100)
    p.add_argument('--verify-only',action='store_true');args=p.parse_args()
    if not 1<=args.stop_after<=100:p.error('stop-after must be 1..100')
    run(args)
