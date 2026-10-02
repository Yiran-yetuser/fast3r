#!/usr/bin/env python3
"""One REAL 100@ epoch0 mapped draw, not a 100-draw benchmark or pose score."""
import argparse
import fcntl
import json
import random
from pathlib import Path

import numpy as np
import torch

from co3d_lazy_dataset import LazyPreparer, StrictLazyCo3d, MAX_PROCESSED
from fast3r.dust3r.datasets.base.easy_dataset import ResizedDataset
from prepare_re10k_rgb_from_archive import sha, save_identical


def same_views(left, right):
    if len(left) != len(right): raise ValueError('Different view count')
    for a, b in zip(left, right):
        if set(a) != set(b): raise ValueError('Different returned keys')
        for key in a:
            x,y = a[key], b[key]
            equal = torch.equal(x,y) if isinstance(x,torch.Tensor) else (
                np.array_equal(x,y) if isinstance(x,np.ndarray) else x==y)
            if not equal: raise ValueError(f'Original/lazy view mismatch: {key}')


def json_index(view):
    return [int(x) for x in view['idx']]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wrapper-index',type=int,default=0)
    args=parser.parse_args()
    if not 0 <= args.wrapper_index < 100: raise ValueError('Outside 100@ length')
    torch.set_num_threads(2)
    prep=LazyPreparer(); lock=(prep.root/'.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dataset=StrictLazyCo3d(prep.cache.selected,prep)
    wrapper=ResizedDataset(100,dataset); wrapper.set_epoch(0)
    base_index=int(wrapper._idxs_mapping[args.wrapper_index])
    dataset.active_request={'wrapper_index':args.wrapper_index,'base_index':base_index}
    # Python oversampling RNG is separate from the base-index-seeded NumPy RNG.
    # Reset to a declared per-request state so restart order cannot alter choices.
    request_seed=42+base_index
    random.seed(request_seed)
    views=wrapper[args.wrapper_index]
    trace=dataset.trace.copy(); attempts=dataset.pool_attempts.copy()
    invalid={f'{c}/{s}':{str(res):[i for i,b in enumerate(bits) if b] for res,bits in by_res.items()}
             for (c,s),by_res in dataset.invalidate.items() if by_res}
    # Independent original class on the exact prepared files/full candidate pool.
    # It can only succeed if every actual attempted frame above was prepared.
    from fast3r.dust3r.datasets.co3d_multiview import Co3d_Multiview
    from fast3r.dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
    ref=Co3d_Multiview.__new__(Co3d_Multiview)
    BaseStereoViewDataset.__init__(ref,split='test',resolution=(512,384),seed=777)
    ref.num_views=10;ref.mask_bg=True;ref.ROOT=str(prep.root)
    ref.scenes=dataset.scenes;ref.scene_list=dataset.scene_list;ref.combinations=dataset.combinations
    ref.invalidate={s:{} for s in ref.scene_list};ref.invalid_scene_tracker=set()
    random.seed(request_seed); reference=ref[base_index]
    same_views(views,reference)
    # v1 interrupted on NumPy index JSON serialization. Preserve that partial
    # evidence, use a fresh result path; prepared data/cache identities unchanged.
    report_path=Path(f'results/co3d_lazy_draw{args.wrapper_index}_v2_20261002.json')
    report={'status':'one_real_mapped_draw_verified_not_full_benchmark',
        'paper_mapping':'section 4.2 / Table 1 lazy data and realized sampling prerequisites',
        'scope':{'wrapper_length':100,'executed_wrapper_indices':[args.wrapper_index],
            'base_index':base_index,'candidate_categories':len(prep.cache.selected),
            'candidate_trajectories':len(dataset.scene_list),
            'candidate_frames':sum(len(f) for f in dataset.scenes.values()),
            'base_dataset_length':len(dataset),'combination_count':len(dataset.combinations),
            'combination_actually_used':list(dataset.combinations[0]),
            'combination_seed':42,'dataset_seed':777,'epoch':0,'python_request_seed':request_seed,
            'sampling_equivalence':'declared audit seeds and per-request oversampling seed; paper draw identity unknown'},
        'fingerprint':{'preparer':prep.fingerprint,'probe_sha256':sha(Path(__file__)),
            'released_loader_sha256':sha(Path('fast3r/dust3r/datasets/co3d_multiview.py')),
            'base_loader_sha256':sha(Path('fast3r/dust3r/datasets/base/base_stereo_view_dataset.py')),
            'wrapper_sha256':sha(Path('fast3r/dust3r/datasets/base/easy_dataset.py'))},
        'pool_attempts':attempts,'actual_load_trace':trace,'invalidated_pool_indices':invalid,
        'invalid_scene_tracker':[list(s) for s in sorted(dataset.invalid_scene_tracker)],
        'returned_views':[{'label':v['label'],'instance':v['instance'],'idx':json_index(v),
            'true_shape_hw':v['true_shape'].tolist(),'img_shape':list(v['img'].shape),
            'gt_valid_fraction':float(v['valid_mask'].mean()),'rng':v['rng'],
            'camera_pose':v['camera_pose'].tolist(),'camera_intrinsics':v['camera_intrinsics'].tolist()}
            for v in views],
        'prepared_frames':[prep.prepared_records[p] for p in sorted(prep.prepared_records)],
        'original_loader_base_outputs_exactly_equal':True,
        'network_bytes_first_successful_draw':prep.cache.network_bytes,
        'limits':{'raw_cache_bytes':2*1024**3,'processed_cache_bytes':MAX_PROCESSED,'reserve_bytes':1024**3},
        'actual_depth_validity_verified_for_this_draw':True,
        'formal_pose_metrics_available':False,'full_candidate_data_ready':False,
        'all_100_draws_executed':False,'source_zip_full_sha_verified':False,
        'paper_selection_and_sampling_equivalence_verified':False,
        'note':'No model inference. Equality includes normalized tensors, points, valid masks, K, poses, rng and landscape transposition. Duplicate views retained. Hard errors propagate; zero-depth sampling remains original. Only this one draw is validated. First download attempt ended on NumPy-index serialization; partial v1 report retained; this successful v2 run reuses verified cache.'}
    if report_path.exists():
        report['network_bytes_first_successful_draw']=json.loads(report_path.read_text())['network_bytes_first_successful_draw']
    save_identical(report_path,report)
    print('ACTUAL DRAW',args.wrapper_index,'base',base_index,flush=True)
    print('Labels/frames',[(v['label'],v['instance']) for v in views],flush=True)
    print('Load attempts',len(trace),'prepared unique frames',len(prep.prepared_records),'pool attempts',len(attempts),flush=True)
    print('Network bytes THIS process',prep.cache.network_bytes,flush=True)
    print('Original/lazy full base outputs EXACTLY EQUAL; no model metrics',flush=True)


if __name__=='__main__':main()
