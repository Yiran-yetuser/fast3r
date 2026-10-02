"""Replay committed actual requests read-only, retaining shared sampler state."""
import argparse
import json
import random
from pathlib import Path
from co3d_lazy_dataset_v4 import StrictLazyCo3d
from co3d_sampling_state import capture, digest
from co3d_request_journal import read_prefix, atomic_new
from prepare_co3d_continuous_v4 import ROOT, CANDIDATE, inputs, validate_result
from fast3r.dust3r.datasets.base.easy_dataset import ResizedDataset
from fast3r_hf_re10k_pose_eval import sha


class ReadOnlyPrefix:
    root=Path('data/co3d_lazy_v4_processed')
    def __init__(self,rows):
        self.frames={f['image_path']:f for row in rows for f in row['result']['prepared_frames']}
    def ensure(self,category,scene,frame):
        f=self.frames[f'{category}/{scene}/images/frame{frame:06d}.jpg']
        for name,h in f['processed_sha256'].items():
            if sha(self.root/name)!=h:raise ValueError('Prepared bytes changed')
        for m in f['raw_members']:
            path=Path('data/co3d_range_cache/raw')/m['path']
            if path.stat().st_size!=m['bytes'] or sha(path)!=m['sha256']:
                raise ValueError('Raw member bytes changed')
        return f


def verify():
    initial=json.loads((ROOT/'initial.json').read_text());bindings=initial['identity']['bindings']
    for p,h in bindings['code_sha256'].items():
        if sha(p)!=h:raise ValueError('Bound code changed')
    for name,key in [('scripts/co3d_lazy_dataset_v4.py','lazy_code_sha256'),
        ('scripts/co3d_lazy_dataset_v3.py','frozen_v3_sha256'),
        ('scripts/co3d_lazy_dataset_v2.py','frozen_v2_sha256'),
        ('scripts/co3d_lazy_dataset.py','frozen_v1_sha256')]:
        if sha(name)!=bindings['preparer'][key]:raise ValueError('Frozen preparer lineage changed')
    if sha(CANDIDATE)!=bindings['candidate_sha256']:raise ValueError('Candidate changed')
    rows,final=read_prefix(ROOT,initial,validate_result)
    d=StrictLazyCo3d(json.loads(CANDIDATE.read_text()),ReadOnlyPrefix(rows),42,777)
    w=ResizedDataset(100,d);w.set_epoch(0);mapping=[int(x) for x in w._idxs_mapping]
    random.seed(bindings['initial_python_seed'])
    if capture(d,mapping,0,bindings)!=initial:raise ValueError('Initial sampler changed')
    for i,row in enumerate(rows):
        if digest(capture(d,mapping,i,bindings))!=row['before_state_sha256']:
            raise ValueError('Before-state mismatch')
        d.trace=[];d.pool_attempts=[];d.active_request=i
        views=w[i]
        if inputs(views)!=row['result']['returned_views'] or d.trace!=row['result']['load_trace'] or d.pool_attempts!=row['result']['pool_attempts']:
            raise ValueError('Actual input/trace replay diverged')
        if capture(d,mapping,i+1,bindings)!=row['after_state']:
            raise ValueError('Shared after-state replay diverged')
    return {'status':'actual_continuous_prefix_readonly_replay_verified_not_pose_scores',
        'verified_request_count':len(rows),'expected_request_count':100,
        'initial_sha256':sha(ROOT/'initial.json'),'final_state_sha256':digest(final),
        'request_sha256':{f'request_{i:03d}.json':sha(ROOT/f'request_{i:03d}.json') for i in range(len(rows))},
        'input_tensors_GT_rng_and_shared_state_exact_equal':True,'network_bytes':0,
        'verifier_sha256':sha(__file__),'model_forward_count':0,
        'all_100_requests_prepared':len(rows)==100,'author_split_equivalence_verified':False,
        'formal_pose_metrics_available':False,'full_paper_completed':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path)
    args=p.parse_args();r=verify()
    if args.output:atomic_new(args.output,r)
    print(json.dumps(r,indent=2))
