"""Offline independent replay of source51 actual input transactions.

Never instantiates a downloader/preprocessor/model, never writes datasets.
Replays original sampler/invalidity/RNG with the declared landscape crop.
A verified prefix is explicitly not all1000, author equivalence or pose scores.
"""
import argparse
import json
import random
from pathlib import Path, PurePosixPath

import torch

from co3d51_source_lazy_v1 import (Source51Landscape,MANIFEST,PROCESSED_ROOT,
    RAW_ROOT,STORAGE,PROOF,RAW_LIMIT,PROCESSED_LIMIT,RESERVE,checked_payload)
from co3d_sampling_state import capture,digest
from co3d_request_journal import atomic_new,read_prefix
from prepare_co3d51_source_v1 import ROOT,SUMMARY,REQUESTS,JOURNAL_LIMIT,validate_result
from prepare_co3d_continuous_v4 import inputs
from prepare_re10k_rgb_from_archive import sha
from fast3r.dust3r.datasets.base.easy_dataset import ResizedDataset


def safe_relative(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or len(p.parts)<3:
        raise ValueError('Unsafe input receipt path')
    return name


class ReadOnlyInputs:
    root=PROCESSED_ROOT
    def __init__(self,rows,fingerprint):
        self.frames={};self.fingerprint=fingerprint;self.prepared_records={}
        for row in rows:
            for f in row['result']['prepared_frames']:
                key=safe_relative(f['image_path'])
                if key in self.frames and self.frames[key]!=f:
                    raise ValueError('Conflicting prepared frame receipts')
                self.frames[key]=f

    def ensure(self,category,scene,frame):
        key=f'{category}/{scene}/images/frame{frame:06d}.jpg'
        f=self.frames[key]
        if f['fingerprint']!=self.fingerprint:raise ValueError('Frame identity changed')
        for name,h in f['processed_sha256'].items():
            if sha(self.root/safe_relative(name))!=h:raise ValueError('Prepared bytes changed')
        record_path=self.root/(key+'.prepared.json')
        if json.loads(record_path.read_text())!=f:raise ValueError('Prepared receipt changed')
        for m in f['raw_members']:
            name=safe_relative(m['path'])
            receipt=json.loads((RAW_ROOT/'records'/(name+'.json')).read_text())
            if (receipt['fingerprint']!=self.fingerprint['cache'] or not receipt['member_crc_verified']
                    or receipt['full_archive_sha_verified'] or not m['member_crc_verified']
                    or m['full_archive_sha_verified'] or receipt['entry']['file_size']!=m['bytes']
                    or receipt['sha256']!=m['sha256']):
                raise ValueError('Raw identity/CRC claim differs')
            if receipt['storage']=='downloaded_raw':path=RAW_ROOT/'raw'/name
            elif receipt['storage']=='borrowed_frozen_raw':
                path=Path('data/co3d_range_cache/raw')/name
                old_path=Path(str(path)+'.json');old=json.loads(old_path.read_text())
                if (sha(old_path)!=receipt['borrowed_receipt_sha256']
                        or old['fingerprint']!=self.fingerprint['cache']['frozen_raw_identity']
                        or old['entry']!=receipt['entry'] or old['sha256']!=m['sha256']):
                    raise ValueError('Borrowed frozen raw receipt changed')
            else:raise ValueError('Unknown raw storage identity')
            if checked_payload(path,receipt['entry'])!=m['sha256']:
                raise ValueError('Raw bytes differ from prepared receipt')
        self.prepared_records[key]=f
        return f


def compare_replay(dataset,mapping,bindings,rows,wrapper):
    for i,row in enumerate(rows):
        if digest(capture(dataset,mapping,i,bindings))!=row['before_state_sha256']:
            raise ValueError('Actual before-state replay differs')
        dataset.trace=[];dataset.pool_attempts=[];dataset.active_request=i
        dataset.preparer.prepared_records={}
        views=wrapper[i];result=row['result']
        checks={'returned_views':inputs(views),'load_trace':dataset.trace,
            'pool_attempts':dataset.pool_attempts,
            'prepared_frames':list(dataset.preparer.prepared_records.values())}
        for k,value in checks.items():
            if value!=result[k]:raise ValueError('Actual independent replay differs: '+k)
        if result['base_index']!=mapping[i]:raise ValueError('Wrapper mapping changed')
        if capture(dataset,mapping,i+1,bindings)!=row['after_state']:
            raise ValueError('Actual shared after-state replay differs')


def verify(expected_requests):
    torch.set_num_threads(2)
    initial_path=ROOT/'initial.json';initial=json.loads(initial_path.read_text())
    bindings=initial['identity']['bindings'];fp=bindings['preparer']
    if (bindings['formal_Table1_result'] or bindings['author_rng_recovered']
            or bindings['workers']!=0 or bindings['epoch']!=0
            or bindings['combination_seed']!=42 or bindings['dataset_seed']!=777
            or bindings['raw_cache_limit_bytes']!=RAW_LIMIT
            or bindings['processed_limit_bytes']!=PROCESSED_LIMIT
            or bindings['journal_limit_bytes']!=JOURNAL_LIMIT or bindings['reserve_bytes']!=RESERVE):
        raise ValueError('Input protocol promoted/changed')
    for p,h in bindings['code_sha256'].items():
        if sha(p)!=h:raise ValueError('Bound input code changed: '+p)
    for p,key in [('scripts/co3d_lazy_dataset_v4.py','lazy_code_sha256'),
        ('scripts/co3d_lazy_dataset_v3.py','frozen_v3_sha256'),
        ('scripts/co3d_lazy_dataset_v2.py','frozen_v2_sha256'),
        ('scripts/co3d_lazy_dataset.py','frozen_v1_sha256')]:
        if sha(p)!=fp[key]:raise ValueError('Frozen reference preparer changed')
    if (sha(MANIFEST)!=bindings['candidate_sha256']
            or sha(STORAGE)!=fp['cache']['directory_report_sha256']
            or sha(PROOF)!=fp['cache']['directory_proof_sha256']):
        raise ValueError('Original candidate/directory proof changed')
    rows,final=read_prefix(ROOT,initial,validate_result)
    if len(rows)!=expected_requests:raise ValueError('Unexpected incomplete/full request count')
    dataset=Source51Landscape(json.loads(MANIFEST.read_text()),ReadOnlyInputs(rows,fp),42,777)
    wrapper=ResizedDataset(REQUESTS,dataset);wrapper.set_epoch(0)
    mapping=[int(x) for x in wrapper._idxs_mapping]
    previous_random=random.getstate()
    try:
        random.seed(bindings['initial_python_seed'])
        if capture(dataset,mapping,0,bindings)!=initial:raise ValueError('Initial source state changed')
        compare_replay(dataset,mapping,bindings,rows,wrapper)
    finally:random.setstate(previous_random)
    if expected_requests==REQUESTS:
        summary=json.loads(SUMMARY.read_text())
        if (summary['request_count']!=REQUESTS or summary['initial_sha256']!=sha(initial_path)
                or summary['final_sampling_state_sha256']!=digest(final)
                or summary['request_sha256']!={f'request_{i:03d}.json':sha(ROOT/f'request_{i:03d}.json') for i in range(REQUESTS)}
                or summary['formal_Table1_result'] or summary['model_forward_count']!=0):
            raise ValueError('Completed input summary differs/promoted')
    return {'status':'actual_source51_landscape_inputs_readonly_replay_verified_not_pose_scores',
        'paper_mapping':'section4.2/Table1 public-HF candidate input prerequisites only',
        'verified_request_count':len(rows),'expected_full_request_count':REQUESTS,
        'all_1000_requests_prepared_and_replayed':len(rows)==REQUESTS,
        'initial_sha256':sha(initial_path),'final_state_sha256':digest(final),
        'request_sha256':{f'request_{i:03d}.json':sha(ROOT/f'request_{i:03d}.json') for i in range(len(rows))},
        'raw_SHA_CRC_processed_SHA_and_receipts_checked':True,
        'input_tensor_GT_rng_trace_and_shared_state_exact_equal':True,
        'network_bytes':0,'model_forward_count':0,'verifier_sha256':sha(__file__),
        'reference_preprocessing_all_new_frames_independently_recomputed':False,
        'author_protocol_equivalence_verified':False,'formal_Table1_result':False,
        'full_paper_completed':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-requests',type=int,default=REQUESTS);p.add_argument('--output',type=Path)
    args=p.parse_args()
    if not 1<=args.expected_requests<=REQUESTS:p.error('expected requests must be1..1000')
    result=verify(args.expected_requests)
    if args.output:atomic_new(args.output,result)
    print(json.dumps(result,indent=2))
