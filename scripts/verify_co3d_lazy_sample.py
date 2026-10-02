#!/usr/bin/env python3
"""Independent raw bytes and actual pinned preprocessor comparison for ONE draw.

No sampling changes, network or inference. Reference fixture is isolated and
never replaces the full candidate manifest. Existing partial outputs preserved.
"""
import argparse
import gzip
import io
import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch

from co3d_lazy_dataset import LazyPreparer, ROOT, RESERVE
from probe_co3d_preprocess import exclusive_bytes
from prepare_re10k_rgb_from_archive import sha, save_identical


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wrapper-index',type=int,default=0)
    args=parser.parse_args()
    torch.set_num_threads(2)
    source=Path(f'results/co3d_lazy_draw{args.wrapper_index}_v2_20261002.json')
    report=json.loads(source.read_text())
    prep=LazyPreparer()
    if report['fingerprint']['preparer']!=prep.fingerprint:raise ValueError('Preparer identity changed')
    if sha(Path('scripts/probe_co3d_lazy_sample.py'))!=report['fingerprint']['probe_sha256']:
        raise ValueError('Probe code changed')
    rows=report['prepared_frames']
    if len({r['image_path'] for r in rows})!=len(rows):raise ValueError('Duplicate preparation record')
    work=Path(f'data/co3d_lazy_reference_draw{args.wrapper_index}')
    raw_root=work/'raw'; ref_root=work/'reference_processed'
    work.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(work).free<RESERVE+len(rows)*8*1024**2:raise RuntimeError('Reference reserve')
    annotations={}; sequences={}
    for row in rows:
        c,s,_,_=row['image_path'].split('/')
        a=prep.annotations(c)[row['image_path']]
        if a['frame_number']!=row['annotation_frame_number']:raise ValueError('Annotation identity changed')
        annotations.setdefault(c,[]).append(a);sequences.setdefault(c,set()).add(s)
        for member in row['raw_members']:
            original=prep.cache.root/'raw'/member['path']
            if original.stat().st_size!=member['bytes'] or sha(original)!=member['sha256']:
                raise ValueError('Saved raw byte identity changed')
            target=raw_root/member['path'];target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if sha(target)!=member['sha256']:raise ValueError('Reference raw link changed')
            else:os.link(original,target)
        for rel,expected in row['processed_sha256'].items():
            if sha(prep.root/rel)!=expected:raise ValueError('Saved processed bytes changed')
    for c,anns in annotations.items():
        with zipfile.ZipFile(ROOT/'archives'/f'{c}_000.zip') as z:
            packed=z.read(c+'/sequence_annotations.jgz')
        with gzip.GzipFile(fileobj=io.BytesIO(packed)) as g:raw=g.read(32*1024**2+1)
        if len(raw)>32*1024**2:raise ValueError('Sequence annotation bound')
        seqs=[s for s in json.loads(raw) if s['sequence_name'] in sequences[c]]
        if {s['sequence_name'] for s in seqs}!=sequences[c]:raise ValueError('Missing selected sequence')
        exclusive_bytes(raw_root/c/'frame_annotations.jgz',gzip.compress(json.dumps(anns).encode(),mtime=0))
        exclusive_bytes(raw_root/c/'sequence_annotations.jgz',gzip.compress(json.dumps(seqs).encode(),mtime=0))
        exclusive_bytes(raw_root/c/'set_lists/probe_fewview_train.json',json.dumps(
            {'test':[[a['sequence_name'],a['frame_number'],a['image']['path']] for a in anns]}).encode())
    marker=work/'complete.json'
    marker_expected={'probe_report_sha256':sha(source),'verifier_sha256':sha(Path(__file__)),
                     'preparer_fingerprint':prep.fingerprint}
    if marker.exists():
        if json.loads(marker.read_text())!=marker_expected:raise ValueError('Reference run identity changed')
    else:
        if ref_root.exists() and any(ref_root.rglob('*')):
            raise ValueError('Partial reference output preserved; choose a reviewed recovery path')
        for c in annotations: prep.ns['prepare_sequences'](c,str(raw_root),str(ref_root),512,'test',.5,50,42,False)
        save_identical(marker,marker_expected)
    checked=[]
    for row in rows:
        a=prep.annotations(row['image_path'].split('/')[0])[row['image_path']]
        for key in ('image','depth','mask'):
            if sha(ref_root/a[key]['path'])!=row['processed_sha256'][a[key]['path']]:
                raise ValueError('Actual pinned preprocessing JPEG/PNG mismatch')
        rel=Path(a['image']['path']).with_suffix('.npz')
        with np.load(ref_root/rel,allow_pickle=False) as ref,np.load(prep.root/rel,allow_pickle=False) as saved:
            if set(ref.files)!=set(saved.files) or any(not np.array_equal(ref[k],saved[k]) for k in ref.files):
                raise ValueError('Actual pinned preprocessing camera/max-depth mismatch')
        checked.append({'image_path':a['image']['path'],'image_depth_mask_bytes_equal':True,'npz_arrays_equal':True})
    result={'status':'one_real_draw_raw_and_pinned_reference_preprocessing_verified_not_benchmark',
            **marker_expected,'checked_frame_count':len(checked),'frames':checked,
            'network_bytes':0,'formal_pose_metrics_available':False,'full_candidate_data_ready':False,
            'paper_selection_and_sampling_equivalence_verified':False,
            'note':'Only actually prepared frames from this one draw checked; same runtime. No original ZIP full SHA claim.'}
    save_identical(Path(f'results/co3d_lazy_draw{args.wrapper_index}_verified_20261002.json'),result)
    print('INDEPENDENT pinned preprocessor output equality:',len(checked),'unique real frames; network0',flush=True)


if __name__=='__main__':main()
