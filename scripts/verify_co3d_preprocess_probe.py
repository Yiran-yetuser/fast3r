#!/usr/bin/env python3
"""Offline independent saved-byte check + actual released single-view loader.

No new sampling list, network, model inference or GPU. Sparse probe files never
replace the original candidate image pool; test only the ten prepared entries.
"""
import json
from pathlib import Path

import numpy as np

from prepare_re10k_rgb_from_archive import sha, save_identical
from fast3r.dust3r.datasets.co3d_multiview import Co3d_Multiview
from fast3r.dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset


def main():
    report_path=Path('results/co3d_preprocess_probe_20261002.json')
    report=json.loads(report_path.read_text())
    root=Path('data/co3d_range_probe')
    candidate=Path('data/co3d_test_metadata/selected_seqs_test_seen41_candidate.json')
    if sha(candidate)!=report['candidate_manifest_sha256']:raise ValueError('Candidate changed')
    for key,path in [('probe_sha256','scripts/probe_co3d_preprocess.py'),
                     ('cache_sha256','scripts/co3d_range_cache.py'),
                     ('local_cropping_sha256','fast3r/dust3r/datasets/utils/cropping.py')]:
        if sha(Path(path))!=report['fingerprint'][key]:raise ValueError('Code fingerprint changed')
    selected=json.loads(candidate.read_text())
    c,s=report['category'],report['sequence']
    full_pool=selected[c][s]
    if report['frame_numbers']!=full_pool[:10] or report['frame_count']!=10:raise ValueError('Probe scope changed')
    dataset=Co3d_Multiview.__new__(Co3d_Multiview)
    BaseStereoViewDataset.__init__(dataset,split='test',resolution=(512,384),seed=777)
    dataset.ROOT=str(root/'adapted_processed');dataset.mask_bg=True
    dataset.invalidate={(c,s):{(512,384):[False]*len(full_pool)}}
    rows=[]
    for frame,record in zip(report['frame_numbers'],report['records']):
        stem=f'frame{frame:06d}'
        if record['image_path']!=f'{c}/{s}/images/{stem}.jpg':raise ValueError('Frame identity mismatch')
        for member in record['raw_members']:
            raw=Path('data/co3d_range_cache/raw')/member['path']
            if raw.stat().st_size!=member['bytes'] or sha(raw)!=member['sha256']:
                raise ValueError('Saved raw member differs')
        for rel,expected in record['adapted_processed_file_sha256'].items():
            if sha(root/'adapted_processed'/rel)!=expected or sha(root/'reference_processed'/rel)!=expected:
                raise ValueError('Saved reference/adapted bytes mismatch')
        rel=f'{c}/{s}/images/{stem}.npz'
        with np.load(root/'reference_processed'/rel,allow_pickle=False) as ref,np.load(root/'adapted_processed'/rel,allow_pickle=False) as adapted:
            if set(ref.files)!=set(adapted.files) or any(not np.array_equal(ref[k],adapted[k]) for k in ref.files):
                raise ValueError('NPZ reference arrays mismatch')
        idx=full_pool.index(frame)
        view=dataset._load_view_data(c,s,full_pool,idx,(512,384),np.random.default_rng(777+idx))
        if view is None:raise ValueError('Released loader rejected a probe view; do not skip it')
        # Released base loader may randomly transpose near-square inputs or
        # choose portrait resolution; preserve that behavior, do not stretch.
        if (view['img'].size not in ((512,384),(384,512)) or view['depthmap'].shape!=view['img'].size[::-1]
                or not all(np.isfinite(view[k]).all() for k in ('depthmap','camera_pose','camera_intrinsics'))):
            raise ValueError('Released loader shape/finite check failed')
        rows.append({'frame_number':frame,'original_pool_index':idx,
                     'loader_depth_positive_fraction':float(np.mean(view['depthmap']>0)),
                     'image_size_wh':list(view['img'].size),'depth_shape_hw':list(view['depthmap'].shape),
                     'camera_pose_dtype':str(view['camera_pose'].dtype)})
    result={'status':'ten_saved_frames_independently_verified_and_released_loader_readable_not_benchmark_ready',
            'paper_mapping':'section 4.2 / Table 1 processed data wiring prerequisites',
            'probe_report_sha256':sha(report_path),'verifier_sha256':sha(Path(__file__)),
            'loader_sha256':sha(Path('fast3r/dust3r/datasets/co3d_multiview.py')),
            'base_loader_sha256':sha(Path('fast3r/dust3r/datasets/base/base_stereo_view_dataset.py')),
            'candidate_manifest_sha256':sha(candidate),'category':c,'sequence':s,
            'original_candidate_pool_length':len(full_pool),'prepared_probe_frame_count':len(rows),'rows':rows,
            'all_loader_probe_views_valid':True,'full_candidate_data_ready':False,
            'actual_sampling_and_retries_verified':False,'formal_pose_metrics_available':False,
            'note':'Only _load_view_data exercised with original full scene pool. Released near-square/portrait resolution behavior preserved. No _get_views/100@ evaluation, no inference, no new sparse sampling manifest.'}
    save_identical(Path('results/co3d_preprocess_probe_verified_20261002.json'),result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
