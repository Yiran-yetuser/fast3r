#!/usr/bin/env python3
"""Read-only consistency/GT audit of completed source scan, no source re-download.

Full compressed SHA comes from the finished streaming run, not an independent
second archive hash. This verifier checks saved evidence and exact target sets.
"""
import json
from pathlib import Path
from check_pose_data import read_split, read_re10k_metadata
from prepare_re10k_rgb_from_archive import sha, save_identical

TOTAL=205763619478
SOURCE_SHA='f6055cd8ea1ccce642482ca623f98c21af1c449210760d23d78a43b09e546523'
REVISION='ea8d2427de59817b2f66f17b26276339841eb142'
REPORT=Path('results/re10k_missing_candidates_full_source.json')
INDEX=Path('results/re10k_rgb_index_audit.json')
SPLIT=Path('scripts/re10k_test_1800.txt')


def check_report(report,index,names):
    missing=index['missing_scene_ids'];covered=index['covered_scene_ids']
    if (len(set(missing))!=len(missing) or len(set(covered))!=len(covered)
        or set(missing)&set(covered) or set(missing)|set(covered)!=set(names)):
        raise ValueError('Index is not a unique exact partition of prescribed split')
    if (report['status']!='source_scanned_candidates_not_promoted'
        or report['full_archive_sha_verified'] is not True
        or report['source_expected_bytes']!=TOTAL or report['compressed_bytes_read']!=TOTAL
        or report['source_expected_sha256']!=SOURCE_SHA or report['compressed_sha256']!=SOURCE_SHA
        or report['revision']!=REVISION):
        raise ValueError('Source length/SHA/revision/completion invalid')
    if (report['target_ids_not_observed']!=sorted(missing)
        or report['missing_target_scene_count']!=len(missing)
        or report['target_ids_observed'] or report['complete_candidate_timestamp_scene_ids']
        or report['frames'] or report['extra_timestamps'] or report['staged_frame_count']!=0):
        raise ValueError('No-recovery evidence or exact missing set changed')
    if report['formal_pose_metrics_available'] or report['full_prescribed_split_ready']:
        raise ValueError('Missing data promoted to full benchmark')
    if set(report['official_metadata_sha256'])!=set(missing):
        raise ValueError('Official missing-ID GT set changed')


def run():
    source=json.loads(REPORT.read_text());index=json.loads(INDEX.read_text())
    names=read_split(SPLIT);check_report(source,index,names)
    if len(names)!=1832 or len(index['missing_scene_ids'])!=76 or source['observed_scene_count']!=4137:
        raise ValueError('Archived actual counts changed')
    official=json.loads(Path('results/re10k_metadata_manifest.json').read_text())
    frame_count=0
    for name in source['target_ids_not_observed']:
        path=Path('data/RealEstate10K/test')/(name+'.txt');h=sha(path)
        if h!=source['official_metadata_sha256'][name] or h!=official['records'][name]['sha256']:
            raise ValueError('Official GT SHA changed: '+name)
        records=read_re10k_metadata(path)
        if len(records)!=official['records'][name]['camera_record_count']:
            raise ValueError('Official GT count changed')
        frame_count+=len(records)
    root=Path('data/RealEstate10K_missing_candidate')
    if root.is_symlink() or list(root.rglob('*.png')):
        raise ValueError('Unexpected staged RGB; do not silently ignore')
    prepared=Path('results/re10k_rgb_prepared_manifest.json')
    prior=json.loads(Path('results/re10k_rgb_verification_20261002.json').read_text())
    if sha(prepared)!=prior['prepared_manifest_sha256']:
        raise ValueError('Previously verified prepared manifest changed')
    result={'status':'completed_source_report_and_missing_GT_consistency_verified_no_recovery',
        'source_report_sha256':sha(REPORT),'verifier_sha256':sha(__file__),
        'scanner_sha256':sha('scripts/stream_re10k_missing_candidates.py'),
        'split_sha256':sha(SPLIT),'index_sha256':sha(INDEX),
        'source_stream_bytes':TOTAL,'source_stream_sha256':SOURCE_SHA,
        'source_stream_sha_verified_by_completed_scanner':True,
        'archive_independently_rehashed_this_verification':False,
        'recognized_source_scene_count':4137,'prescribed_scene_count':1832,
        'previously_verified_prepared_scene_count':1756,'missing_scene_count':76,
        'recovered_scene_count':0,'staged_frame_count':0,'missing_GT_camera_record_count':frame_count,
        'missing_scene_ids':source['target_ids_not_observed'],
        'prepared_manifest_sha256':sha(prepared),'prior_RGB_verification_sha256':sha('results/re10k_rgb_verification_20261002.json'),
        'network_bytes_this_verification':0,'formal_pose_metrics_available':False,
        'full_prescribed_split_ready':False,'new_image_geometry_verified':False,
        'limitations':'Absence under fixed recognized tar path/naming protocol; no exhaustive audit of ignored tar paths. Does not establish permanent unavailability or authorize login/cookies. Full-stream SHA is scanner evidence, not independently re-downloaded.'}
    save_identical(Path('results/re10k_missing_full_source_verified_20261002.json'),result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
