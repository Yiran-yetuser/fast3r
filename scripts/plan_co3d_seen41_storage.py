#!/usr/bin/env python3
"""Resume per-category central-directory-only budgets; never download RGB.

Full archive SHA, image CRC/decode and exact Fast3R scene/sampling equivalence
remain unverified. No sparse frame draw is silently chosen by this planner.
"""
import argparse
import fcntl
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from prepare_re10k_rgb_from_archive import save_identical, sha
from probe_co3d_zip_ranges import HttpRangeFile, inventory

RESERVE=1024**3
V1_CODE_SHA={'planner_sha256':'706c7d82cbd7d04f215e790c4b96e33e824a90d8deeedbcdfbefe513901f7eab',
             'range_reader_sha256':'a7a6f3776ee44d3827c338c6315736a8c3c286f4d422e065685de9104981c5e8'}


def expected_paths(category,selected):
    return {f'{category}/{scene}/{kind}/frame{frame:06d}{suffix}'
            for scene,frames in selected.items() for frame in frames
            for kind,suffix in [('images','.jpg'),('depths','.jpg.geometric.png'),('masks','.png')]}


def validate_saved(saved,category,selected,urls,checksums,fingerprint,legacy=False):
    """Only audited v1 code can migrate; inputs, paths hash and all sums must match."""
    expected=dict(fingerprint)
    if legacy:expected.update(V1_CODE_SHA)
    if saved['fingerprint']!=expected or saved['category']!=category:
        raise ValueError('Existing budget journal belongs to different inputs/code; preserved')
    required=expected_paths(category,selected)
    count=sum(map(len,selected.values()))
    if (saved['status']!='complete_category_name_and_size_inventory_not_data_ready'
            or saved['expected_frame_count']!=count or saved['cross_archive_duplicate_count']!=0
            or saved['expected_paths_sha256']!=hashlib.sha256('\n'.join(sorted(required)).encode()).hexdigest()
            or saved['full_archive_sha_verified'] or saved['member_crc_verified']):
        raise ValueError('Invalid completed category journal')
    records=saved['archives']
    if [r['url'] for r in records]!=urls:raise ValueError('Archive list/order changed')
    for record in records:
        if record['expected_full_archive_sha256']!=checksums[Path(record['url']).name]:
            raise ValueError('Official ZIP checksum reference changed')
        if (record['index_transport_bytes']>32*1024**2 or not record['etag']
                or sum(r['end']-r['start']+1 for r in record['ranges'])!=record['index_transport_bytes']
                or any(not 0<=r['start']<=r['end']<record['archive_bytes'] for r in record['ranges'])):
            raise ValueError('Invalid saved Range bounds')
    for kind in ('images','depths','masks'):
        if (saved['matched_member_counts'][kind]!=count
                or sum(r['matched_member_counts'][kind] for r in records)!=count
                or sum(r['matched_uncompressed_bytes'][kind] for r in records)!=saved['matched_uncompressed_bytes'][kind]):
            raise ValueError('Category aggregate differs from its archive records')
    if sum(r['index_transport_bytes'] for r in records)!=saved['index_transport_bytes']:
        raise ValueError('Category transport sum differs')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-verified-v1',action='store_true',
                        help='Read-only validate pinned v1 completed categories into a NEW v2 journal')
    args=parser.parse_args()
    root=Path('data/co3d_test_metadata')
    journal=Path('results/co3d_seen41_storage_v2_progress')
    journal.mkdir(parents=True,exist_ok=True)
    with (journal/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol_path=Path('results/co3d_seen41_protocol_20261002.json')
        protocol=json.loads(protocol_path.read_text())
        manifest=root/'selected_seqs_test_seen41_candidate.json'
        if sha(manifest)!=protocol['candidate_manifest_sha256']:raise ValueError('Candidate changed')
        selected=json.loads(manifest.read_text())
        parent=json.loads(Path('results/co3d_test_selection_manifest.json').read_text())
        for filename,key in [('links.json','links_sha256'),('co3d_sha256.json','checksums_sha256')]:
            if sha(root/'references'/filename)!=parent[key]:raise ValueError('Archive reference changed')
        links=json.loads((root/'references/links.json').read_text())['full']
        checksums=json.loads((root/'references/co3d_sha256.json').read_text())['full']
        fingerprint={'candidate_manifest_sha256':sha(manifest),'protocol_sha256':sha(protocol_path),
                     'links_sha256':parent['links_sha256'],'checksums_sha256':parent['checksums_sha256'],
                     'planner_sha256':sha(Path(__file__)),'range_reader_sha256':sha(Path('scripts/probe_co3d_zip_ranges.py'))}
        summaries=[]
        for category in protocol['seen_categories']:
            path=journal/(category+'.json')
            if path.exists():
                saved=json.loads(path.read_text())
                validate_saved(saved,category,selected[category],links[category][1:],checksums,fingerprint)
                summaries.append(saved)
                print(f'REUSE INDEX {category}',flush=True)
                continue
            if shutil.disk_usage(journal).free<RESERVE+1024**2:raise RuntimeError('Preserve 1GiB reserve')
            legacy_path=Path('results/co3d_seen41_storage_progress')/(category+'.json')
            if args.import_verified_v1 and legacy_path.exists():
                saved=json.loads(legacy_path.read_text())
                validate_saved(saved,category,selected[category],links[category][1:],checksums,fingerprint,legacy=True)
                saved['imported_v1_journal_sha256']=sha(legacy_path)
                saved['imported_v1_fingerprint']=saved['fingerprint']
                saved['fingerprint']=fingerprint
                save_identical(path,saved)
                summaries.append(saved)
                print(f'VALIDATED V1 IMPORT {category}; no network replay, original preserved',flush=True)
                continue
            matched=set()
            records=[]
            for url in links[category][1:]:
                with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=45) as response:
                    total=int(response.headers['Content-Length'])
                if not 0<total<128*1024**3:raise ValueError('Unexpected archive size')
                reader=HttpRangeFile(url,total)
                record=inventory(reader,category,{s:set(f) for s,f in selected[category].items()},matched)
                record.update(url=url,archive_bytes=total,expected_full_archive_sha256=checksums[Path(url).name],
                              index_transport_bytes=reader.bytes_read,etag=reader.etag,ranges=reader.ranges)
                records.append(record)
                print(f'INDEX {Path(url).name}: {reader.bytes_read} bytes',flush=True)
            required=expected_paths(category,selected[category])
            if matched!=required:
                raise ValueError(f'Member path coverage mismatch {category}: missing={len(required-matched)}, extra={len(matched-required)}')
            count={k:sum(r['matched_member_counts'][k] for r in records) for k in ('images','depths','masks')}
            size={k:sum(r['matched_uncompressed_bytes'][k] for r in records) for k in count}
            saved={'status':'complete_category_name_and_size_inventory_not_data_ready','category':category,
                   'fingerprint':fingerprint,'expected_frame_count':sum(map(len,selected[category].values())),
                   'matched_member_counts':count,'matched_uncompressed_bytes':size,
                   'index_transport_bytes':sum(r['index_transport_bytes'] for r in records),
                   'expected_paths_sha256':hashlib.sha256('\n'.join(sorted(required)).encode()).hexdigest(),
                   'cross_archive_duplicate_count':0,'archives':records,
                   'full_archive_sha_verified':False,'member_crc_verified':False}
            save_identical(path,saved)
            summaries.append(saved)
            print(f'CATEGORY COMPLETE {category}: {count}, {size}',flush=True)
        count={k:sum(s['matched_member_counts'][k] for s in summaries) for k in ('images','depths','masks')}
        sizes={k:sum(s['matched_uncompressed_bytes'][k] for s in summaries) for k in count}
        result={'status':'seen41_directory_name_and_size_budget_complete_not_data_ready',
                'paper_mapping':'section 4.2 / Table 1 CO3D candidate storage feasibility',
                'fingerprint':fingerprint,'category_count':len(summaries),
                'selected_sequence_count':protocol['selected_sequence_count'],
                'candidate_frame_count':protocol['candidate_frame_count'],
                'matched_member_counts':count,'advertised_uncompressed_bytes_by_kind':sizes,
                'advertised_total_uncompressed_bytes':sum(sizes.values()),
                'index_transport_bytes':sum(s['index_transport_bytes'] for s in summaries),
                'category_summaries':[{k:v for k,v in s.items() if k not in ('archives','fingerprint')} for s in summaries],
                'full_archive_sha_verified':False,'member_crc_verified':False,
                'rgb_depth_mask_and_camera_npz_ready':False,'formal_pose_metrics_available':False,
                'original_fast3r_split_equivalence_verified':False,
                'note':'Complete candidate-name coverage only; no RGB extraction, sparse sampling change or benchmark promotion'}
        if any(v!=protocol['candidate_frame_count'] for v in count.values()):raise ValueError('Aggregate frame count mismatch')
        save_identical('results/co3d_seen41_storage_budget_v2_20261002.json',result)
        print('ALL 41 DIRECTORY BUDGETS COMPLETE',flush=True)


if __name__=='__main__':main()
