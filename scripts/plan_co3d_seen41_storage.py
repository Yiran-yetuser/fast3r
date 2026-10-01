#!/usr/bin/env python3
"""Resume per-category central-directory-only budgets; never download RGB.

Full archive SHA, image CRC/decode and exact Fast3R scene/sampling equivalence
remain unverified. No sparse frame draw is silently chosen by this planner.
"""
import fcntl
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from prepare_re10k_rgb_from_archive import save_identical, sha
from probe_co3d_zip_ranges import HttpRangeFile, inventory

RESERVE=1024**3


def expected_paths(category,selected):
    return {f'{category}/{scene}/{kind}/frame{frame:06d}{suffix}'
            for scene,frames in selected.items() for frame in frames
            for kind,suffix in [('images','.jpg'),('depths','.jpg.geometric.png'),('masks','.png')]}


def main():
    root=Path('data/co3d_test_metadata')
    journal=Path('results/co3d_seen41_storage_progress')
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
                if saved['fingerprint']!=fingerprint or saved['category']!=category:
                    raise ValueError('Existing budget journal belongs to different inputs/code; preserved')
                summaries.append(saved)
                print(f'REUSE INDEX {category}',flush=True)
                continue
            if shutil.disk_usage(journal).free<RESERVE+1024**2:raise RuntimeError('Preserve 1GiB reserve')
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
        save_identical('results/co3d_seen41_storage_budget_20261002.json',result)
        print('ALL 41 DIRECTORY BUDGETS COMPLETE',flush=True)


if __name__=='__main__':main()
