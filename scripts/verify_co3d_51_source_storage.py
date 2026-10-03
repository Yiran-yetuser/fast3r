#!/usr/bin/env python3
"""Independent OFFLINE recomputation of sparse source-51 storage budgets.

Reads frozen index/directory receipts, not remote ZIPs or RGB. This verifies
only coverage, recorded source identities and advertised sizes, never GT.
"""
import argparse
import hashlib
import json
from pathlib import Path

from prepare_re10k_rgb_from_archive import sha, save_identical

ROOT = Path('data/co3d_51_source_storage_v1')
REPORT = Path('results/co3d_51_source_storage_v1_20261003.json')
PLAN = Path('data/co3d_test_metadata/released_nominal_51_1000_seed777_v1.json')
KINDS = ('images', 'depths', 'masks')
FALSE_FLAGS = ('raw_rgb_depth_mask_downloaded', 'member_crc_verified', 'full_archive_sha_verified',
    'actual_gt_validity_verified', 'processed_storage_budget_measured',
    'author_protocol_equivalence_verified', 'formal_Table1_result', 'full_paper_completed')


def assert_honesty(report):
    if any(report[k] is not False for k in FALSE_FLAGS) or report['model_forward_count'] != 0:
        raise ValueError('Directory evidence promoted to images/GT/model/paper completion')


def paths(category, scenes):
    return {f'{category}/{scene}/{kind}/frame{f:06d}{suffix}'
        for scene, frames in scenes.items() for f in frames
        for kind, suffix in (('images','.jpg'),('depths','.jpg.geometric.png'),('masks','.png'))}


def measure(entries, required):
    if not set(required) <= set(entries): raise ValueError('Missing required members')
    size = {k: 0 for k in KINDS}; packed = {k: 0 for k in KINDS}; count = {k: 0 for k in KINDS}
    for name in required:
        e = entries[name]; kind = name.split('/')[2]
        for key in ('file_size','compress_size'):
            if type(e[key]) is not int or not 0 < e[key] <= 32*1024**2:
                raise ValueError('Invalid selected member lengths')
        if (e['filename'] != name or e['compress_type'] not in (0,8) or e['flag_bits'] & 1
                or not 0 <= e['header_offset'] < e['central_directory_offset'] < e['archive_bytes']):
            raise ValueError('Invalid selected member metadata')
        count[kind] += 1; size[kind] += e['file_size']; packed[kind] += e['compress_size']
    return {'member_count_by_kind': count, 'advertised_raw_bytes_by_kind': size,
        'advertised_compressed_bytes_by_kind': packed, 'advertised_raw_bytes': sum(size.values()),
        'advertised_compressed_bytes': sum(packed.values()),
        'required_paths_sha256': hashlib.sha256('\n'.join(sorted(required)).encode()).hexdigest()}


def verify():
    r = json.loads(REPORT.read_text()); assert_honesty(r)
    plan = json.loads(PLAN.read_text()); fingerprint = r['fingerprint']
    for key, path in (('plan_file_sha256', PLAN),
            ('sampler_report_sha256', Path('results/co3d_51_released_sampler_20261003.json')),
            ('links_sha256', Path('data/co3d_test_metadata/references/links.json')),
            ('checksums_sha256', Path('data/co3d_test_metadata/references/co3d_sha256.json')),
            ('planner_code_sha256', Path('scripts/plan_co3d_51_source_storage.py')),
            ('range_code_sha256', Path('scripts/probe_co3d_zip_ranges.py')),
            ('footer_code_sha256', Path('scripts/audit_co3d_zip_footers.py')),
            ('frozen_cache_code_sha256', Path('scripts/co3d_range_cache.py'))):
        if sha(path) != fingerprint[key]: raise ValueError('Source/input fingerprint changed')
    links = json.loads(Path('data/co3d_test_metadata/references/links.json').read_text())['full']
    checksums = json.loads(Path('data/co3d_test_metadata/references/co3d_sha256.json').read_text())['full']
    closure = plan['potential_retry_jitter_frames']; nominal = {}
    for row in plan['rows']:
        nominal.setdefault(row['category'], {}).setdefault(row['scene'], set()).update(row['frame_numbers'])
    if (r['category_count'] != len(closure) or set(r['category_journals_sha256']) != set(closure)
            or r['request_count'] != len(plan['rows']) or len(closure) != 51
            or len(plan['rows']) != 1000): raise ValueError('Incomplete category/request coverage')
    totals = {scope:{field:{k:0 for k in KINDS} for field in ('member_count_by_kind',
        'advertised_raw_bytes_by_kind','advertised_compressed_bytes_by_kind')}
        for scope in ('nominal','retry_jitter_closure')}
    reused, new, network = [], [], 0
    for category in sorted(closure):
        p = ROOT/'categories'/(category+'.json')
        if sha(p) != r['category_journals_sha256'][category]: raise ValueError('Category journal changed')
        row = json.loads(p.read_text())
        if (row['category'] != category or row['fingerprint'] != fingerprint
                or row['formal_Table1_result'] or row['member_crc_verified'] or row['rgb_bytes_transferred'] != 0):
            raise ValueError('Category identity/honesty changed')
        source = row['source']; required = paths(category, closure[category])
        if source['mode'] == 'reused_frozen_local_index':
            index = Path('data/co3d_range_cache/indices')/(category+'.json')
            if source['path'] != str(index) or sha(index) != source['sha256'] or source['recorded_network_bytes'] != 0:
                raise ValueError('Frozen index binding changed')
            saved = json.loads(index.read_text())
            if saved['category'] != category: raise ValueError('Wrong frozen category')
            entries = {n:saved['entries'][n] for n in required}; reused.append(category)
        elif source['mode'] == 'new_directory_read':
            if [a['url'] for a in source['archive_receipts']] != links[category][1:]:
                raise ValueError('Source archive universe/order changed')
            entries, transferred = {}, 0
            for a in source['archive_receipts']:
                index = ROOT/'archives'/(Path(a['url']).name+'.json')
                if a['path'] != str(index) or sha(index) != a['sha256']: raise ValueError('Archive receipt changed')
                receipt = json.loads(index.read_text()); footer = receipt['footer']
                if (receipt['fingerprint'] != fingerprint or receipt['category'] != category
                        or receipt['url'] != a['url'] or footer['url'] != a['url']
                        or receipt['full_archive_sha_verified'] or receipt['member_crc_verified']
                        or receipt['index_transport_bytes'] != a['index_transport_bytes']
                        or footer['footer_transport_bytes'] != a['footer_transport_bytes']
                        or sum(x['end']-x['start']+1 for x in receipt['ranges']) != a['index_transport_bytes']
                        or not 0 < footer['member_count'] <= 400000
                        or not 0 < footer['central_directory_bytes']+65685 <= 64*1024**2
                        or not 0 < a['index_transport_bytes'] <= footer['central_directory_bytes']+65685
                        or any(not 0 <= x['start'] <= x['end'] < footer['archive_bytes'] or len(x['sha256']) != 64
                               for x in receipt['ranges'])):
                    raise ValueError('Archive transport/source binding changed')
                if set(entries) & set(receipt['entries']): raise ValueError('Duplicate cross-archive selected members')
                for name,e in receipt['entries'].items():
                    if (e['url'] != a['url'] or e['etag'] != footer['etag']
                            or e['archive_bytes'] != footer['archive_bytes']
                            or e['central_directory_offset'] != footer['central_directory_offset']):
                        raise ValueError('Entry/footer identity changed')
                entries.update(receipt['entries'])
                transferred += a['index_transport_bytes']+a['footer_transport_bytes']
            if transferred != source['recorded_network_bytes']: raise ValueError('Network sum changed')
            network += transferred; new.append(category)
        else: raise ValueError('Unknown directory source')
        if set(entries) != required: raise ValueError('Exact closure paths missing/extra')
        for name,e in entries.items():
            if (e['url'] not in links[category][1:]
                    or e['expected_full_archive_sha256'] != checksums[Path(e['url']).name]):
                raise ValueError('Unpinned official member source')
        for scope, wanted in (('nominal', paths(category,nominal.get(category,{}))), ('retry_jitter_closure',required)):
            got = measure(entries,wanted)
            if row[scope] != got: raise ValueError('Independent category count/size/hash differs')
            for field,values in totals[scope].items():
                for k in KINDS: values[k] += got[field][k]
    for scope, total in totals.items():
        total['advertised_raw_bytes'] = sum(total['advertised_raw_bytes_by_kind'].values())
        total['advertised_compressed_bytes'] = sum(total['advertised_compressed_bytes_by_kind'].values())
        if r[scope] != total: raise ValueError('Independent aggregate count/size differs')
    if (r['reused_local_index_categories'] != reused or r['new_directory_categories'] != new
            or r['recorded_new_directory_transport_bytes'] != network
            or r['historical_cache_ceiling_bytes'] != 2*1024**3
            or r['nominal_raw_exceeds_historical_cache_ceiling'] != (totals['nominal']['advertised_raw_bytes']>2*1024**3)):
        raise ValueError('Reuse/network/storage conclusion differs')
    return {'status':'source_51_directory_budget_independently_recomputed_not_gt_ready',
        'paper_mapping':'section 4.2 / Table 1 source sampling storage feasibility only',
        'report_sha256':sha(REPORT), 'verifier_sha256':sha(Path(__file__)),
        'category_count':len(closure),'request_count':len(plan['rows']),
        'nominal':totals['nominal'],'retry_jitter_closure':totals['retry_jitter_closure'],
        'reused_index_category_count':len(reused),'new_directory_category_count':len(new),
        'recorded_new_directory_transport_bytes':network,
        'raw_rgb_depth_mask_downloaded':False,'actual_gt_validity_verified':False,
        'processed_storage_budget_measured':False,'formal_Table1_result':False,'full_paper_completed':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-json',type=Path,default=Path('results/co3d_51_source_storage_verified_20261003.json'))
    args=parser.parse_args();result=verify();save_identical(args.output_json,result)
    print(json.dumps(result,indent=2))
