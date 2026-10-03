#!/usr/bin/env python3
"""Resume sparse 51/1000 SOURCE-sampler storage audit; no image download.

Reuse frozen seen41 local indices read-only. Only categories without local
indices need new bounded remote ZIP directory reads. Include nominal frames
and the conservative five-scene retry/jitter closure; neither is a GT proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from audit_co3d_51_released_sampler import PLAN, OUT as SAMPLER, digest
from audit_co3d_zip_footers import probe
from co3d_range_cache import RawCache, RecordedRanges, ReplayIndex, MAX_MEMBER
from co3d_request_journal import atomic_new
from plan_co3d_seen41_storage import expected_paths
from prepare_re10k_rgb_from_archive import sha
from probe_co3d_zip_ranges import inventory, MAX_INDEX_BYTES, MAX_MEMBERS

ROOT = Path('data/co3d_51_source_storage_v1')
OUT = Path('results/co3d_51_source_storage_v1_20261003.json')
REFS = Path('data/co3d_test_metadata/references')
RESERVE = 1024**3
MAX_LOCAL_AUDIT = 512*1024**2
KINDS = ('images', 'depths', 'masks')
FIELDS = ('filename','header_offset','compress_size','file_size','CRC','compress_type','flag_bits')


def save_checkpoint(path, value):
    """Atomic immutable commit; interrupted writes never publish partial JSON."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Existing checkpoint differs; preserve history')
    else:
        atomic_new(path, value)


def validate_receipt(receipt, category, url, fingerprint):
    f = receipt['footer']
    if (receipt['fingerprint'] != fingerprint or receipt['category'] != category
            or receipt['url'] != url or f['url'] != url or not f['etag']
            or receipt['status'] != 'directory_only_not_crc_or_rgb_ready'
            or receipt['full_archive_sha_verified'] or receipt['member_crc_verified']
            or not 0 < f['member_count'] <= MAX_MEMBERS
            or not 0 < f['central_directory_bytes']+65685 <= MAX_INDEX_BYTES
            or not 0 <= f['central_directory_offset'] < f['archive_bytes']
            or f['central_directory_offset']+f['central_directory_bytes'] > f['archive_bytes']
            or not 0 < f['footer_transport_bytes'] <= 65557 or len(f['footer_sha256']) != 64
            or not 0 < receipt['index_transport_bytes'] <= f['central_directory_bytes']+65685
            or sum(r['end']-r['start']+1 for r in receipt['ranges']) != receipt['index_transport_bytes']
            or any(not 0 <= r['start'] <= r['end'] < f['archive_bytes'] or len(r['sha256']) != 64
                   for r in receipt['ranges'])):
        raise ValueError('Invalid directory receipt/transport/source identity')
    for e in receipt['entries'].values():
        if (e['url'] != url or e['etag'] != f['etag'] or e['archive_bytes'] != f['archive_bytes']
                or e['central_directory_offset'] != f['central_directory_offset']):
            raise ValueError('Entry/footer identity mismatch')


def nominal_frames(plan):
    result = {}
    for row in plan['rows']:
        result.setdefault(row['category'], {}).setdefault(row['scene'], set()).update(row['frame_numbers'])
    return {c: {s: sorted(f) for s, f in ss.items()} for c, ss in sorted(result.items())}


def validate_entry(name, entry, urls, checksums):
    parts = Path(name).parts
    if (len(parts) != 4 or parts[2] not in KINDS or entry['filename'] != name
            or entry['url'] not in urls
            or entry['expected_full_archive_sha256'] != checksums[Path(entry['url']).name]
            or any(type(entry[k]) is not int or not 0 < entry[k] <= MAX_MEMBER
                   for k in ('file_size', 'compress_size'))
            or entry['flag_bits'] & 1 or entry['compress_type'] not in (0, 8)
            or not 0 <= entry['header_offset'] < entry['central_directory_offset'] < entry['archive_bytes']
            or not entry['etag'] or not 0 <= entry['CRC'] <= 0xffffffff):
        raise ValueError('Invalid member budget/source identity')


def summarize(entries, required):
    if not required <= set(entries):
        raise ValueError('Incomplete exact member coverage')
    counts = {k: 0 for k in KINDS}
    raw = {k: 0 for k in KINDS}
    compressed = {k: 0 for k in KINDS}
    for name in required:
        e = entries[name]; kind = Path(name).parts[2]
        counts[kind] += 1; raw[kind] += e['file_size']; compressed[kind] += e['compress_size']
    return {'member_count_by_kind': counts, 'advertised_raw_bytes_by_kind': raw,
        'advertised_compressed_bytes_by_kind': compressed,
        'advertised_raw_bytes': sum(raw.values()),
        'advertised_compressed_bytes': sum(compressed.values()),
        'required_paths_sha256': hashlib.sha256('\n'.join(sorted(required)).encode()).hexdigest()}


def check_space(additional=0):
    used = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) if ROOT.exists() else 0
    if used+additional > MAX_LOCAL_AUDIT or shutil.disk_usage('.').free < RESERVE+additional:
        raise RuntimeError('Audit 512MiB ceiling or 1GiB free reserve; never delete existing files')


def read_remote_category(category, frames, urls, checksums, fingerprint):
    entries, receipts = {}, []
    wanted = expected_paths(category, frames)
    for url in urls:
        address = urlparse(url)
        if address.scheme != 'https' or address.hostname != 'dl.fbaipublicfiles.com':
            raise ValueError('Unexpected unreviewed archive host')
        path = ROOT/'archives'/(Path(url).name+'.json')
        if path.exists():
            receipt = json.loads(path.read_text())
            if receipt['fingerprint'] != fingerprint or receipt['url'] != url or receipt['category'] != category:
                raise ValueError('Archive checkpoint belongs to different inputs/code; preserve it')
            part = receipt['entries']
        else:
            check_space(MAX_INDEX_BYTES)
            footer = probe(url)
            if footer['member_count'] > MAX_MEMBERS or footer['central_directory_bytes']+65685 > MAX_INDEX_BYTES:
                raise ValueError('Archive central directory exceeds reviewed hard bound')
            reader = RecordedRanges(url, footer['archive_bytes'], budget=footer['central_directory_bytes']+65685)
            reader.etag = footer['etag']
            stats = inventory(reader, category, {s: set(f) for s, f in frames.items()})
            if stats['zip_member_count'] != footer['member_count']:
                raise ValueError('Footer/directory count changed')
            replay = ReplayIndex(reader.total, reader.blocks)
            part = {}
            with zipfile.ZipFile(replay) as archive:
                for e in archive.infolist():
                    if e.filename not in wanted: continue
                    part[e.filename] = {k: getattr(e, k) for k in FIELDS}
                    part[e.filename].update(url=url, archive_bytes=reader.total, etag=reader.etag,
                        central_directory_offset=footer['central_directory_offset'],
                        expected_full_archive_sha256=checksums[Path(url).name])
            if len(part) != sum(stats['matched_member_counts'].values()):
                raise ValueError('Independent replay member count mismatch')
            receipt = {'status': 'directory_only_not_crc_or_rgb_ready',
                'fingerprint': fingerprint, 'category': category, 'url': url,
                'footer': footer, 'ranges': reader.ranges, 'entries': part,
                'index_transport_bytes': reader.bytes_read,
                'full_archive_sha_verified': False, 'member_crc_verified': False}
            check_space(len(json.dumps(receipt).encode())*2)
            validate_receipt(receipt, category, url, fingerprint)
            save_checkpoint(path, receipt)
            print(f'NEW DIRECTORY {Path(url).name}: {reader.bytes_read} bytes; no RGB', flush=True)
        validate_receipt(receipt, category, url, fingerprint)
        for name, e in part.items():
            if name not in wanted or name in entries:
                raise ValueError('Unexpected/duplicate member across archives')
            validate_entry(name, e, urls, checksums)
            entries[name] = e
        if receipt['full_archive_sha_verified'] or receipt['member_crc_verified']:
            raise ValueError('Directory evidence falsely promoted')
        receipts.append({'path': str(path), 'sha256': sha(path), 'url': url,
            'index_transport_bytes': receipt['index_transport_bytes'],
            'footer_transport_bytes': receipt['footer']['footer_transport_bytes']})
    if set(entries) != wanted:
        raise ValueError(f'Incomplete {category} closure member set; never skip missing frames')
    return entries, {'mode': 'new_directory_read', 'archive_receipts': receipts,
        'recorded_network_bytes': sum(r['index_transport_bytes']+r['footer_transport_bytes'] for r in receipts)}


def inputs():
    plan = json.loads(PLAN.read_text()); report = json.loads(SAMPLER.read_text())
    if sha(PLAN) != report['plan_file_sha256'] or digest(plan) != report['plan_content_sha256']:
        raise ValueError('Released-sampler plan changed')
    parent_path = Path('results/co3d_test_selection_manifest.json')
    parent = json.loads(parent_path.read_text())
    if sha(parent_path) != report['input_sha256'][str(parent_path)]:
        raise ValueError('Pinned selection proof changed')
    for filename, key in (('links.json','links_sha256'),('co3d_sha256.json','checksums_sha256')):
        if sha(REFS/filename) != parent[key]: raise ValueError('Official source references changed')
    links = json.loads((REFS/'links.json').read_text())['full']
    checksums = json.loads((REFS/'co3d_sha256.json').read_text())['full']
    fingerprint = {'plan_file_sha256': sha(PLAN), 'sampler_report_sha256': sha(SAMPLER),
        'links_sha256': parent['links_sha256'], 'checksums_sha256': parent['checksums_sha256'],
        'planner_code_sha256': sha(Path(__file__)),
        'range_code_sha256': sha(Path('scripts/probe_co3d_zip_ranges.py')),
        'footer_code_sha256': sha(Path('scripts/audit_co3d_zip_footers.py')),
        'frozen_cache_code_sha256': sha(Path('scripts/co3d_range_cache.py'))}
    return plan, links, checksums, fingerprint


def category_budget(category, closure, nominal, links, checksums, fingerprint, old_cache, verify_only):
    path = ROOT/'categories'/(category+'.json')
    required = expected_paths(category, closure)
    nominal_required = expected_paths(category, nominal)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['fingerprint'] != fingerprint or saved['category'] != category:
            raise ValueError('Category budget checkpoint changed; preserve history')
        source = saved['source']
        if source['mode'] == 'reused_frozen_local_index':
            original = Path(source['path'])
            if sha(original) != source['sha256']: raise ValueError('Frozen local index changed')
            full = old_cache.category_index(category)
            entries = {n: full[n] for n in required}
        elif source['mode'] == 'new_directory_read':
            entries = {}
            for record in source['archive_receipts']:
                p = Path(record['path'])
                if sha(p) != record['sha256']: raise ValueError('Archive directory checkpoint changed')
                receipt = json.loads(p.read_text())
                validate_receipt(receipt, category, record['url'], fingerprint)
                if (record['index_transport_bytes'] != receipt['index_transport_bytes']
                        or record['footer_transport_bytes'] != receipt['footer']['footer_transport_bytes']):
                    raise ValueError('Recorded archive transport changed')
                if set(entries) & set(receipt['entries']): raise ValueError('Duplicate receipt members')
                entries.update(receipt['entries'])
        else:
            raise ValueError('Unknown directory evidence source')
    else:
        if verify_only: raise FileNotFoundError(path)
        original = old_cache.root/'indices'/(category+'.json')
        if original.exists() and category in old_cache.selected:
            # Guard existence BEFORE using the old cache: never trigger its download branch.
            full = old_cache.category_index(category)
            entries = {n: full[n] for n in required}
            source = {'mode': 'reused_frozen_local_index', 'path': str(original),
                'sha256': sha(original), 'recorded_network_bytes': 0}
        else:
            entries, source = read_remote_category(category, closure, links[category][1:], checksums, fingerprint)
    if set(entries) != required: raise ValueError('Closure member coverage differs')
    for name, e in entries.items(): validate_entry(name, e, links[category][1:], checksums)
    value = {'status': 'exact_nominal_and_retry_closure_directory_budget_not_gt_ready',
        'category': category, 'fingerprint': fingerprint, 'source': source,
        'nominal': summarize(entries, nominal_required), 'retry_jitter_closure': summarize(entries, required),
        'rgb_bytes_transferred': 0, 'member_crc_verified': False, 'formal_Table1_result': False}
    if path.exists():
        if value != saved: raise ValueError('Category budget recomputation differs')
    else:
        check_space(len(json.dumps(value).encode())*2); save_checkpoint(path, value)
    return value


def aggregate(rows, key):
    result = {}
    for field in ('member_count_by_kind','advertised_raw_bytes_by_kind','advertised_compressed_bytes_by_kind'):
        result[field] = {k: sum(r[key][field][k] for r in rows) for k in KINDS}
    result['advertised_raw_bytes'] = sum(result['advertised_raw_bytes_by_kind'].values())
    result['advertised_compressed_bytes'] = sum(result['advertised_compressed_bytes_by_kind'].values())
    return result


def run(verify_only=False):
    plan, links, checksums, fingerprint = inputs()
    closure = plan['potential_retry_jitter_frames']; nominal = nominal_frames(plan)
    old_cache = RawCache()
    rows = []
    for category, scenes in closure.items():
        row = category_budget(category, scenes, nominal.get(category, {}), links, checksums,
                              fingerprint, old_cache, verify_only)
        rows.append(row)
        old_cache.indexes.clear()  # release only in-memory dictionaries; NO file deletion
        print(f"BUDGET {len(rows)}/{len(closure)} {category}: {row['source']['mode']}; "
              f"nominal_raw={row['nominal']['advertised_raw_bytes']}", flush=True)
    a, b = aggregate(rows, 'nominal'), aggregate(rows, 'retry_jitter_closure')
    expected_a = len({(r['category'],r['scene'],f) for r in plan['rows'] for f in r['frame_numbers']})
    expected_b = sum(len(f) for ss in closure.values() for f in ss.values())
    if a['member_count_by_kind'] != {k: expected_a for k in KINDS} or b['member_count_by_kind'] != {k: expected_b for k in KINDS}:
        raise ValueError('Aggregate member coverage differs')
    result = {'status': 'source_51_1000_nominal_and_retry_directory_budget_complete_not_gt_or_pose',
        'paper_mapping': 'section 4.2 / Table 1 source sampling storage feasibility',
        'fingerprint': fingerprint, 'category_count': len(rows), 'request_count': len(plan['rows']),
        'nominal': a, 'retry_jitter_closure': b,
        'reused_local_index_categories': [r['category'] for r in rows if r['source']['mode']=='reused_frozen_local_index'],
        'new_directory_categories': [r['category'] for r in rows if r['source']['mode']=='new_directory_read'],
        'recorded_new_directory_transport_bytes': sum(r['source']['recorded_network_bytes'] for r in rows),
        'category_journals_sha256': {r['category']: sha(ROOT/'categories'/(r['category']+'.json')) for r in rows},
        'raw_rgb_depth_mask_downloaded': False, 'member_crc_verified': False,
        'full_archive_sha_verified': False, 'actual_gt_validity_verified': False,
        'processed_storage_budget_measured': False, 'author_protocol_equivalence_verified': False,
        'formal_Table1_result': False, 'model_forward_count': 0, 'full_paper_completed': False,
        'historical_cache_ceiling_bytes': 2*1024**3,
        'nominal_raw_exceeds_historical_cache_ceiling': a['advertised_raw_bytes'] > 2*1024**3,
        'note': 'Directory lengths only. Closure is conservative; real GT may change draws. '
                'The old235 complete-scope scan is NOT repeated; frozen local indices are reused. '
                'Missing13-category indices are read with archive checkpoints. No RGB, model or eviction.'}
    if OUT.exists():
        saved = json.loads(OUT.read_text())
        if saved != result: raise ValueError('Final budget differs; preserve existing report')
    elif verify_only: raise FileNotFoundError(OUT)
    else: save_checkpoint(OUT, result)
    print(json.dumps({k: result[k] for k in ('status','nominal','retry_jitter_closure',
        'recorded_new_directory_transport_bytes','nominal_raw_exceeds_historical_cache_ceiling')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    run(parser.parse_args().verify_only)
