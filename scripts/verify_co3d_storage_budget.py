#!/usr/bin/env python3
"""Offline independent verification of the completed directory-budget journals.

Recompute candidate-path hashes, all sums and identities. Does NOT reread the
remote directories, validate member CRC, decode images or certify paper splits.
"""
import argparse
import hashlib
import json
from pathlib import Path

from prepare_re10k_rgb_from_archive import sha, save_identical

KINDS = ('images', 'depths', 'masks')


def verify_category(saved, category, selected, urls, checksums, fingerprint, footers):
    count = sum(len(f) for f in selected.values())
    if any(len(f) < 10 or len(set(f)) != len(f) for f in selected.values()):
        raise ValueError('Invalid candidate pool')
    paths = {f'{category}/{scene}/{kind}/frame{frame:06d}{suffix}'
             for scene, frames in selected.items() for frame in frames
             for kind, suffix in [('images', '.jpg'), ('depths', '.jpg.geometric.png'), ('masks', '.png')]}
    if (saved['category'] != category or saved['fingerprint'] != fingerprint
            or saved['status'] != 'complete_category_name_and_size_inventory_not_data_ready'
            or saved['expected_frame_count'] != count or len(paths) != 3 * count
            or saved['expected_paths_sha256'] != hashlib.sha256('\n'.join(sorted(paths)).encode()).hexdigest()
            or saved['cross_archive_duplicate_count'] != 0
            or saved['full_archive_sha_verified'] or saved['member_crc_verified']):
        raise ValueError('Candidate coverage/hash/status differs')
    records = saved['archives']
    if [a['url'] for a in records] != urls or len(set(urls)) != len(urls):
        raise ValueError('Archive URL list/order differs')
    for a in records:
        footer = footers[a['url']]
        if (a['expected_full_archive_sha256'] != checksums[Path(a['url']).name]
                or a['etag'] != footer['etag'] or a['archive_bytes'] != footer['archive_bytes']
                or a['zip_member_count'] != footer['member_count']
                or not 0 < a['zip_member_count'] <= 400000
                or not 0 < a['index_transport_bytes'] <= footer['central_directory_bytes'] + 65557 + 128
                or sum(r['end'] - r['start'] + 1 for r in a['ranges']) != a['index_transport_bytes']
                or any(not 0 <= r['start'] <= r['end'] < a['archive_bytes']
                       or len(r['sha256']) != 64 for r in a['ranges'])
                or sum(a['matched_member_counts'].values()) > a['zip_member_count']):
            raise ValueError('Invalid archive identity/transport/count')
        if set(a['matched_member_counts']) != set(KINDS) or set(a['matched_uncompressed_bytes']) != set(KINDS):
            raise ValueError('Member kinds differ')
        if any(not isinstance(v, int) or v < 0 for v in [*a['matched_member_counts'].values(),
                                                       *a['matched_uncompressed_bytes'].values()]):
            raise ValueError('Invalid counts/sizes')
    for kind in KINDS:
        if (saved['matched_member_counts'][kind] != count
                or sum(a['matched_member_counts'][kind] for a in records) != count
                or sum(a['matched_uncompressed_bytes'][kind] for a in records) != saved['matched_uncompressed_bytes'][kind]):
            raise ValueError('Category aggregate differs')
    if sum(a['index_transport_bytes'] for a in records) != saved['index_transport_bytes']:
        raise ValueError('Category transport sum differs')


def verify(root, report_path, journal):
    root, report_path, journal = Path(root), Path(report_path), Path(journal)
    r = json.loads(report_path.read_text())
    protocol_path = Path('results/co3d_seen41_protocol_20261002.json')
    footer_path = Path('results/co3d_zip_footer_preflight_20261002.json')
    manifest = root / 'selected_seqs_test_seen41_candidate.json'
    protocol = json.loads(protocol_path.read_text())
    selected = json.loads(manifest.read_text())
    links = json.loads((root / 'references/links.json').read_text())['full']
    checksums = json.loads((root / 'references/co3d_sha256.json').read_text())['full']
    f = json.loads(footer_path.read_text())
    fingerprint = {'candidate_manifest_sha256': sha(manifest), 'protocol_sha256': sha(protocol_path),
                   'links_sha256': sha(root / 'references/links.json'),
                   'checksums_sha256': sha(root / 'references/co3d_sha256.json'),
                   'planner_sha256': sha(Path('scripts/plan_co3d_seen41_storage.py')),
                   'range_reader_sha256': sha(Path('scripts/probe_co3d_zip_ranges.py')),
                   'footer_preflight_sha256': sha(footer_path)}
    if (list(selected) != protocol['seen_categories'] or len(selected) != 41
            or sha(manifest) != protocol['candidate_manifest_sha256'] or r['fingerprint'] != fingerprint
            or f['protocol_sha256'] != fingerprint['protocol_sha256']
            or f['links_sha256'] != fingerprint['links_sha256']):
        raise ValueError('Pinned inputs or code differ')
    urls = [u for c in selected for u in links[c][1:]]
    if len(urls) != 235 or [a['url'] for a in f['archives']] != urls:
        raise ValueError('All-source URL set differs')
    footers = {a['url']: a for a in f['archives']}
    rows = []
    originals = []
    for c, scenes in selected.items():
        p = journal / (c + '.json')
        entry = json.loads(p.read_text())
        verify_category(entry, c, scenes, links[c][1:], checksums, fingerprint, footers)
        imported = entry.get('imported_prior_journal')
        if imported:
            version = imported['version']
            if version not in ('v1', 'v2'):
                raise ValueError('Unapproved prior journal')
            directory = 'results/co3d_seen41_storage_progress' if version == 'v1' else 'results/co3d_seen41_storage_v2_progress'
            prior = Path(directory) / (c + '.json')
            if str(prior) != imported['path'] or sha(prior) != imported['sha256']:
                raise ValueError('Prior journal changed')
            old = json.loads(prior.read_text())
            if old['fingerprint'] != imported['fingerprint']:
                raise ValueError('Prior fingerprint changed')
            for key, value in old.items():
                if key != 'fingerprint' and entry[key] != value:
                    raise ValueError('Migration changed historical values')
        originals.append({k: v for k, v in entry.items() if k not in ('archives', 'fingerprint')})
        rows.append({'category': c, 'sequence_count': len(scenes),
                     'candidate_frame_count': entry['expected_frame_count'], 'journal_sha256': sha(p),
                     'advertised_uncompressed_bytes_by_kind': entry['matched_uncompressed_bytes']})
    frames = sum(row['candidate_frame_count'] for row in rows)
    sequences = sum(row['sequence_count'] for row in rows)
    sizes = {k: sum(row['advertised_uncompressed_bytes_by_kind'][k] for row in rows) for k in KINDS}
    if (r['status'] != 'seen41_directory_name_and_size_budget_complete_not_data_ready'
            or r['category_summaries'] != originals or r['category_count'] != len(rows)
            or frames != r['candidate_frame_count'] or frames != protocol['candidate_frame_count']
            or sequences != r['selected_sequence_count'] or sequences != protocol['selected_sequence_count']
            or r['matched_member_counts'] != {k: frames for k in KINDS}
            or r['advertised_uncompressed_bytes_by_kind'] != sizes
            or r['advertised_total_uncompressed_bytes'] != sum(sizes.values())
            or r['index_transport_bytes'] != sum(x['index_transport_bytes'] for x in originals)
            or any(r[k] for k in ('full_archive_sha_verified', 'member_crc_verified',
                                 'rgb_depth_mask_and_camera_npz_ready', 'formal_pose_metrics_available',
                                 'original_fast3r_split_equivalence_verified'))):
        raise ValueError('Complete aggregate or honesty flags differ')
    return {'status': 'all_candidate_directory_journals_independently_verified_not_data_ready',
            'paper_mapping': 'section 4.2 / Table 1 data storage feasibility only',
            'budget_report_sha256': sha(report_path), 'verifier_sha256': sha(Path(__file__)),
            'fingerprint': fingerprint, 'category_count': len(rows), 'data_archive_count': len(urls),
            'selected_sequence_count': sequences, 'candidate_frame_count': frames,
            'matched_member_counts': r['matched_member_counts'],
            'advertised_uncompressed_bytes_by_kind': sizes, 'advertised_total_uncompressed_bytes': sum(sizes.values()),
            'recorded_index_transport_bytes_including_reused_journals': r['index_transport_bytes'],
            'categories': rows, 'member_crc_verified': False, 'full_archive_sha_verified': False,
            'rgb_depth_mask_and_camera_npz_ready': False, 'formal_pose_metrics_available': False,
            'original_fast3r_split_equivalence_verified': False,
            'note': 'Offline journal verification, not remote directory reread or new image transfer'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-json', type=Path, default=Path('results/co3d_storage_budget_verified_20261002.json'))
    args = parser.parse_args()
    result = verify('data/co3d_test_metadata', 'results/co3d_seen41_storage_budget_v3_20261002.json',
                    'results/co3d_seen41_storage_v3_progress')
    save_identical(args.output_json, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('categories', 'fingerprint')}, indent=2))


if __name__ == '__main__':
    main()
