#!/usr/bin/env python3
"""Read-only independent verification of saved RGB/official GT inventories."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from check_pose_data import read_split
from fast3r_hf_re10k_pose_eval import sha
from prepare_re10k_rgb_from_archive import save_identical


def verify(data_root, metadata_root, report_path,
           official_path=Path('results/re10k_metadata_manifest.json'),
           index_path=Path('results/re10k_rgb_index_audit.json'),
           split_path=Path('scripts/re10k_test_1800.txt')):
    prepared = json.loads(report_path.read_text())
    official = json.loads(official_path.read_text())
    index = json.loads(index_path.read_text())
    names = set(read_split(split_path))
    if prepared['split_sha256'] != sha(split_path):
        raise ValueError('Prepared split changed')
    if set(prepared['scenes']) != set(index['covered_scene_ids']) or set(prepared['missing_scene_ids']) != names-set(prepared['scenes']):
        raise ValueError('Prepared/missing set changed')
    count, shapes = 0, collections.Counter()
    for name, row in prepared['scenes'].items():
        folder = data_root/name
        inventory_path = folder/'.fast3r_rgb_inventory_v2.json'
        if sha(inventory_path) != row['inventory_sha256']:
            raise ValueError('Inventory hash mismatch: '+name)
        inventory = json.loads(inventory_path.read_text())
        metadata_path = metadata_root/(name+'.txt')
        if inventory['metadata_sha256'] != official['records'][name]['sha256'] or sha(metadata_path) != inventory['metadata_sha256']:
            raise ValueError('Official GT hash mismatch: '+name)
        timestamps = {line.split()[0] for line in metadata_path.read_text().splitlines()[1:] if line.strip()}
        frames = inventory['frames']
        if len(frames) != row['frames'] or len(frames) != official['records'][name]['camera_record_count'] or {f['timestamp'] for f in frames} != timestamps:
            raise ValueError('Candidate frame inventory mismatch: '+name)
        if {p.name for p in folder.glob('*.jpg')} != {f['filename'] for f in frames}:
            raise ValueError('Actual JPG candidate set mismatch: '+name)
        if inventory['member_sha256'] != prepared['selected_chunk_sha256'][inventory['source_member']]:
            raise ValueError('Chunk provenance mismatch: '+name)
        if not np.isfinite(row['maximum_camera_abs_delta']) or row['maximum_camera_abs_delta'] > 1e-4:
            raise ValueError('Invalid GT camera delta: '+name)
        for frame in frames:
            if not frame['timestamp'].isdigit() or frame['filename'] != frame['timestamp']+'.jpg':
                raise ValueError('Unsafe timestamp filename')
            path = folder/frame['filename']
            if path.stat().st_size != frame['bytes'] or sha(path) != frame['sha256']:
                raise ValueError('Saved RGB byte/hash mismatch: '+str(path))
        count += len(frames)
        shapes['x'.join(map(str,row['rgb_shape_h_w']))] += 1
    if count != prepared['frame_count'] or len(prepared['scenes']) != prepared['prepared_scene_count']:
        raise ValueError('Prepared count mismatch')
    return {'status': 'covered_rgb_bytes_gt_and_inventory_verified_split_incomplete' if prepared['missing_scene_ids'] else 'verified',
            'expected_scene_count': len(names), 'verified_scene_count': len(prepared['scenes']),
            'verified_frame_count': count, 'missing_scene_count': len(prepared['missing_scene_ids']),
            'shape_h_w_scene_counts': dict(sorted(shapes.items())),
            'prepared_manifest_sha256': sha(report_path), 'archive_sha256': prepared['archive_sha256'],
            'maximum_camera_abs_delta': max(r['maximum_camera_abs_delta'] for r in prepared['scenes'].values()),
            'full_scene_coverage_verified': not prepared['missing_scene_ids'], 'formal_pose_metrics_available': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('data/RealEstate10K/videos/test'))
    parser.add_argument('--metadata-root',type=Path,default=Path('data/RealEstate10K/test'))
    parser.add_argument('--prepared-manifest',type=Path,default=Path('results/re10k_rgb_prepared_manifest.json'))
    parser.add_argument('--output-json',type=Path)
    args = parser.parse_args()
    result = verify(args.data_root,args.metadata_root,args.prepared_manifest)
    if args.output_json:
        save_identical(args.output_json,result)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
