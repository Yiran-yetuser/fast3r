#!/usr/bin/env python3
"""Read-only validation of the complete, paired public-weight 7-Scenes report."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(root):
    manifest = json.loads((root / 'results/7scenes_data_manifest.json').read_text())
    report = json.loads((root / 'results/7scenes_paired_seed42_stride20.json').read_text())
    require(manifest['status'] == 'prepared', 'Data preparation is incomplete')
    sequences = manifest['sequences']
    expected = {r['scene'] + '/' + r['sequence']: r for r in sequences}
    categories = set(manifest['scenes_expected'])
    require(len(categories) == 7 and len(expected) == len(sequences) ==
            manifest['trajectory_count'] == 18, 'Incomplete or duplicated trajectory inventory')
    split_set = set()
    for scene in categories:
        split = root / 'data/7_scenes_processed' / scene / 'TestSplit.txt'
        for line in split.read_text().splitlines():
            if line.strip():
                split_set.add(scene + '/seq-' + f'{int(line.strip().removeprefix("sequence")):02d}')
    require(split_set == set(expected), 'Manifest differs from official saved TestSplit')
    for key, row in expected.items():
        indices = list(range(0, row['original_frame_count'], 20))
        require(row['stride'] == 20 and row['stored_indices'] == indices,
                f'Incorrect original stride indices: {key}')
        require(row['selected_member_crc_checked'] and row['sequence_zip_crc_checked'],
                f'Missing CRC checks: {key}')
        require(len(row['sequence_zip_sha256']) == 64 and
                row['preprocessing_source_sha256'] == manifest['preprocessing_source_sha256'],
                f'Incomplete provenance: {key}')
        directory = root / 'data/7_scenes_processed' / key
        inventory = json.loads((directory / 'frame_inventory.json').read_text())
        require(inventory['stored_indices'] == indices and
                inventory['original_frame_count'] == row['original_frame_count'],
                f'On-disk inventory differs: {key}')
        for index in indices:
            for suffix in ('color.png', 'depth.proj.png', 'pose.txt'):
                path = directory / f'frame-{index:06d}.{suffix}'
                require(path.is_file() and path.stat().st_size > 0, f'Missing data: {path}')
    rows = report['scenes']
    require(report['dataset'] == '7scenes' and report['seed'] == 42 and
            report['kf_every'] == 20 and report['head'] == 'both', 'Unexpected protocol')
    require(report['paired_same_forward'] is True and report['paper_distance_multiplier'] == 100,
            'Missing same-forward or distance convention')
    require(report['scene_count'] == len(rows) == len({r['scene'] for r in rows}) == 18 and
            {r['scene'] for r in rows} == set(expected), 'Report trajectory set is incomplete')
    for row in rows:
        key = row['scene']
        seed = 42 + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')
        require(row['seed'] == seed and row['views'] == len(expected[key]['stored_indices']),
                f'Incorrect scene seed/view count: {key}')
        require(row['paired_same_forward'] is True and row['metrics'] ==
                row['metrics_by_head']['local'], f'Incorrect pairing: {key}')
    require(report['aggregate_mean_head'] == 'local' and report['aggregate_mean'] ==
            report['aggregate_by_head']['local'], 'Legacy aggregate is not local')
    for head in ('local', 'global'):
        for metric, aggregate in report['aggregate_by_head'][head].items():
            values = [row['metrics_by_head'][head][metric] for row in rows]
            require(all(math.isfinite(v) for v in values) and math.isfinite(aggregate),
                    f'Non-finite {head}/{metric}')
            require(math.isclose(aggregate, statistics.mean(values), rel_tol=1e-12, abs_tol=1e-12),
                    f'Incorrect aggregate {head}/{metric}')
    original = json.loads((root / 'results/nrgbd_seed42_stride40.json').read_text())
    paired = json.loads((root / 'results/nrgbd_paired_seed42_stride40.json').read_text())
    historical = {r['scene']: r for r in original['scenes']}
    require(len(historical) == len(paired['scenes']) == 9 and
            set(historical) == {r['scene'] for r in paired['scenes']}, 'NRGBD inventory mismatch')
    require(all(r['metrics_by_head']['local'] == historical[r['scene']]['metrics'] and
                r['seed'] == historical[r['scene']]['seed'] and
                r['views'] == historical[r['scene']]['views'] for r in paired['scenes']),
            'NRGBD local differs from historical report')
    local = report['aggregate_by_head']['local']
    return {'status': 'passed', 'categories': 7, 'trajectories': len(rows),
            'original_frames': sum(r['original_frame_count'] for r in sequences),
            'sampled_views': sum(r['views'] for r in rows),
            'same_forward': True, 'nrgbd_historical_local_exact_match': True,
            'table3_local_median_x100': {k: local[k] * 100 for k in
                                       ('accuracy_median', 'completion_median')},
            'table3_error_increase_percent': {
                k: (local[k] * 100 / reference - 1) * 100 for k, reference in
                [('accuracy_median', 1.58), ('completion_median', .93)]},
            'table5_mean_x100': {head: {k: metrics[k] * 100 for k in ('accuracy', 'completion')}
                                for head, metrics in report['aggregate_by_head'].items()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(validate(args.root), indent=2))
