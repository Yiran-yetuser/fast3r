#!/usr/bin/env python3
"""Read-only, fail-closed data preflight for paper section 4.2 (no model/GPU).

This checks availability, not equivalence to the paper's unpublished frame draws.
It never downloads data, creates dataset files, or skips failed scenes silently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_split(path):
    names = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("Split must be nonempty and contain unique identifiers")
    if any(Path(name).name != name or name in ('.', '..') for name in names):
        raise ValueError("Unsafe split identifier")
    return names


def read_re10k_metadata(path):
    """19 columns: timestamp, six intrinsic fields, world-to-camera 3x4.

    Pixel intrinsics use fields 1:5; fields 5:7 are unused by upstream eval.
    The returned pose is inverted to camera-to-world, as required by Fast3R.
    """
    lines = Path(path).read_text().splitlines()
    if len(lines) < 2 or not lines[0].strip():
        raise ValueError("Missing video URL or camera records")
    records = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        columns = line.split()
        if len(columns) != 19:
            raise ValueError("Expected exactly 19 metadata columns")
        timestamp = columns[0]
        if not timestamp.isdigit() or timestamp in records:
            raise ValueError("Invalid or duplicate timestamp")
        values = np.array(columns[1:], dtype=np.float64)
        if not np.isfinite(values).all() or min(values[:2]) <= 0:
            raise ValueError("Nonfinite camera parameters or nonpositive focal length")
        w2c = np.eye(4)
        w2c[:3] = values[6:].reshape(3, 4)
        c2w = np.linalg.inv(w2c)
        if not np.isfinite(c2w).all():
            raise ValueError("Invalid camera pose")
        records[timestamp] = {'intrinsics_normalized': values[:4], 'camera_pose': c2w}
    if not records:
        raise ValueError("No camera records")
    return records


def check_image(path, decode):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty file: {path}")
    if decode:
        with Image.open(path) as image:
            image.load()


def check_re10k(video_root, metadata_root, split_file, num_views=10, decode=True):
    names = read_split(split_file)
    if num_views < 2:
        raise ValueError("At least two distinct views are required")
    issues, frame_counts = [], {}
    for name in names:
        try:
            folder = Path(video_root) / name
            metadata = Path(metadata_root) / (name + '.txt')
            if not folder.is_dir() or not metadata.is_file():
                raise ValueError("Missing video folder or pose TXT")
            records = read_re10k_metadata(metadata)
            frames = sorted(folder.glob('*.jpg'))
            if len(frames) < num_views:
                raise ValueError(f"Expected >= {num_views} JPG views, got {len(frames)}")
            for frame in frames:
                if frame.stem not in records:
                    raise ValueError(f"JPG timestamp has no GT record: {frame.name}")
                check_image(frame, decode)
            frame_counts[name] = len(frames)
        except (OSError, ValueError, np.linalg.LinAlgError) as error:
            issues.append({'scene': name, 'reason': str(error)})
    return {'dataset': 're10k', 'status': 'ready' if not issues else 'incomplete',
            'expected_scene_count': len(names), 'ready_scene_count': len(frame_counts),
            'split_sha256': hashlib.sha256(Path(split_file).read_bytes()).hexdigest(),
            'minimum_distinct_views': num_views, 'frame_counts': frame_counts,
            'image_decoding_requested': decode,
            'image_decoding_checked': bool(decode and frame_counts), 'issues': issues,
            'paper_protocol_equivalence': 'unverified; original frame draws not supplied'}


def check_co3d(root, decode=True):
    root = Path(root)
    split = root / 'selected_seqs_test.json'
    if not split.is_file():
        return {'dataset': 'co3d', 'status': 'incomplete', 'issues': [
            {'reason': f'Missing prescribed processed split: {split}'}]}
    selected = json.loads(split.read_text())
    if not isinstance(selected, dict) or not selected:
        raise ValueError('Empty CO3D split')
    issues, frame_counts = [], {}
    for category, scenes in selected.items():
        for scene, indices in scenes.items():
            label = category + '/' + scene
            try:
                if (Path(category).name != category or Path(scene).name != scene
                        or category in ('.', '..') or scene in ('.', '..')):
                    raise ValueError('Unsafe scene identifier')
                if len(indices) < 10 or len(indices) != len(set(indices)):
                    raise ValueError('Expected >=10 unique frame indices')
                for index in indices:
                    if not isinstance(index, int) or index < 0:
                        raise ValueError('Invalid frame index')
                    stem = f'frame{index:06d}'
                    directory = root / category / scene
                    for path in [directory / 'images' / (stem + '.jpg'),
                                 directory / 'depths' / (stem + '.jpg.geometric.png'),
                                 directory / 'masks' / (stem + '.png')]:
                        check_image(path, decode)
                    with np.load(directory / 'images' / (stem + '.npz')) as data:
                        for key, shape in [('camera_pose', (4, 4)),
                                           ('camera_intrinsics', (3, 3))]:
                            if data[key].shape != shape or not np.isfinite(data[key]).all():
                                raise ValueError(f'Invalid {key}')
                        if not np.isfinite(data['maximum_depth']).all():
                            raise ValueError('Invalid maximum_depth')
                frame_counts[label] = len(indices)
            except (OSError, ValueError, KeyError) as error:
                issues.append({'scene': label, 'reason': str(error)})
    count = sum(len(scenes) for scenes in selected.values())
    if count == 0:
        raise ValueError('No CO3D sequences')
    return {'dataset': 'co3d', 'status': 'ready' if not issues else 'incomplete',
            'expected_scene_count': count, 'ready_scene_count': len(frame_counts),
            'split_sha256': hashlib.sha256(split.read_bytes()).hexdigest(),
            'frame_counts': frame_counts, 'image_decoding_requested': decode,
            'image_decoding_checked': bool(decode and frame_counts), 'issues': issues,
            'paper_protocol_equivalence': 'unverified; selection provenance still required'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['co3d', 're10k'], required=True)
    parser.add_argument('--data-root', type=Path, required=True,
                        help='CO3D processed root or RE10K videos/test root')
    parser.add_argument('--metadata-root', type=Path, help='RE10K pose test TXT directory')
    parser.add_argument('--split-file', type=Path, default=Path('scripts/re10k_test_1800.txt'))
    parser.add_argument('--skip-decode', action='store_true', help='Only file/metadata checks')
    args = parser.parse_args()
    if args.dataset == 're10k':
        if args.metadata_root is None:
            parser.error('--metadata-root is required for re10k')
        report = check_re10k(args.data_root, args.metadata_root, args.split_file,
                             decode=not args.skip_decode)
    else:
        report = check_co3d(args.data_root, decode=not args.skip_decode)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report['status'] == 'ready' else 2)


if __name__ == '__main__':
    main()
