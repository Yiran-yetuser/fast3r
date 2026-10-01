#!/usr/bin/env python3
"""Audit/extract prescribed RGB from the author archive; never drop missing IDs.

Covered scenes may be prepared while the prescribed split remains incomplete.
All original official timestamps are required per covered clip. No model runs,
network, unrestricted pickle, or whole-archive extraction is performed.
"""
import argparse
import hashlib
import io
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from check_pose_data import read_re10k_metadata, read_split
from fast3r_hf_re10k_pose_eval import sha

RESERVE = 1024**3
MAX_MEMBER = 256*1024**2


def save_identical(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Existing report differs; preserve history: {path}')
    else:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')


def budget(root, additional=0):
    if shutil.disk_usage(root).free < RESERVE + additional:
        raise RuntimeError('RGB preparation would violate 1GiB reserve')


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate JSON index key')
        value[key] = item
    return value


def audit_index(archive, names, metadata_root):
    entries = archive.infolist()
    if len({entry.filename for entry in entries}) != len(entries):
        raise ValueError('Duplicate ZIP member names')
    member = 're10k/test/index.json'
    if archive.getinfo(member).file_size > 10*1024**2:
        raise ValueError('Oversized index')
    raw = archive.read(member)  # zipfile checks this member CRC
    index = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(index, dict) or not index:
        raise ValueError('Invalid index')
    for key, chunk in index.items():
        if not re.fullmatch(r'[0-9a-f]{16}', key) or not isinstance(chunk, str) or not re.fullmatch(r'[0-9]{6}\.torch', chunk):
            raise ValueError('Unsafe/invalid scene or chunk index')
        entry = archive.getinfo('re10k/test/'+chunk)
        if entry.file_size > MAX_MEMBER:
            raise ValueError('RGB chunk exceeds bounded memory limit')
    missing = sorted(set(names)-set(index))
    covered = sorted(set(names)&set(index))
    chunks = sorted({index[name] for name in covered})
    report = {'status': 'index_incomplete' if missing else 'index_complete_not_rgb_verified',
              'paper_mapping': 'section 4.2 / Table 1 data preparation only',
              'expected_scene_count': len(names), 'mirror_index_scene_count': len(index),
              'covered_scene_count': len(covered), 'missing_scene_count': len(missing),
              'covered_scene_ids': covered, 'missing_scene_ids': missing,
              'missing_video_urls': {name: (Path(metadata_root)/(name+'.txt')).read_text().splitlines()[0]
                                     for name in missing},
              'index_member': member, 'index_sha256': hashlib.sha256(raw).hexdigest(),
              'index_crc_verified': True, 'selected_chunk_count': len(chunks),
              'selected_chunk_uncompressed_bytes': sum(archive.getinfo('re10k/test/'+c).file_size for c in chunks),
              'full_scene_coverage_verified': not missing,
              'rgb_gt_equivalence_verified': False, 'formal_pose_metrics_available': False}
    return index, report


def validate_example(example, metadata):
    timestamps = example['timestamps'].tolist()
    if any(not isinstance(t, int) or t < 0 for t in timestamps) or len(set(timestamps)) != len(timestamps):
        raise ValueError('Invalid/duplicate timestamps')
    if set(map(str, timestamps)) != set(metadata) or len(timestamps) < 10:
        raise ValueError('Mirror does not preserve complete official candidate frame inventory')
    cameras = example['cameras'].numpy()
    images = example['images']
    if cameras.shape != (len(timestamps), 18) or len(images) != len(timestamps) or not np.isfinite(cameras).all():
        raise ValueError('Invalid camera/image arrays')
    expected = np.array([np.r_[metadata[str(t)]['intrinsics_normalized'], 0., 0.,
                                      np.linalg.inv(metadata[str(t)]['camera_pose'])[:3].ravel()]
                         for t in timestamps])
    delta = float(np.max(np.abs(cameras-expected)))
    if delta > 1e-4:
        raise ValueError('Mirror cameras differ from official GT')
    frames, payloads = [], []
    for t, image in zip(timestamps, images):
        if image.dtype != torch.uint8 or image.ndim != 1:
            raise ValueError('Expected encoded uint8 JPEG')
        payload = image.numpy().tobytes()
        with Image.open(io.BytesIO(payload)) as decoded:
            decoded.load()
            if decoded.format != 'JPEG' or min(decoded.size) <= 0 or max(decoded.size) > 8192:
                raise ValueError('Invalid original RGB format/dimensions')
            shape = [decoded.height, decoded.width]
        frames.append({'timestamp': str(t), 'filename': str(t)+'.jpg', 'bytes': len(payload),
                       'sha256': hashlib.sha256(payload).hexdigest(), 'shape_h_w': shape})
        payloads.append(payload)
    if len({tuple(f['shape_h_w']) for f in frames}) != 1:
        raise ValueError('Inconsistent original dimensions within a clip')
    return frames, payloads, delta


def write_scene(root, name, frames, payloads):
    folder = Path(root)/name
    folder.mkdir(parents=True, exist_ok=True)
    expected = {frame['filename'] for frame in frames}
    if {p.name for p in folder.glob('*.jpg')}-expected:
        raise ValueError('Existing directory has extra frames; do not change sampling population')
    budget(folder, sum(len(p) for f,p in zip(frames,payloads) if not (folder/f['filename']).exists()))
    for frame, payload in zip(frames, payloads):
        path = folder/frame['filename']
        if path.exists():
            if sha(path) != frame['sha256']:
                raise ValueError(f'Existing RGB differs; do not overwrite: {path}')
        else:
            budget(folder, len(payload))
            with path.open('xb') as stream:
                stream.write(payload)


def prepare(args):
    names = read_split(args.split_file)
    source = json.loads(args.archive_manifest.read_text())
    if args.archive.stat().st_size != source['archive_bytes']:
        raise ValueError('Archive size differs from completed download manifest')
    args.data_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        index, report = audit_index(archive, names, args.metadata_root)
        report.update({'archive_sha256': source['archive_sha256'], 'split_sha256': sha(args.split_file)})
        save_identical(args.index_report, report)
        print(f"INDEX: {report['covered_scene_count']}/{len(names)} covered; {report['missing_scene_count']} missing; no full benchmark", flush=True)
        if args.index_only:
            return report
        # Conservative upper bound: all selected chunk bytes, though only prescribed RGB is saved.
        existing_bytes = sum(p.stat().st_size for name in report['covered_scene_ids']
                             for p in (args.data_root/name).glob('*.jpg'))
        budget(args.data_root, max(0, report['selected_chunk_uncompressed_bytes']-existing_bytes) + MAX_MEMBER)
        if sha(args.archive) != source['archive_sha256']:
            raise ValueError('Archive SHA changed; do not extract')
        selected = set(report['covered_scene_ids'])
        rows, chunks = {}, {}
        for chunk in sorted({index[name] for name in selected}):
            member = 're10k/test/'+chunk
            content = archive.read(member)  # CRC validated by zipfile for every selected member
            digest = hashlib.sha256(content).hexdigest()
            examples = torch.load(io.BytesIO(content), weights_only=True, map_location='cpu')
            if not isinstance(examples, list) or not examples:
                raise ValueError('Expected nonempty safe tensor example list')
            keys = [e['key'] for e in examples]
            if len(keys) != len(set(keys)) or any(index.get(k) != chunk for k in keys):
                raise ValueError('Chunk scene keys disagree with index')
            required = {name for name in selected if index[name] == chunk}
            if not required.issubset(keys):
                raise ValueError('Indexed prescribed scene missing from chunk')
            for example in examples:
                name = example['key']
                if name not in selected:
                    continue
                if name in rows:
                    raise ValueError('Duplicate prescribed scene')
                metadata_path = args.metadata_root/(name+'.txt')
                try:
                    frames, payloads, delta = validate_example(example, read_re10k_metadata(metadata_path))
                except ValueError as error:
                    raise ValueError(f'{name}: {error}') from error
                inventory = {'scene': name, 'source_member': member, 'member_sha256': digest,
                             'metadata_sha256': sha(metadata_path), 'frames': frames,
                             'all_official_candidate_frames_preserved': True,
                             'maximum_camera_abs_delta': delta, 'decoded_rgb_shape_h_w': frames[0]['shape_h_w']}
                write_scene(args.data_root, name, frames, payloads)
                # Preserve earlier fixed-dimension audit inventories; do not overwrite history.
                inventory_path = args.data_root/name/'.fast3r_rgb_inventory_v2.json'
                save_identical(inventory_path, inventory)
                rows[name] = {'frames': len(frames), 'maximum_camera_abs_delta': delta,
                              'inventory_sha256': sha(inventory_path),
                              'rgb_shape_h_w': frames[0]['shape_h_w']}
                print(f"PREPARED {len(rows)}/{len(selected)} {name}: {len(frames)} frames", flush=True)
            chunks[member] = digest
            del examples, content
        if set(rows) != selected:
            raise ValueError('Prepared set differs from all covered prescribed scenes')
        result = {'status': 'covered_prepared_split_incomplete' if report['missing_scene_ids'] else 'prepared',
                  'paper_mapping': 'section 4.2 / Table 1 data preparation only',
                  'archive_sha256': source['archive_sha256'], 'split_sha256': sha(args.split_file),
                  'expected_scene_count': len(names), 'prepared_scene_count': len(rows),
                  'frame_count': sum(r['frames'] for r in rows.values()), 'scenes': rows,
                  'missing_scene_ids': report['missing_scene_ids'], 'selected_chunk_sha256': chunks,
                  'selected_chunk_crc_verified': True, 'safe_weights_only_load': True,
                  'all_official_candidate_frames_preserved_for_prepared_scenes': True,
                  'full_scene_coverage_verified': not report['missing_scene_ids'],
                  'rgb_gt_camera_verified_for_prepared_scenes': True,
                  'formal_pose_metrics_available': False, 'paper_equivalence': 'unverified checkpoint/draw mapping'}
        save_identical(args.output_json, result)
        print(f"RGB PREPARATION COMPLETE: {len(rows)}/{len(names)}; missing IDs explicitly retained", flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('data/re10k_test_only.zip.part'))
    parser.add_argument('--archive-manifest', type=Path, default=Path('results/re10k_rgb_archive_manifest.json'))
    parser.add_argument('--split-file', type=Path, default=Path('scripts/re10k_test_1800.txt'))
    parser.add_argument('--metadata-root', type=Path, default=Path('data/RealEstate10K/test'))
    parser.add_argument('--data-root', type=Path, default=Path('data/RealEstate10K/videos/test'))
    parser.add_argument('--index-report', type=Path, default=Path('results/re10k_rgb_index_audit.json'))
    parser.add_argument('--output-json', type=Path, default=Path('results/re10k_rgb_prepared_manifest.json'))
    parser.add_argument('--index-only', action='store_true')
    args = parser.parse_args()
    import fcntl
    args.data_root.mkdir(parents=True, exist_ok=True)
    with (args.data_root/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(args)


if __name__ == '__main__':
    main()
