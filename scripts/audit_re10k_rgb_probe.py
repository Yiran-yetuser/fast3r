#!/usr/bin/env python3
"""Audit the first fully downloaded RGB chunk, without network or unsafe pickle.

This bounded probe establishes format only, NOT full prescribed split coverage.
"""
import argparse
import hashlib
import io
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from check_pose_data import read_re10k_metadata, read_split


def first_chunk(path):
    with Path(path).open('rb') as stream:
        while True:
            header = stream.read(30)
            if len(header) != 30 or header[:4] != b'PK\x03\x04':
                raise ValueError('Missing/truncated local ZIP header')
            fields = struct.unpack('<4s5H3I2H', header)
            flags, method, crc, compressed, uncompressed = fields[2], fields[3], *fields[6:9]
            name = stream.read(fields[-2]).decode('utf-8')
            stream.read(fields[-1])
            if flags & 9 or method not in (0, 8):
                raise ValueError('Unsupported encrypted/data-descriptor ZIP member')
            if compressed > 256*1024**2 or uncompressed > 256*1024**2:
                raise ValueError('Probe chunk exceeds bounded memory budget')
            if not name.endswith('.torch'):
                stream.seek(compressed, 1)
                continue
            payload = stream.read(compressed)
            if len(payload) != compressed:
                raise ValueError('First RGB chunk still downloading')
            if method == 8:
                decoder = zlib.decompressobj(-15)
                content = decoder.decompress(payload, uncompressed+1)
                if not decoder.eof or decoder.unconsumed_tail:
                    raise ValueError('Incomplete/oversized deflate stream')
            else:
                content = payload
            if len(content) != uncompressed or zlib.crc32(content) != crc:
                raise ValueError('RGB probe ZIP member CRC/length mismatch')
            return name, content


def audit(archive, metadata_root, split_file):
    name, content = first_chunk(archive)
    examples = torch.load(io.BytesIO(content), weights_only=True, map_location='cpu')
    if not isinstance(examples, list) or not examples:
        raise ValueError('Expected nonempty list of safe tensor examples')
    prescribed = set(read_split(split_file))
    shapes, matched, issues = set(), [], []
    keys = set()
    for example in examples:
        key = example['key']
        if not isinstance(key, str) or key in keys:
            raise ValueError('Invalid/duplicate probe clip key')
        keys.add(key)
        timestamps = example['timestamps'].tolist()
        cameras = example['cameras'].numpy()
        images = example['images']
        if cameras.shape != (len(timestamps), 18) or len(images) != len(timestamps):
            raise ValueError('Inconsistent chunk frame/camera arrays')
        if not np.isfinite(cameras).all() or len(set(timestamps)) != len(timestamps):
            raise ValueError('Invalid cameras/duplicate timestamps')
        # Decode all frames in this one chunk; no files extracted or overwritten.
        for image in images:
            if image.dtype != torch.uint8 or image.ndim != 1:
                raise ValueError('Expected encoded uint8 image')
            with Image.open(io.BytesIO(image.numpy().tobytes())) as decoded:
                decoded.load()
                shapes.add((decoded.height, decoded.width))
        if key not in prescribed:
            continue
        records = read_re10k_metadata(Path(metadata_root)/(key+'.txt'))
        max_delta = 0.
        for i, timestamp in enumerate(timestamps):
            record = records.get(str(timestamp))
            if record is None:
                issues.append({'scene': key, 'reason': 'timestamp absent from official GT'})
                break
            normalized = np.r_[record['intrinsics_normalized'], 0., 0.]
            expected = np.r_[normalized, np.linalg.inv(record['camera_pose'])[:3].ravel()]
            delta = float(np.max(np.abs(cameras[i]-expected)))
            max_delta = max(max_delta, delta)
        matched.append({'scene': key, 'frames': len(timestamps), 'maximum_camera_abs_delta': max_delta})
        if max_delta > 1e-4:
            issues.append({'scene': key, 'reason': 'mirror camera differs from official GT'})
    return {'status': 'bounded_format_probe_only', 'zip_member': name,
            'member_sha256': hashlib.sha256(content).hexdigest(), 'member_crc_verified': True,
            'safe_weights_only_load': True, 'probe_clip_count': len(examples),
            'decoded_image_shapes_h_w': sorted(shapes), 'prescribed_matches': matched,
            'issues': issues, 'full_split_coverage_verified': False,
            'paper_equivalence': 'not established; first chunk is not a complete dataset audit'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('data/re10k_test_only.zip.part'))
    parser.add_argument('--metadata-root', type=Path, default=Path('data/RealEstate10K/test'))
    parser.add_argument('--split-file', type=Path, default=Path('scripts/re10k_test_1800.txt'))
    parser.add_argument('--output-json', type=Path, default=Path('results/re10k_rgb_probe_20261002.json'))
    args = parser.parse_args()
    report = audit(args.archive, args.metadata_root, args.split_file)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
