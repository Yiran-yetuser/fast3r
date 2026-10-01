#!/usr/bin/env python3
"""Prepare one real, GT-verified clip for GPU wiring checks, NOT full Table1."""
import hashlib
import io
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from audit_re10k_rgb_probe import first_chunk
from check_pose_data import read_re10k_metadata


def main():
    name = '3ce8f87fcfc988a8'
    _, content = first_chunk('data/re10k_test_only.zip.part')
    expected = '80046bc31d9aa850e063a92bc2c64a70310e23a28ef7d3fae56b93db60f586a3'
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError('Audited probe member changed')
    examples = torch.load(io.BytesIO(content), weights_only=True, map_location='cpu')
    selected = [example for example in examples if example['key'] == name]
    if len(selected) != 1:
        raise ValueError('Expected unique smoke clip')
    example = selected[0]
    records = read_re10k_metadata(Path('data/RealEstate10K/test')/(name+'.txt'))
    directory = Path('data/re10k_smoke/videos/test')/name
    directory.mkdir(parents=True, exist_ok=True)
    count = len(example['timestamps'])
    for i, timestamp in enumerate(example['timestamps'].tolist()):
        record = records[str(timestamp)]
        expected_camera = np.r_[record['intrinsics_normalized'], 0., 0.,
                                np.linalg.inv(record['camera_pose'])[:3].ravel()]
        if not np.allclose(example['cameras'][i].numpy(), expected_camera, atol=1e-4, rtol=0):
            raise ValueError('Mirror/official camera mismatch')
        payload = example['images'][i].numpy().tobytes()
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            if image.size != (640, 360):
                raise ValueError('Unexpected raw RGB dimensions')
        path = directory/(str(timestamp)+'.jpg')
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError('Existing smoke RGB differs; do not overwrite')
        else:
            if shutil.disk_usage(directory).free < 1024**3 + len(payload):
                raise RuntimeError('1GiB reserve violated')
            with path.open('xb') as stream:
                stream.write(payload)
    report = {'scope': 'one real clip GPU wiring smoke, not full Table1',
              'scene': name, 'original_candidate_frames': count,
              'all_candidate_frames_saved': True, 'rgb_shape_h_w': [360, 640],
              'gt_camera_verified': True, 'source_member_sha256': expected}
    path = Path('results/re10k_smoke_manifest.json')
    if path.exists():
        if json.loads(path.read_text()) != report:
            raise ValueError('Old smoke manifest differs')
    else:
        with path.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
