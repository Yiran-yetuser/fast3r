"""Validate explicit sparse storage of an otherwise unchanged full-video stride."""
import json
from pathlib import Path


def sparse_frame_count(directory, *, full_video, kf_every):
    inventory = Path(directory) / 'frame_inventory.json'
    if not inventory.exists():
        return None
    metadata = json.loads(inventory.read_text())
    if metadata.get('schema') != '7scenes_stride_storage_v1':
        raise ValueError('Unknown sparse frame inventory schema')
    if not full_video or kf_every != metadata['stride']:
        raise ValueError('Sparse storage supports only its recorded full-video stride')
    count = metadata['original_frame_count']
    if not isinstance(count, int) or count < 1 or kf_every < 1:
        raise ValueError('Invalid original frame count or stride')
    expected = list(range(0, count, kf_every))
    if metadata['stored_indices'] != expected:
        raise ValueError('Sparse stored indices do not match original stride sampling')
    for index in expected:
        for suffix in ['color.png', 'depth.proj.png', 'pose.txt']:
            path = Path(directory) / f'frame-{index:06d}.{suffix}'
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f'Missing or empty sparse evaluation file: {path}')
    return count
