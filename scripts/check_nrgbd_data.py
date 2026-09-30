#!/usr/bin/env python3
"""Check extracted NRGBD without its ZIP/manifest (not PNG decoding or CRC)."""
import argparse
from pathlib import Path


# Frame counts from the locally CRC-verified full official release.
EXPECTED_FRAMES = {
    'breakfast_room': 1167, 'complete_kitchen': 1211, 'green_room': 1442,
    'grey_white_room': 1493, 'kitchen': 1517, 'morning_apartment': 920,
    'staircase': 1149, 'thin_geometry': 395, 'whiteroom': 1676,
}


def validate_data(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f'Missing extracted dataset directory: {root}')
    for scene, count in EXPECTED_FRAMES.items():
        directory = root / scene
        for folder, prefix in [('images', 'img'), ('depth', 'depth')]:
            files = {p.name: p for p in (directory / folder).glob('*.png')}
            expected = {f'{prefix}{i}.png' for i in range(count)}
            if files.keys() != expected:
                raise ValueError(f'{scene}/{folder}: expected {count} contiguous frames, '
                                 f'found {len(files)}; missing={len(expected - files.keys())}, '
                                 f'unexpected={len(files.keys() - expected)}')
            if any(not p.is_file() or p.stat().st_size == 0 for p in files.values()):
                raise ValueError(f'{scene}/{folder}: empty or invalid frame file')
        poses = directory / 'poses.txt'
        if not poses.is_file():
            raise ValueError(f'{scene}: missing poses.txt')
        rows = [line.split() for line in poses.read_text().splitlines() if line.strip()]
        if len(rows) != count * 4 or any(len(row) != 4 for row in rows):
            raise ValueError(f'{scene}: poses.txt must contain {count} 4x4 poses')
        # Nonfinite values are allowed: the official loader filters invalid poses.
        for row in rows:
            for value in row:
                float(value)
    return dict(EXPECTED_FRAMES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data/neural_rgbd')
    args = parser.parse_args()
    try:
        scenes = validate_data(args.data_root)
    except (OSError, ValueError) as error:
        parser.exit(1, f'NRGBD data check failed: {error}\n')
    print(f'NRGBD data ready: {len(scenes)} scenes, {sum(scenes.values())} paired frames')


if __name__ == '__main__':
    main()
