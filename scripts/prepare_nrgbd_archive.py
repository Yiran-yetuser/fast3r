#!/usr/bin/env python3
"""Validate and extract the official Neural RGB-D archive for the published loader."""
import json
import shutil
import stat
import zipfile
from pathlib import Path


def main():
    archive = Path('data/neural_rgbd_data.zip.part')
    expected_bytes = 7785287298  # Official Content-Length checked 2026-10-01.
    if archive.stat().st_size != expected_bytes:
        raise ValueError('Archive is not the expected complete download')
    destination = Path('data/nrgbd_extracted')
    destination.mkdir(exist_ok=True)
    boundary = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        for member in members:
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(boundary):
                raise ValueError('Archive member escapes extraction directory')
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError('Unexpected symlink in dataset archive')
        required = sum(member.file_size for member in members)
        if shutil.disk_usage(destination).free < required + 1024**3:
            raise RuntimeError(f'Insufficient space: need {required} bytes plus 1GiB reserve')
        print(f'Extracting {len(members)} members, {required} bytes; ZIP CRC checked during extraction', flush=True)
        zf.extractall(destination)
    poses = sorted(destination.rglob('poses.txt'))
    scene_dirs = [p.parent for p in poses if (p.parent/'images/img0.png').is_file()
                  and (p.parent/'depth/depth0.png').is_file()]
    if not scene_dirs:
        raise RuntimeError('Archive layout does not match official NRGBD loader; inspect extracted data before proceeding')
    roots = {p.parent.resolve() for p in scene_dirs}
    if len(roots) != 1:
        raise RuntimeError('Multiple candidate dataset roots')
    root = roots.pop()
    target = Path('data/neural_rgbd')
    if target.exists() or target.is_symlink():
        if target.resolve() != root:
            raise FileExistsError('Existing data/neural_rgbd refers to other data')
    else:
        target.symlink_to(root, target_is_directory=True)
    manifest = {'source': 'https://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip',
                'archive_bytes': expected_bytes, 'uncompressed_bytes': required,
                'root': str(root), 'scene_names': [p.name for p in scene_dirs],
                'status': 'prepared', 'zip_crc_checked': True}
    Path('results').mkdir(exist_ok=True)
    Path('results/nrgbd_data_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
