#!/usr/bin/env python3
"""Prepare all official test trajectories at stride20 within limited disk space.

Read official scene ZIP directories over HTTP Range, download only test-sequence
ZIP members, CRC-check them, and retain original stride-selected RGB/pose frames
with SimpleRecon-style registered depth. Temporary ZIPs created here are removed
after each sequence; existing dataset files are never overwritten. No synthetic
GT or reduced scene set. Storage is sparse; other strides/training are unsupported.
"""
import argparse
import hashlib
import io
import json
import re
import shutil
import struct
import tempfile
import zipfile
import zlib
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SCENES = ['heads', 'chess', 'fire', 'office', 'pumpkin', 'stairs', 'redkitchen']
BASE_URL = 'https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/'
REFERENCE = ('https://raw.githubusercontent.com/nianticlabs/simplerecon/'
             '477aa5b32aa1b93f53abc72828f86023b6e46ce7/data_scripts/7scenes_preprocessing.py')
REFERENCE_SHA256 = 'ac8dee029c28f600fc0e72ac79f8be19b83b7edfe9a2cdc88381abe5afe4e808'
D_TO_RGB = np.array([
    [9.9996518012567637e-01, 2.6765126468950343e-03, -7.9041012313000904e-03, -2.5558943178152542e-02],
    [-2.7409311281316700e-03, 9.9996302803027592e-01, -8.1504520778013286e-03, 1.0109636268061706e-04],
    [7.8819942130445332e-03, 8.1718328771890631e-03, 9.9993554558014031e-01, 2.0318321729487039e-03],
    [0, 0, 0, 1],
])


def register_depth(depth):
    """Vectorized equivalent of SimpleRecon's depth-to-RGB projection/z-buffer."""
    if depth.shape != (480, 640):
        raise ValueError(f'Expected original 480x640 depth, got {depth.shape}')
    depth = depth.astype(np.float32) / 1000
    yy, xx = np.indices(depth.shape, dtype=np.float64)
    valid = (depth > 0) & (depth < 100)
    z = depth[valid]
    eye = np.stack([(xx[valid] + .5 - 320) / 585 * z,
                    (yy[valid] + .5 - 240) / 585 * z, z, np.ones(len(z))])
    rgb = D_TO_RGB @ eye
    projected_z = rgb[2]
    x = np.rint(rgb[0] / projected_z * 525 + 320).astype(np.int64)
    y = np.rint(rgb[1] / projected_z * 525 + 240).astype(np.int64)
    inside = (x >= 0) & (x < 640) & (y >= 0) & (y < 480)
    registered = np.full((480, 640), 2e3, dtype=np.float32)
    np.minimum.at(registered, (y[inside], x[inside]), projected_z[inside])
    registered[registered > 1e3] = 0
    return (registered * 1000).astype(np.uint16)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


class RemoteZipReader(io.RawIOBase):
    """Small read-only HTTP seek adapter for ZIP directory/header reads only."""
    def __init__(self, session, url):
        self.session, self.url, self.position = session, url, 0
        response = session.head(url, timeout=30)
        response.raise_for_status()
        self.size = int(response.headers['Content-Length'])
        self.etag = response.headers.get('ETag')

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position + offset if whence == 1 else self.size + offset)
        if self.position < 0:
            raise ValueError('Negative seek')
        return self.position

    def get_range(self, start, length, *, stream=False):
        headers = {'Range': f'bytes={start}-{start + length - 1}', 'Accept-Encoding': 'identity'}
        if self.etag:
            headers['If-Match'] = self.etag
        response = self.session.get(self.url, headers=headers, timeout=(20, 90), stream=stream)
        response.raise_for_status()
        if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{start + length - 1}/{self.size}':
            response.close()
            raise RuntimeError('Server did not honor exact byte range; refusing full ZIP download')
        return response

    def read(self, size=-1):
        if size < 0: size = self.size - self.position
        size = min(size, self.size - self.position)
        if size <= 0: return b''
        if size > 1024**2:
            raise RuntimeError('Use streamed member download for large reads')
        with self.get_range(self.position, size) as response:
            content = response.content
        if len(content) != size: raise RuntimeError('Truncated range response')
        self.position += size
        return content


def download_member(remote, member, destination):
    if member.flag_bits & 1 or member.compress_type not in [0, 8]:
        raise ValueError('Encrypted or unsupported ZIP compression')
    remote.seek(member.header_offset)
    header = remote.read(30)
    if header[:4] != b'PK\x03\x04': raise ValueError('Invalid local ZIP header')
    name_length, extra_length = struct.unpack_from('<HH', header, 26)
    start = member.header_offset + 30 + name_length + extra_length
    decoder = zlib.decompressobj(-zlib.MAX_WBITS) if member.compress_type == 8 else None
    checksum, written, received = 0, 0, 0
    sha = hashlib.sha256()
    with remote.get_range(start, member.compress_size, stream=True) as response, destination.open('xb') as output:
        for chunk in response.iter_content(1024**2):
            received += len(chunk)
            block = decoder.decompress(chunk) if decoder else chunk
            written += len(block)
            if written > member.file_size: raise ValueError('ZIP member exceeds declared size')
            if shutil.disk_usage(destination.parent).free < len(block) + 1024**3:
                raise RuntimeError('Disk space fell below 1GiB reserve during download')
            checksum = zlib.crc32(block, checksum)
            sha.update(block)
            output.write(block)
        if decoder:
            tail = decoder.flush()
            output.write(tail)
            checksum = zlib.crc32(tail, checksum)
            sha.update(tail)
            written += len(tail)
            if not decoder.eof: raise ValueError('Incomplete compressed member')
    if received != member.compress_size or written != member.file_size or checksum != member.CRC:
        raise ValueError('Downloaded test ZIP size/CRC mismatch')
    return sha.hexdigest()


def prepare_sequence(archive, destination, stride, provenance):
    if destination.exists():
        from fast3r.data.components.spann3r_datasets.sparse_eval_frames import sparse_frame_count
        sparse_frame_count(destination, full_video=True, kf_every=stride)
        if not (destination / 'frame_inventory.json').is_file():
            raise FileExistsError(f'Refusing to overwrite non-sparse existing sequence: {destination}')
        return json.loads((destination / 'frame_inventory.json').read_text())
    with zipfile.ZipFile(archive) as zf:
        members = {}
        for member in zf.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or '..' in name.parts:
                raise ValueError('Unsafe ZIP path')
            if member.is_dir(): continue
            if name.name in members: raise ValueError('Ambiguous ZIP basename')
            members[name.name] = member
        ids = sorted(int(m.group(1)) for name in members
                     if (m := re.fullmatch(r'frame-(\d{6})\.color\.png', name)))
        if ids != list(range(len(ids))) or not ids:
            raise ValueError('Original color frame inventory is not contiguous')
        selected = list(range(0, len(ids), stride))
        required = sum(members[f'frame-{i:06d}.{suffix}'].file_size
                       for i in selected for suffix in ['color.png', 'depth.png', 'pose.txt'])
        if shutil.disk_usage(destination.parent).free < required + len(selected) * 640 * 480 * 2 + 1024**3:
            raise RuntimeError('Insufficient disk for selected frames plus 1GiB reserve')
        # Publish only after all selected frames are prepared; a failed run leaves no partial sequence.
        with tempfile.TemporaryDirectory(prefix='.prepare-', dir=destination.parent) as stage:
            work = Path(stage) / 'sequence'
            work.mkdir()
            for index in selected:
                if shutil.disk_usage(work).free < 4 * 1024**2 + 1024**3:
                    raise RuntimeError('Disk space fell below 1GiB reserve during preparation')
                prefix = f'frame-{index:06d}'
                color = zf.read(members[prefix + '.color.png'])
                if cv2.imdecode(np.frombuffer(color, np.uint8), cv2.IMREAD_COLOR) is None:
                    raise ValueError('Undecodable RGB image')
                depth = cv2.imdecode(np.frombuffer(zf.read(members[prefix + '.depth.png']), np.uint8), cv2.IMREAD_UNCHANGED)
                if depth is None or depth.dtype != np.uint16: raise ValueError('Invalid depth image')
                pose = zf.read(members[prefix + '.pose.txt'])
                if np.loadtxt(io.BytesIO(pose)).shape != (4, 4): raise ValueError('Invalid pose shape')
                (work / (prefix + '.color.png')).write_bytes(color)
                (work / (prefix + '.pose.txt')).write_bytes(pose)
                if not cv2.imwrite(str(work / (prefix + '.depth.proj.png')), register_depth(depth)):
                    raise RuntimeError('Failed to encode registered depth')
            inventory = dict(schema='7scenes_stride_storage_v1', stride=stride,
                             original_frame_count=len(ids), stored_indices=selected,
                             selected_member_crc_checked=True, **provenance)
            save_json(work / 'frame_inventory.json', inventory)
            work.rename(destination)
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('data/7_scenes_processed'))
    parser.add_argument('--manifest', type=Path, default=Path('results/7scenes_data_manifest.json'))
    parser.add_argument('--scenes', nargs='+', choices=SCENES, default=SCENES)
    parser.add_argument('--stride', type=int, default=20)
    args = parser.parse_args()
    if args.stride < 1: parser.error('Stride must be positive')
    args.output_root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    reference = session.get(REFERENCE, timeout=30)
    reference.raise_for_status()
    reference_sha = hashlib.sha256(reference.content).hexdigest()
    if reference_sha != REFERENCE_SHA256:
        raise ValueError('Pinned preprocessing reference checksum changed')
    report = {'status': 'preparing', 'scenes_expected': args.scenes, 'stride': args.stride,
              'preprocessing_source': REFERENCE, 'preprocessing_source_sha256': reference_sha,
              'preprocessing': 'Vectorized SimpleRecon depth-to-RGB registration; not TSDF raycasting',
              'storage': 'Only exact original stride-selected test frames; no training frames',
              'license': 'Microsoft Research 7-Scenes, non-commercial academic reproduction', 'sequences': []}
    save_json(args.manifest, report)
    for scene in args.scenes:
        url = BASE_URL + scene + '.zip'
        remote = RemoteZipReader(session, url)
        with zipfile.ZipFile(remote) as zf:
            split = zf.read(f'{scene}/TestSplit.txt')
            lines = split.decode().splitlines()
            if not lines or any(not re.fullmatch(r'sequence\d+', line.strip()) for line in lines):
                raise ValueError('Unexpected official test split')
            sequences = [f'seq-{int(line.strip()[8:]):02d}' for line in lines]
            scene_dir = args.output_root / scene
            scene_dir.mkdir(exist_ok=True)
            split_path = scene_dir / 'TestSplit.txt'
            if split_path.exists() and split_path.read_bytes() != split:
                raise FileExistsError('Existing test split differs from official release')
            if not split_path.exists(): split_path.write_bytes(split)
            for sequence in sequences:
                destination = scene_dir / sequence
                if destination.exists():
                    inventory = prepare_sequence(None, destination, args.stride, {})
                else:
                    member = zf.getinfo(f'{scene}/{sequence}.zip')
                    if shutil.disk_usage(scene_dir).free < member.file_size + 1024**3:
                        raise RuntimeError(f'Need temporary test ZIP {member.file_size} bytes plus 1GiB reserve; exact extracted budget checked next')
                    print(f'DOWNLOAD {scene}/{sequence}: {member.compress_size} bytes (test sequence only)', flush=True)
                    with tempfile.TemporaryDirectory(prefix='.download-', dir=scene_dir) as temporary:
                        archive = Path(temporary) / 'sequence.zip'
                        sha = download_member(remote, member, archive)
                        provenance = dict(source_url=url, source_etag=remote.etag,
                                          source_member=member.filename, sequence_zip_sha256=sha,
                                          sequence_zip_crc_checked=True,
                                          preprocessing_source=REFERENCE, preprocessing_source_sha256=reference_sha)
                        inventory = prepare_sequence(archive, destination, args.stride, provenance)
                report['sequences'].append(dict(scene=scene, sequence=sequence, **inventory))
                save_json(args.manifest, report)
                print(f'PREPARED {scene}/{sequence}: {len(inventory["stored_indices"])} views', flush=True)
        remote.close()
    report['status'] = 'prepared' if set(args.scenes) == set(SCENES) else 'prepared_subset'
    report['trajectory_count'] = len(report['sequences'])
    save_json(args.manifest, report)
    print('DATA PREPARATION COMPLETE', report['trajectory_count'], flush=True)


if __name__ == '__main__':
    main()
