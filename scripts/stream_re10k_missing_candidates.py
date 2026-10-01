#!/usr/bin/env python3
"""Bounded, staged RGB source audit, NOT a Table 1 evaluation or data promotion.

The gzip must be traversed sequentially; Range retries resume transport within
one process. A process restart replays the stream and verifies existing PNGs.
Only the 76 missing prescribed IDs are stored, separately from prepared JPEGs.
"""
import argparse
import fcntl
import gzip
import hashlib
import io
import json
import re
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path, PurePosixPath

from PIL import Image

from check_pose_data import read_re10k_metadata
from prepare_re10k_rgb_from_archive import save_identical, sha

REVISION = 'ea8d2427de59817b2f66f17b26276339841eb142'
URL = ('https://huggingface.co/datasets/DavidYan2001/RealEstate10K/resolve/'
       + REVISION + '/dataset/test.tar.gz')
TOTAL = 205763619478
EXPECTED_SHA = 'f6055cd8ea1ccce642482ca623f98c21af1c449210760d23d78a43b09e546523'
PREFIX = 'mnt/datamnt/weilong/RealEstate10K_Downloader/RealEstate10K/dataset/test'
RESERVE = 1024**3
MAX_IMAGE = 8 * 1024**2


class ProbeLimit(Exception):
    """Intentional bounded prefix, never interpreted as complete archive."""


class RangeReader(io.RawIOBase):
    def __init__(self, url, total, limit, opener=urllib.request.urlopen):
        self.url, self.total, self.limit = url, total, min(limit, total)
        self.opener = opener
        self.offset = self.end = 0
        self.response = None
        self.etag = None
        self.digest = hashlib.sha256()
        self.next_log = time.monotonic() + 60

    def readable(self):
        return True

    def close(self):
        if self.response is not None:
            self.response.close()
        super().close()

    def read(self, size=-1):
        if size < 0:
            raise ValueError('Unbounded read prohibited')
        if size == 0 or self.offset == self.total:
            return b''
        if self.offset == self.limit:
            raise ProbeLimit('Compressed prefix budget reached')
        for attempt in range(4):
            try:
                if self.response is None:
                    self.end = min(self.offset + 256 * 1024**2, self.limit)
                    request = urllib.request.Request(
                        self.url + f'?download=true&range_start={self.offset}',
                        headers={'Range': f'bytes={self.offset}-{self.end-1}',
                                 'Accept-Encoding': 'identity'})
                    self.response = self.opener(request, timeout=45)
                    expected = f'bytes {self.offset}-{self.end-1}/{self.total}'
                    if (self.response.status != 206
                            or self.response.headers.get('Content-Range') != expected
                            or self.response.headers.get('Content-Encoding', 'identity') != 'identity'):
                        raise ValueError('Server did not honor exact pinned range')
                    tag = self.response.headers.get('ETag')
                    if not tag or (self.etag is not None and tag != self.etag):
                        raise ValueError('Missing or changed response ETag')
                    self.etag = tag
                data = self.response.read(min(size, 1024**2, self.end-self.offset))
                if not data:
                    raise OSError('Premature HTTP EOF')
                self.offset += len(data)
                self.digest.update(data)
                if self.offset == self.end:
                    self.response.close()
                    self.response = None
                if time.monotonic() >= self.next_log:
                    print(f'Transport {self.offset}/{self.total} compressed bytes', flush=True)
                    self.next_log = time.monotonic() + 60
                return data
            except OSError:
                if self.response is not None:
                    self.response.close()
                    self.response = None
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt+1))
        raise AssertionError('unreachable')


def candidate_path(member):
    path = PurePosixPath(member.name)
    if path.is_absolute() or '..' in path.parts or '\\' in member.name:
        raise ValueError('Unsafe tar path')
    if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
        raise ValueError('Tar links/special files prohibited')
    if member.isdir():
        return None
    parts = path.parts
    prefix = PurePosixPath(PREFIX).parts
    if parts[:len(prefix)] != prefix or len(parts) != len(prefix)+2:
        return None
    scene, filename = parts[-2:]
    match = re.fullmatch(r'([0-9]+)\.png', filename)
    if not re.fullmatch(r'[0-9a-f]{16}', scene) or match is None:
        raise ValueError('Unexpected scene/image naming in source')
    if not 0 < member.size <= MAX_IMAGE:
        raise ValueError('Image size exceeds bounded member budget')
    return scene, match.group(1)


def store_png(root, scene, timestamp, payload, cap):
    with Image.open(io.BytesIO(payload)) as image:
        if image.format != 'PNG' or max(image.size) > 8192 or min(image.size) < 1:
            raise ValueError('Expected bounded PNG dimensions')
        image.load()
        size = list(image.size)
    folder = Path(root)/scene
    if Path(root).is_symlink() or folder.is_symlink():
        raise ValueError('Staging symlink prohibited')
    folder.mkdir(parents=True, exist_ok=True)
    target = folder/(timestamp+'.png')
    digest = hashlib.sha256(payload).hexdigest()
    if target.exists() or target.is_symlink():
        if target.is_symlink() or sha(target) != digest:
            raise ValueError('Existing staged RGB differs; preserved without overwrite')
    else:
        if shutil.disk_usage(root).free < RESERVE + len(payload):
            raise RuntimeError('Must preserve at least 1 GiB free')
        if cap[0] + len(payload) > cap[1]:
            raise RuntimeError('Selected RGB storage cap reached')
        with target.open('xb') as out:
            out.write(payload)
        cap[0] += len(payload)
    return {'bytes': len(payload), 'sha256': digest, 'size_wh': size}


def audit(reader, root, metadata, missing, cap):
    seen, files, extras = set(), {}, {}
    complete = False
    with gzip.GzipFile(fileobj=reader, mode='rb') as stream:
        try:
            with tarfile.open(fileobj=stream, mode='r|') as archive:
                # Streaming mode otherwise retains every TarInfo in memory.
                for member in archive:
                    archive.members.clear()
                    identity = candidate_path(member)
                    if identity is None:
                        continue
                    scene, timestamp = identity
                    seen.add(scene)
                    if scene not in missing:
                        continue
                    frames = files.setdefault(scene, {})
                    if timestamp in frames:
                        raise ValueError('Duplicate selected tar frame')
                    if timestamp not in metadata[scene]:
                        extras.setdefault(scene, []).append(timestamp)
                        continue
                    payload = archive.extractfile(member).read(MAX_IMAGE+1)
                    if len(payload) != member.size:
                        raise ValueError('Truncated selected PNG')
                    frames[timestamp] = store_png(root, scene, timestamp, payload, cap)
                    if len(frames) == len(metadata[scene]):
                        print(f'Candidate GT timestamp set seen: {scene} ({len(frames)} frames); '
                              'full source SHA pending', flush=True)
            # Check gzip trailer/CRC and hash the ENTIRE compressed source.
            while stream.read(1024**2):
                pass
            if reader.offset != reader.total or reader.digest.hexdigest() != EXPECTED_SHA:
                raise ValueError('Full pinned archive SHA/length mismatch')
            complete = True
        except ProbeLimit:
            pass
    matched = sorted(s for s in files if set(files[s]) == set(metadata[s]) and not extras.get(s))
    return {'status': 'source_scanned_candidates_not_promoted' if complete else 'bounded_prefix_only',
            'revision': REVISION, 'source_url': URL, 'source_expected_bytes': TOTAL,
            'source_expected_sha256': EXPECTED_SHA, 'compressed_bytes_read': reader.offset,
            'compressed_sha256': reader.digest.hexdigest(), 'full_archive_sha_verified': complete,
            'observed_scene_count': len(seen), 'missing_target_scene_count': len(missing),
            'target_ids_observed': sorted(seen & set(missing)),
            'target_ids_not_observed': sorted(set(missing)-seen),
            'complete_candidate_timestamp_scene_ids': matched, 'extra_timestamps': extras,
            'staged_frame_count': sum(len(v) for v in files.values()), 'frames': files,
            'image_source_geometry_equivalence': 'unverified; PNG metadata has no camera record',
            'formal_pose_metrics_available': False,
            'full_prescribed_split_ready': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-scan', action='store_true')
    parser.add_argument('--prefix-mib', type=int, default=8)
    parser.add_argument('--data-root', type=Path, default=Path('data/RealEstate10K_missing_candidate'))
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise ValueError('Historical source report exists; choose a new output')
    if not 1 <= args.prefix_mib <= 1024:
        raise ValueError('Prefix budget must be 1..1024 MiB')
    index = json.loads(Path('results/re10k_rgb_index_audit.json').read_text())
    missing = index['missing_scene_ids']
    metadata = {s: read_re10k_metadata(Path('data/RealEstate10K/test')/(s+'.txt')) for s in missing}
    args.data_root.mkdir(parents=True, exist_ok=True)
    if args.data_root.is_symlink():
        raise ValueError('Staging root symlink prohibited')
    with (args.data_root/'.stream.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = sum(p.stat().st_size for p in args.data_root.glob('*/*.png'))
        if existing > 16*RESERVE or shutil.disk_usage(args.data_root).free < RESERVE:
            raise RuntimeError('Staging storage budget unavailable')
        with RangeReader(URL, TOTAL, TOTAL if args.full_scan else args.prefix_mib*1024**2) as reader:
            report = audit(reader, args.data_root, metadata, missing, [existing,16*RESERVE])
        report['official_metadata_sha256'] = {
            s: sha(Path('data/RealEstate10K/test')/(s+'.txt')) for s in missing}
        save_identical(args.output_json, report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('frames','official_metadata_sha256')},
                         indent=2), flush=True)


if __name__ == '__main__':
    main()
