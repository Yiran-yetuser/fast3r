#!/usr/bin/env python3
"""Resume the official RE10K archive and prepare only the prescribed test TXT.

This is camera metadata, NOT RGB readiness or a paper benchmark result.
Existing dataset files must agree byte-for-byte; they are never overwritten.
"""
import argparse
import hashlib
import json
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path, PurePosixPath

from check_pose_data import read_re10k_metadata, read_split

URL = 'https://storage.googleapis.com/realestate10k-public-files/RealEstate10K.tar.gz'
EXPECTED_BYTES = 752332631
EXPECTED_MD5 = '7dadaf85e559bc93ce75378e31064492'
RESERVE = 1024**3


def budget(directory, additional=0):
    if shutil.disk_usage(directory).free < RESERVE + additional:
        raise RuntimeError('Insufficient free space; keep at least 1GiB reserve')


def checksum(path, algorithm='sha256'):
    digest = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def download(archive):
    archive.parent.mkdir(parents=True, exist_ok=True)
    start = archive.stat().st_size if archive.exists() else 0
    if start > EXPECTED_BYTES:
        raise ValueError('Oversized archive; preserve it for diagnosis')
    budget(archive.parent, EXPECTED_BYTES - start + 100*1024**2)
    if start < EXPECTED_BYTES:
        headers = {'Range': f'bytes={start}-', 'If-Match': '"' + EXPECTED_MD5 + '"'}
        with urllib.request.urlopen(urllib.request.Request(URL, headers=headers), timeout=90) as response:
            if response.headers.get('ETag', '').strip('"') != EXPECTED_MD5:
                raise ValueError('Official archive changed; audit before downloading')
            if start and (response.status != 206 or not response.headers.get(
                    'Content-Range', '').startswith(f'bytes {start}-')):
                raise ValueError('Server did not honor resume range; no data appended')
            if start == 0 and response.status not in (200, 206):
                raise ValueError('Unexpected download response')
            with archive.open('ab') as stream:
                last_log = 0
                while block := response.read(1024**2):
                    if stream.tell() + len(block) > EXPECTED_BYTES:
                        raise ValueError('Download exceeds expected length')
                    budget(archive.parent, len(block))
                    stream.write(block)
                    if time.monotonic() - last_log > 30:
                        stream.flush()
                        print(f'downloaded {stream.tell()}/{EXPECTED_BYTES} bytes', flush=True)
                        last_log = time.monotonic()
    if archive.stat().st_size != EXPECTED_BYTES or checksum(archive, 'md5') != EXPECTED_MD5:
        raise ValueError('Archive length/MD5 mismatch; preserve file, do not extract')


def extract_selected(archive, root, names):
    wanted = set(names)
    root.mkdir(parents=True, exist_ok=True)
    seen = set()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if not member.isfile() or len(path.parts) < 2 or path.parts[-2] != 'test':
                continue
            name = path.stem
            if path.suffix != '.txt' or name not in wanted:
                continue
            if name in seen or not 0 < member.size <= 10*1024**2:
                raise ValueError('Duplicate/oversized metadata entry')
            budget(root, member.size)
            content = tar.extractfile(member).read()
            target = root / (name + '.txt')
            if target.exists():
                if target.read_bytes() != content:
                    raise ValueError(f'Existing metadata differs: {target}')
            else:
                with target.open('xb') as stream:
                    stream.write(content)
            read_re10k_metadata(target)
            seen.add(name)
    if seen != wanted:
        raise ValueError(f'Official test archive missing {len(wanted-seen)} prescribed IDs')
    return seen


def prepare(args):
    names = read_split(args.split_file)
    # Verified manifest + every existing file hash => no network or re-download.
    if args.manifest.exists():
        old = json.loads(args.manifest.read_text())
        if (old.get('status') == 'metadata_prepared'
                and old.get('split_sha256') == checksum(args.split_file)
                and old.get('archive_md5') == EXPECTED_MD5
                and old.get('metadata_root') == str(args.metadata_root.resolve())
                and set(old.get('records', {})) == set(names)):
            for name in names:
                path = args.metadata_root / (name + '.txt')
                if checksum(path) != old['records'][name]['sha256']:
                    raise ValueError(f'Prepared metadata changed: {name}')
                read_re10k_metadata(path)
            print('Verified prepared metadata exists; skip download', flush=True)
            return old
        raise ValueError('Existing manifest differs; preserve it and choose a new output')
    download(args.archive)
    extract_selected(args.archive, args.metadata_root, names)
    records = {}
    for name in names:
        path = args.metadata_root / (name + '.txt')
        camera = read_re10k_metadata(path)
        if len(camera) < 10:
            raise ValueError(f'Too few prescribed camera records: {name}')
        records[name] = {'sha256': checksum(path), 'camera_record_count': len(camera)}
    report = {'dataset': 'RealEstate10K', 'status': 'metadata_prepared',
              'source_url': URL, 'license': 'CC BY 4.0, Google LLC',
              'archive_bytes': EXPECTED_BYTES, 'archive_md5': EXPECTED_MD5,
              'archive_sha256': checksum(args.archive),
              'split_sha256': checksum(args.split_file), 'expected_scene_count': len(names),
              'metadata_scene_count': len(records), 'rgb_prepared': False,
              'metadata_root': str(args.metadata_root.resolve()), 'records': records,
              'paper_mapping': 'section 4.2 / Table 1 data preparation, not pose metrics'}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'METADATA COMPLETE: {len(records)} test clips; RGB still required', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('data/RealEstate10K.tar.gz.part'))
    parser.add_argument('--metadata-root', type=Path, default=Path('data/RealEstate10K/test'))
    parser.add_argument('--split-file', type=Path, default=Path('scripts/re10k_test_1800.txt'))
    parser.add_argument('--manifest', type=Path, default=Path('results/re10k_metadata_manifest.json'))
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
