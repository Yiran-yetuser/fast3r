#!/usr/bin/env python3
"""Download the pixelSplat author's test-only archive, without extracting data.

The source does not honor HTTP Range. Resume consumes/discards the existing
prefix of a fresh response before appending, rather than duplicating bytes.
This preserves local progress but may re-transfer many bytes after interruption.
Mirror format, coverage and RGB/GT equivalence still require a separate audit.
"""
import argparse
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

URL = 'http://schadenfreude.csail.mit.edu:8000/re10k_test_only.zip'
EXPECTED_SIZE = 55604889849
LAST_MODIFIED = 'Tue, 20 Feb 2024 23:31:38 GMT'
RESERVE = 1024**3


def check_source(response):
    if response.headers.get('Last-Modified') != LAST_MODIFIED:
        raise ValueError('RGB mirror identity changed; audit before proceeding')
    if int(response.headers.get('Content-Length', 0)) != EXPECTED_SIZE:
        raise ValueError('Unexpected RGB archive length')


def space(root, additional=0):
    if shutil.disk_usage(root).free < RESERVE + additional:
        raise RuntimeError('Download budget would violate 1GiB reserve')


def consume_prefix(response, count, local):
    """Verify skipped bytes against existing prefix (server cannot seek)."""
    total = count
    last_log = time.monotonic()
    with local.open('rb') as stream:
        while count:
            block = response.read(min(1024**2, count))
            if not block or block != stream.read(len(block)):
                raise ValueError('Resume prefix differs/truncated; keep existing data')
            count -= len(block)
            if time.monotonic()-last_log > 60:
                print(f'Resume prefix verified {total-count}/{total} bytes; '
                      'existing archive preserved', flush=True)
                last_log = time.monotonic()
    print(f'Resume prefix verification complete: {total} bytes', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('data/re10k_test_only.zip.part'))
    parser.add_argument('--manifest', type=Path, default=Path('results/re10k_rgb_archive_manifest.json'))
    args = parser.parse_args()
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    offset = args.archive.stat().st_size if args.archive.exists() else 0
    if offset > EXPECTED_SIZE:
        raise ValueError('Existing archive oversized; do not overwrite')
    # Reserve extraction/probe headroom as well as the minimum 1GiB.
    space(args.archive.parent, EXPECTED_SIZE-offset + 2*1024**3)
    if offset < EXPECTED_SIZE:
        with urllib.request.urlopen(URL, timeout=120) as response:
            check_source(response)
            if response.status != 200:
                raise ValueError('Expected full response from non-Range server')
            if offset:
                print(f'Resuming: verify and re-transfer {offset} prefix bytes', flush=True)
                consume_prefix(response, offset, args.archive)
            with args.archive.open('ab') as stream:
                last_log = 0
                while block := response.read(1024**2):
                    if stream.tell()+len(block) > EXPECTED_SIZE:
                        raise ValueError('Oversized mirror response')
                    space(args.archive.parent, len(block))
                    stream.write(block)
                    if time.monotonic()-last_log > 60:
                        stream.flush()
                        print(f'RGB archive {stream.tell()}/{EXPECTED_SIZE} bytes', flush=True)
                        last_log = time.monotonic()
    if args.archive.stat().st_size != EXPECTED_SIZE:
        raise ValueError('Incomplete RGB archive; resume rather than evaluate')
    digest = hashlib.sha256()
    with args.archive.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(block)
    with zipfile.ZipFile(args.archive) as archive:
        entries = archive.infolist()
        if len({entry.filename for entry in entries}) != len(entries):
            raise ValueError('Duplicate archive names')
        # Do not extract untrusted paths, nor execute unrestricted torch pickle.
        index = [entry.filename for entry in entries if entry.filename.endswith('/test/index.json')]
        report = {'status': 'archive_downloaded_not_data_prepared', 'source_url': URL,
                  'source_link': 'https://github.com/dcharatan/pixelsplat#acquiring-datasets',
                  'archive_bytes': EXPECTED_SIZE, 'last_modified': LAST_MODIFIED,
                  'archive_sha256': digest.hexdigest(), 'zip_entry_count': len(entries),
                  'test_index_members': index, 'archive_path': str(args.archive.resolve()),
                  'rgb_gt_equivalence_verified': False, 'full_scene_coverage_verified': False,
                  'crc_verified': False, 'note': 'ZIP directory read; individual CRC and camera audit still required'}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.manifest.exists():
        if json.loads(args.manifest.read_text()) != report:
            raise ValueError('Existing archive manifest differs; preserve history')
    else:
        with args.manifest.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print('RGB ARCHIVE DOWNLOAD COMPLETE; data audit/preparation still required', flush=True)


if __name__ == '__main__':
    main()
