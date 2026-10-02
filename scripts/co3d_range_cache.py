#!/usr/bin/env python3
"""Bounded original-candidate CO3D member cache, not a new sampling manifest.

Central metadata is cached once. Local ZIP headers, exact compressed ranges,
bounded inflate, CRC and SHA are verified before exclusive raw-member writes.
No full archive SHA claim, eviction, recursive cleanup or GPU work.
"""
import hashlib
import io
import json
import shutil
import struct
import zipfile
import zlib
from pathlib import Path, PurePosixPath

from prepare_re10k_rgb_from_archive import sha, save_identical
from probe_co3d_zip_ranges import HttpRangeFile, inventory, MAX_MEMBERS
from plan_co3d_seen41_storage import expected_paths

MAX_MEMBER = 32 * 1024**2
MAX_CACHE = 2 * 1024**3
RESERVE = 1024**3


class RecordedRanges(HttpRangeFile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = []

    def read(self, size=-1):
        start = self.tell()
        data = super().read(size)
        self.blocks.append((start, data))
        return data


class ReplayIndex(io.RawIOBase):
    """Sparse in-memory replay of bounded directory reads, not full ZIP data."""
    def __init__(self, total, blocks):
        self.total, self.blocks, self.position = total, blocks, 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        target = offset + {0:0,1:self.position,2:self.total}[whence]
        if not 0 <= target <= self.total: raise ValueError('Replay seek out of bounds')
        self.position = target
        return target
    def read(self, size=-1):
        if size < 0: size = self.total-self.position
        size = min(size,self.total-self.position)
        if not size: return b''
        for start,data in self.blocks:
            at = self.position-start
            if 0 <= at and at+size <= len(data):
                self.position += size
                return data[at:at+size]
        raise ValueError('Replay attempted non-index bytes')


def decode_member(reader, entry):
    """Read one verified central entry. Reader remains bound to source ETag."""
    if (not 0 < entry['file_size'] <= MAX_MEMBER or not 0 < entry['compress_size'] <= MAX_MEMBER
            or entry['flag_bits'] & 1 or entry['compress_type'] not in (0, 8)
            or not 0 <= entry['header_offset'] < reader.total):
        raise ValueError('Unsafe member sizes/type/offset')
    reader.seek(entry['header_offset'])
    header = reader.read(30)
    if len(header) != 30:
        raise ValueError('Truncated local header')
    sig, version, flags, method, time, date, crc, packed, size, name_len, extra_len = struct.unpack('<4s5H3I2H', header)
    if (sig != b'PK\x03\x04' or flags != entry['flag_bits'] or method != entry['compress_type']
            or not 0 < name_len <= 4096):
        raise ValueError('Local header differs from central entry')
    name = reader.read(name_len).decode('utf-8' if flags & 0x800 else 'cp437')
    extra = reader.read(extra_len)
    if name != entry['filename']:
        raise ValueError('Local filename mismatch')
    if not flags & 8:
        if size == 0xffffffff or packed == 0xffffffff:
            at, zip64 = 0, None
            while at < len(extra):
                if at + 4 > len(extra): raise ValueError('Malformed extra header')
                kind, length = struct.unpack_from('<HH', extra, at)
                at += 4
                if at + length > len(extra): raise ValueError('Malformed extra length')
                if kind == 1: zip64 = extra[at:at + length]
                at += length
            cursor = 0
            for field in ('size', 'packed'):
                value = size if field == 'size' else packed
                if value == 0xffffffff:
                    if zip64 is None or cursor + 8 > len(zip64): raise ValueError('Missing ZIP64 sizes')
                    value = struct.unpack_from('<Q', zip64, cursor)[0]; cursor += 8
                    if field == 'size': size = value
                    else: packed = value
        if (size, packed, crc) != (entry['file_size'], entry['compress_size'], entry['CRC']):
            raise ValueError('Local CRC/sizes differ from central entry')
    if reader.tell() + entry['compress_size'] > entry['central_directory_offset']:
        raise ValueError('Member overlaps central directory')
    data = reader.read(entry['compress_size'])
    if len(data) != entry['compress_size']: raise ValueError('Truncated compressed member')
    if method == 8:
        decoder = zlib.decompressobj(-15)
        payload = decoder.decompress(data, entry['file_size'] + 1)
        if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
            raise ValueError('Deflate exceeds bound or has trailing/truncated input')
    else:
        payload = data
    if len(payload) != entry['file_size'] or zlib.crc32(payload) & 0xffffffff != entry['CRC']:
        raise ValueError('Member length/CRC mismatch')
    return payload


class RawCache:
    def __init__(self, root=Path('data/co3d_range_cache')):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = Path('data/co3d_test_metadata/selected_seqs_test_seen41_candidate.json')
        self.selected = json.loads(self.manifest.read_text())
        parent = json.loads(Path('results/co3d_test_selection_manifest.json').read_text())
        refs = Path('data/co3d_test_metadata/references')
        for name, key in [('links.json','links_sha256'),('co3d_sha256.json','checksums_sha256')]:
            if sha(refs / name) != parent[key]: raise ValueError('Pinned source references changed')
        self.links = json.loads((refs / 'links.json').read_text())['full']
        self.checksums = json.loads((refs / 'co3d_sha256.json').read_text())['full']
        footer_path = Path('results/co3d_zip_footer_preflight_20261002.json')
        footer = json.loads(footer_path.read_text())
        protocol = json.loads(Path('results/co3d_seen41_protocol_20261002.json').read_text())
        if sha(self.manifest) != protocol['candidate_manifest_sha256']:
            raise ValueError('Candidate manifest changed')
        self.footers = {r['url']: r for r in footer['archives']}
        self.fingerprint = {'candidate_sha256':sha(self.manifest), 'footer_sha256':sha(footer_path),
                            'cache_code_sha256':sha(Path(__file__)),
                            'range_code_sha256':sha(Path('scripts/probe_co3d_zip_ranges.py'))}
        self.indexes = {}
        self.network_bytes = 0

    def budget(self, additional=0):
        used = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        if used + additional > MAX_CACHE or shutil.disk_usage(self.root).free < RESERVE + additional:
            raise RuntimeError('Cache 2GiB ceiling or 1GiB reserve would be violated; no automatic deletion')

    def category_index(self, category):
        if category not in self.selected: raise ValueError('Not an original seen41 candidate category')
        if category in self.indexes: return self.indexes[category]
        path = self.root / 'indices' / (category + '.json')
        if path.exists():
            if path.stat().st_size > 32*1024**2: raise ValueError('Oversized cached index')
            saved = json.loads(path.read_text())
            if saved['fingerprint'] != self.fingerprint: raise ValueError('Cache identity changed; preserve old cache')
            entries = saved['entries']
        else:
            entries = {}
            matched = set()
            for url in self.links[category][1:]:
                footer = self.footers[url]
                reader = RecordedRanges(url, footer['archive_bytes'], budget=footer['central_directory_bytes']+65685)
                reader.etag = footer['etag']
                # inventory validates ALL names/types/counts and candidate membership.
                stats = inventory(reader, category, {s:set(f) for s,f in self.selected[category].items()}, matched)
                if stats['zip_member_count'] != footer['member_count']: raise ValueError('Member count changed')
                reader2 = ReplayIndex(reader.total, reader.blocks)
                with zipfile.ZipFile(reader2) as z:
                    if len(z.infolist()) > MAX_MEMBERS: raise ValueError('Member count exceeds bound')
                    for e in z.infolist():
                        if e.filename not in matched or e.is_dir(): continue
                        if e.filename in entries: raise ValueError('Duplicate selected entry across archives')
                        entries[e.filename] = {k:getattr(e,k) for k in ('filename','header_offset','compress_size','file_size','CRC','compress_type','flag_bits')}
                        entries[e.filename].update(url=url, archive_bytes=reader.total, etag=reader.etag,
                            central_directory_offset=footer['central_directory_offset'],
                            expected_full_archive_sha256=self.checksums[Path(url).name])
                self.network_bytes += reader.bytes_read
            if set(entries) != expected_paths(category,self.selected[category]): raise ValueError('Full original category index mismatch')
            saved = {'fingerprint':self.fingerprint,'category':category,'entries':entries}
            self.budget(len(json.dumps(saved).encode())*2)
            save_identical(path,saved)
        if set(entries) != expected_paths(category,self.selected[category]): raise ValueError('Cached candidate paths mismatch')
        for name,e in entries.items():
            if e['url'] not in self.links[category][1:]: raise ValueError('Cached source URL changed')
            footer = self.footers[e['url']]
            if (e['filename'] != name or e['archive_bytes'] != footer['archive_bytes']
                    or e['etag'] != footer['etag'] or e['central_directory_offset'] != footer['central_directory_offset']
                    or e['expected_full_archive_sha256'] != self.checksums[Path(e['url']).name]):
                raise ValueError('Cached source identity changed')
        self.indexes[category] = entries
        return entries

    def fetch(self, member):
        parts = PurePosixPath(member).parts
        if len(parts)!=4 or '..' in parts or '\\' in member or PurePosixPath(member).is_absolute():
            raise ValueError('Unsafe raw member path')
        entry = self.category_index(parts[0])[member]
        path = self.root / 'raw' / member
        record_path = Path(str(path)+'.json')
        if path.exists():
            if not record_path.exists(): raise ValueError('Incomplete raw cache; preserve existing bytes')
            record = json.loads(record_path.read_text())
            if (record['fingerprint']!=self.fingerprint or record['entry']!=entry
                    or path.stat().st_size!=entry['file_size'] or sha(path)!=record['sha256']):
                raise ValueError('Existing raw cache differs; no overwrite')
            return path, record
        self.budget(entry['file_size']+65536)
        reader = HttpRangeFile(entry['url'],entry['archive_bytes'],budget=entry['compress_size']+69662)
        reader.etag = entry['etag']
        payload = decode_member(reader,entry)
        self.network_bytes += reader.bytes_read
        record = {'fingerprint':self.fingerprint,'entry':entry,'sha256':hashlib.sha256(payload).hexdigest(),
                  'member_crc_verified':True,'full_archive_sha_verified':False,
                  'transport_bytes':reader.bytes_read,'ranges':reader.ranges}
        self.budget(len(payload)+len(json.dumps(record).encode())*2)
        path.parent.mkdir(parents=True,exist_ok=True)
        # Exclusive creation; an interrupted write is diagnosed, never overwritten.
        with path.open('xb') as out: out.write(payload)
        save_identical(record_path,record)
        return path,record
