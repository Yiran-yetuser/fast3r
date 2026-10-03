"""Separate, explicitly bounded source51 cache and forced-landscape candidate.

Frozen raw2GiB/processed512MiB roots/code remain untouched. This NEW version
has raw8GiB/processed3GiB limits, plus a separately bounded request journal.
It preserves source pool/sampling/retry logic, V4 reference preprocessing,
and fails on IO/CRC/unsupported geometry instead of sampling past errors.
"""
import hashlib
import json
import os
import shutil
import tempfile
import zlib
from pathlib import Path, PurePosixPath

import numpy as np

from audit_co3d_portrait_geometry import force_landscape_method
from co3d_lazy_dataset_v4 import LazyPreparer as FrozenV4, StrictLazyCo3d
from co3d_range_cache import RawCache as FrozenRaw, decode_member
from co3d_request_journal import atomic_new
from prepare_re10k_rgb_from_archive import sha
from probe_co3d_zip_ranges import HttpRangeFile

MANIFEST = Path('data/co3d_test_metadata/selected_seqs_test_reconstructed.json')
STORAGE = Path('results/co3d_51_source_storage_v1_20261003.json')
PROOF = Path('results/co3d_51_source_storage_verified_20261003.json')
DIRECTORIES = Path('data/co3d_51_source_storage_v1')
RAW_ROOT = Path('data/co3d51_source_raw_v1')
PROCESSED_ROOT = Path('data/co3d51_source_processed_v1')
RAW_LIMIT = 8*1024**3
PROCESSED_LIMIT = 3*1024**3
RESERVE = 1024**3
FRAME_WRITE_ALLOWANCE = 32*1024**2
MAX_PROCESSED_PIXELS = 1024**2


def file_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())


def check_budget(used, additional, limit, root):
    if used+additional > limit or shutil.disk_usage(root).free < RESERVE+additional:
        raise RuntimeError('Declared new-version byte ceiling/1GiB reserve; checkpoint preserved, no eviction')


def atomic_payload(path, payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.owned-raw-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        os.link(temp,path)
        directory=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:os.unlink(temp)  # only our unpublished temporary inode


def checked_payload(path, entry):
    if path.stat().st_size != entry['file_size']:
        raise ValueError('Incomplete raw member preserved; never overwrite it')
    payload=path.read_bytes()
    if zlib.crc32(payload)&0xffffffff != entry['CRC']:
        raise ValueError('Raw member CRC differs; never skip it')
    return hashlib.sha256(payload).hexdigest()


class Source51Raw:
    def __init__(self, root=RAW_ROOT):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.manifest=MANIFEST;self.selected=json.loads(MANIFEST.read_text())
        proof=json.loads(PROOF.read_text());self.storage=json.loads(STORAGE.read_text())
        if (proof['report_sha256'] != sha(STORAGE) or proof['category_count'] != 51
                or proof['verifier_sha256'] != sha('scripts/verify_co3d_51_source_storage.py')
                or proof['formal_Table1_result'] or proof['actual_gt_validity_verified']):
            raise ValueError('Independent directory budget proof absent/stale/promoted')
        sampler_path=Path('results/co3d_51_released_sampler_20261003.json')
        sampler=json.loads(sampler_path.read_text())
        if (sha(sampler_path)!=self.storage['fingerprint']['sampler_report_sha256']
                or sha(MANIFEST)!=sampler['input_sha256'][str(MANIFEST)]):
            raise ValueError('Current candidate differs from source-sampler/budget proof')
        self.frozen=FrozenRaw();self.indexes={};self.network_bytes=0
        self.used_bytes=file_bytes(self.root)
        self.fingerprint={'scope':'source51_raw8GiB_v1_not_historical_cache_expansion',
            'candidate_sha256':sha(MANIFEST),'directory_report_sha256':sha(STORAGE),
            'directory_proof_sha256':sha(PROOF),'code_sha256':sha(__file__),
            'range_code_sha256':sha('scripts/probe_co3d_zip_ranges.py'),
            'frozen_raw_identity':self.frozen.fingerprint,'raw_limit_bytes':RAW_LIMIT,'reserve_bytes':RESERVE}

    def budget(self, additional=0):
        check_budget(self.used_bytes,additional,RAW_LIMIT,self.root)

    def category_index(self, category):
        if category in self.indexes:return self.indexes[category]
        path=DIRECTORIES/'categories'/(category+'.json')
        if sha(path) != self.storage['category_journals_sha256'][category]:
            raise ValueError('Verified category directory changed')
        row=json.loads(path.read_text());source=row['source']
        if source['mode']=='reused_frozen_local_index':
            old=self.frozen.root/'indices'/(category+'.json')
            if not old.exists() or sha(old)!=source['sha256']:
                raise ValueError('Frozen local index absent/changed; never redownload via old cache')
            entries=self.frozen.category_index(category)
            self.frozen.indexes.clear()
        elif source['mode']=='new_directory_read':
            entries={}
            for record in source['archive_receipts']:
                p=DIRECTORIES/'archives'/(Path(record['url']).name+'.json')
                if sha(p)!=record['sha256']:raise ValueError('New directory receipt changed')
                part=json.loads(p.read_text())['entries']
                if set(entries)&set(part):raise ValueError('Duplicate source member')
                entries.update(part)
        else:raise ValueError('Unknown directory source')
        self.indexes[category]=entries
        return entries

    def fetch(self, member):
        parts=PurePosixPath(member).parts
        if (len(parts)!=4 or PurePosixPath(member).is_absolute() or '..' in parts
                or '\\' in member or parts[0] not in self.selected):
            raise ValueError('Unsafe/not-original raw member')
        entry=self.category_index(parts[0])[member]
        own=self.root/'raw'/member;old=self.frozen.root/'raw'/member
        receipt_path=self.root/'records'/(member+'.json')
        if receipt_path.exists():
            record=json.loads(receipt_path.read_text())
            if record['fingerprint']!=self.fingerprint or record['entry']!=entry:
                raise ValueError('New-version raw identity changed')
            path=old if record['storage']=='borrowed_frozen_raw' else own
            if checked_payload(path,entry)!=record['sha256']:
                raise ValueError('Raw bytes differ from immutable new-version receipt')
            return path,record
        record={'fingerprint':self.fingerprint,'entry':entry,'member_crc_verified':True,
                'full_archive_sha_verified':False}
        if old.exists():
            old_receipt=Path(str(old)+'.json')
            previous=json.loads(old_receipt.read_text())
            if (previous['fingerprint']!=self.frozen.fingerprint or previous['entry']!=entry
                    or previous['sha256']!=checked_payload(old,entry) or not previous['member_crc_verified']):
                raise ValueError('Frozen raw borrow verification failed')
            path=old;record.update(storage='borrowed_frozen_raw',sha256=previous['sha256'],
                borrowed_receipt_sha256=sha(old_receipt),transport_bytes=0)
        elif own.exists():
            # A crash after atomic raw commit but before receipt is recoverable.
            # Missing transport receipt is explicit, never fabricated as zero.
            path=own;record.update(storage='downloaded_raw',sha256=checked_payload(own,entry),
                recovered_atomic_orphan=True,transport_bytes=None)
        else:
            self.budget(entry['file_size']+65536)
            reader=HttpRangeFile(entry['url'],entry['archive_bytes'],budget=entry['compress_size']+69662)
            reader.etag=entry['etag'];payload=decode_member(reader,entry)
            self.network_bytes+=reader.bytes_read
            record.update(storage='downloaded_raw',sha256=hashlib.sha256(payload).hexdigest(),
                transport_bytes=reader.bytes_read,ranges=reader.ranges)
            self.budget(len(payload)+len(json.dumps(record).encode())*2)
            atomic_payload(own,payload);self.used_bytes+=len(payload);path=own
        self.budget(len(json.dumps(record).encode())*2)
        atomic_new(receipt_path,record);self.used_bytes+=receipt_path.stat().st_size
        return path,record


class Source51Preparer(FrozenV4):
    def __init__(self, root=PROCESSED_ROOT):
        super().__init__(root)
        proof=json.loads(Path('results/co3d_finite_signed_reference_v4_verified_20261002.json').read_text())
        if (proof['v4_fingerprint']['lazy_code_sha256']!=sha('scripts/co3d_lazy_dataset_v4.py')
                or not all(proof[k] for k in ('image_depth_mask_bytes_equal',
                    'npz_arrays_and_dtypes_equal','strict_and_original_loader_outputs_equal'))):
            raise ValueError('Frozen V4 reference proof changed')
        self.cache=Source51Raw();self.used_bytes=file_bytes(self.root)
        self.fingerprint.update(protocol_version='reference_v4_source51_landscape_candidate_v1',
            candidate_sha256=sha(MANIFEST),cache=self.cache.fingerprint,
            new_adapter_code_sha256=sha(__file__),processed_limit_bytes=PROCESSED_LIMIT,
            processed_pixel_guard=MAX_PROCESSED_PIXELS,
            frozen_v4_sha256=sha('scripts/co3d_lazy_dataset_v4.py'))

    def budget(self, additional=0):
        check_budget(self.used_bytes,max(additional,FRAME_WRITE_ALLOWANCE) if additional else 0,
                     PROCESSED_LIMIT,self.root)

    def ensure(self, category, scene, frame):
        member=f'{category}/{scene}/images/frame{frame:06d}.jpg'
        annotation=self.annotations(category)[member];h,w=annotation['image']['size']
        scale=384/min(h,w)+1e-8
        target=np.floor(np.asarray([w,h])*scale).astype(int)
        if max(target)<512:target=np.floor(np.asarray([w,h])*(512/max(h,w)+1e-8)).astype(int)
        if np.prod(target)>MAX_PROCESSED_PIXELS:
            raise ValueError('Unreviewed processed size; never crop/resample past this error')
        record_path=self.root/(member+'.prepared.json')
        targets=[self.root/annotation[k]['path'] for k in ('image','depth','mask')]
        targets += [targets[0].with_suffix('.npz'),record_path]
        before=sum(p.stat().st_size for p in targets if p.exists())
        if not record_path.exists():self.budget(FRAME_WRITE_ALLOWANCE)
        record=super().ensure(category,scene,frame)
        after=sum(p.stat().st_size for p in targets)
        if after<before:raise ValueError('Unexpected processed deletion')
        self.used_bytes+=after-before;self.budget()
        return record


class Source51Landscape(StrictLazyCo3d):
    # Already tested AST-guarded crop: remove only portrait/square reversal.
    # Actual shared sampler state changes under this declared NEW crop protocol.
    _crop_resize_if_necessary=force_landscape_method()
