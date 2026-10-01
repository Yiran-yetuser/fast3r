#!/usr/bin/env python3
"""Read bounded ZIP central directories over exact HTTP Range, no image download.

Reports advertised member budgets only: partial Range SHA is not full ZIP SHA,
nor member CRC validation. The default apple probe is NOT full dataset coverage.
"""
import argparse
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from prepare_re10k_rgb_from_archive import save_identical, sha

MAX_INDEX_BYTES=32*1024**2
MAX_MEMBERS=200000


class HttpRangeFile(io.RawIOBase):
    def __init__(self,url,total,opener=urllib.request.urlopen,budget=MAX_INDEX_BYTES):
        self.url,self.total,self.opener,self.budget=url,total,opener,budget
        self.position=0
        self.etag=None
        self.bytes_read=0
        self.ranges=[]

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position

    def seek(self,offset,whence=0):
        if whence not in (0,1,2):raise ValueError('Invalid seek mode')
        position=offset+{0:0,1:self.position,2:self.total}[whence]
        if not 0<=position<=self.total:raise ValueError('Seek outside archive')
        self.position=position
        return position

    def read(self,size=-1):
        if size<0:size=self.total-self.position
        size=min(size,self.total-self.position)
        if size==0:return b''
        if size>self.budget-self.bytes_read:raise ValueError('ZIP directory read exceeds budget')
        begin,end=self.position,self.position+size-1
        request=urllib.request.Request(self.url,headers={
            'Range':f'bytes={begin}-{end}','Accept-Encoding':'identity'})
        with self.opener(request,timeout=45) as response:
            if (response.status!=206 or response.headers.get('Content-Range')!=f'bytes {begin}-{end}/{self.total}'
                    or response.headers.get('Content-Encoding','identity')!='identity'):
                raise ValueError('Server ignored exact ZIP Range')
            tag=response.headers.get('ETag')
            if not tag or (self.etag is not None and self.etag!=tag):
                raise ValueError('ZIP source ETag changed or absent')
            payload=response.read(size+1)
            if len(payload)!=size:raise ValueError('ZIP Range response length mismatch')
        self.etag=tag
        self.position+=size
        self.bytes_read+=size
        self.ranges.append({'start':begin,'end':end,'sha256':hashlib.sha256(payload).hexdigest()})
        return payload


def inventory(reader,category,selected,cross_archive_paths=None):
    counts={kind:0 for kind in ('images','depths','masks')}
    compressed={kind:0 for kind in counts}
    expanded={kind:0 for kind in counts}
    matched=set()
    with zipfile.ZipFile(reader) as archive:
        entries=archive.infolist()
        if len(entries)>MAX_MEMBERS:raise ValueError('ZIP member count exceeds budget')
        names=set()
        for entry in entries:
            parts=PurePosixPath(entry.filename).parts
            if (entry.filename in names or not parts or PurePosixPath(entry.filename).is_absolute()
                    or '..' in parts or '\\' in entry.filename or parts[0]!=category):
                raise ValueError('Unsafe, duplicate or unexpected ZIP member')
            names.add(entry.filename)
            if entry.is_dir():continue
            if (entry.flag_bits&1 or entry.compress_type not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED)
                    or entry.header_offset<0 or entry.header_offset>=reader.total
                    or ((entry.external_attr>>16)&0o170000)==0o120000):
                raise ValueError('Unsupported/encrypted/link ZIP member')
            if len(parts)!=4 or parts[1] not in selected or parts[2] not in counts:continue
            match=re.fullmatch(r'frame([0-9]{6})(\.jpg|\.png|\.jpg\.geometric\.png)',parts[3])
            if not match or int(match[1]) not in selected[parts[1]]:continue
            kind=parts[2]
            expected={'images':'.jpg','masks':'.png','depths':'.jpg.geometric.png'}[kind]
            if match[2]!=expected:continue
            if not 0<entry.file_size<=32*1024**2 or not 0<entry.compress_size<=32*1024**2:
                raise ValueError('Selected frame member exceeds bound')
            if cross_archive_paths is not None:
                if entry.filename in cross_archive_paths:raise ValueError('Duplicate selected member across ZIPs')
                cross_archive_paths.add(entry.filename)
            matched.add(entry.filename)
            counts[kind]+=1
            compressed[kind]+=entry.compress_size
            expanded[kind]+=entry.file_size
    return {'zip_member_count':len(entries),'matched_member_counts':counts,
            'matched_compressed_bytes':compressed,'matched_uncompressed_bytes':expanded,
            'matched_member_paths_sha256':hashlib.sha256('\n'.join(sorted(matched)).encode()).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--category',default='apple')
    parser.add_argument('--output-json',type=Path,default=Path('results/co3d_apple_range_probe_20261002.json'))
    args=parser.parse_args()
    root=Path('data/co3d_test_metadata')
    protocol=json.loads(Path('results/co3d_seen41_protocol_20261002.json').read_text())
    manifest=root/'selected_seqs_test_seen41_candidate.json'
    if sha(manifest)!=protocol['candidate_manifest_sha256']:raise ValueError('Candidate manifest changed')
    selected=json.loads(manifest.read_text())
    if args.category not in protocol['seen_categories']:raise ValueError('Not a seen41 category')
    links=json.loads((root/'references/links.json').read_text())['full']
    checksums=json.loads((root/'references/co3d_sha256.json').read_text())['full']
    parent=json.loads(Path('results/co3d_test_selection_manifest.json').read_text())
    if (sha(root/'references/links.json')!=parent['links_sha256']
            or sha(root/'references/co3d_sha256.json')!=parent['checksums_sha256']):
        raise ValueError('Pinned archive references changed')
    records=[]
    for url in links[args.category][1:]:
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=45) as response:
            total=int(response.headers['Content-Length'])
        if not 0<total<128*1024**3:raise ValueError('Unexpected image archive length')
        reader=HttpRangeFile(url,total)
        record=inventory(reader,args.category,{s:set(f) for s,f in selected[args.category].items()})
        record.update(url=url,archive_bytes=total,expected_full_archive_sha256=checksums[Path(url).name],
                      full_archive_sha_verified=False,member_crc_verified=False,
                      index_transport_bytes=reader.bytes_read,etag=reader.etag,ranges=reader.ranges)
        records.append(record)
        print(f"INDEX ONLY {Path(url).name}: {reader.bytes_read} bytes, {record['matched_member_counts']}",flush=True)
    expected=sum(map(len,selected[args.category].values()))
    count={k:sum(r['matched_member_counts'][k] for r in records) for k in ('images','depths','masks')}
    result={'status':'one_category_central_directory_range_probe_not_rgb_ready','category':args.category,
            'paper_mapping':'section 4.2 / Table 1 data budget feasibility only',
            'candidate_manifest_sha256':sha(manifest),'selected_sequence_count':len(selected[args.category]),
            'expected_candidate_frame_count':expected,'matched_member_counts':count,
            'counts_match_candidate_frames':all(v==expected for v in count.values()),
            'advertised_selected_uncompressed_bytes':sum(sum(r['matched_uncompressed_bytes'].values()) for r in records),
            'index_transport_bytes':sum(r['index_transport_bytes'] for r in records),
            'archives':records,'formal_pose_metrics_available':False,
            'all_41_category_inventory_complete':False,'member_crc_verified':False,
            'note':'Names/sizes only; cross-archive duplicates, frame annotations, CRC, decoding and full scope remain unverified'}
    save_identical(args.output_json,result)
    print(json.dumps({k:v for k,v in result.items() if k!='archives'},indent=2))


if __name__=='__main__':main()
