#!/usr/bin/env python3
"""Bounded EOCD/ZIP64 preflight for all seen41 archives; no central directories.

Only <=65557 final bytes per ZIP are read, enough for a standard ZIP comment
and adjacent ZIP64 records. Nonstandard/multidisk ZIPs fail closed. This is
size/count planning, NOT image CRC, archive SHA, or pose evaluation.
"""
import concurrent.futures
import hashlib
import json
import shutil
import struct
import urllib.request
from pathlib import Path

from prepare_re10k_rgb_from_archive import sha, save_identical
from probe_co3d_zip_ranges import HttpRangeFile

TAIL_LIMIT=65557


def parse_footer(tail,total):
    begin=total-len(tail)
    at=tail.rfind(b'PK\x05\x06')
    if at<0 or len(tail)-at<22:raise ValueError('ZIP EOCD not found in bounded footer')
    sig,disk,cd_disk,n_disk,n,cd_size,cd_offset,comment=struct.unpack_from('<4s4H2LH',tail,at)
    if at+22+comment!=len(tail) or disk or cd_disk or n_disk!=n:
        raise ValueError('Nonstandard/multidisk EOCD unsupported')
    boundary=begin+at
    is_zip64=n==65535 or cd_size==0xffffffff or cd_offset==0xffffffff
    if is_zip64:
        if at<20:raise ValueError('ZIP64 locator missing')
        sig,disk,zip64_offset,disks=struct.unpack_from('<4sLQL',tail,at-20)
        rel=zip64_offset-begin
        if sig!=b'PK\x06\x07' or disk or disks!=1 or rel<0 or rel+56>at-20:
            raise ValueError('ZIP64 locator outside bounded footer or multidisk')
        fields=struct.unpack_from('<4sQ2H2L4Q',tail,rel)
        sig,record_size,_,_,disk,cd_disk,n_disk,n,cd_size,cd_offset=fields
        if (sig!=b'PK\x06\x06' or record_size<44 or rel+12+record_size!=at-20
                or disk or cd_disk or n_disk!=n):
            raise ValueError('Unsupported ZIP64 EOCD')
        boundary=zip64_offset
    if n<=0 or cd_size<=0 or cd_offset<0 or cd_offset+cd_size>boundary:
        raise ValueError('Invalid central directory bounds/count')
    return {'zip64':is_zip64,'member_count':n,'central_directory_bytes':cd_size,
            'central_directory_offset':cd_offset}


def probe(url):
    with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=45) as response:
        total=int(response.headers['Content-Length'])
    if not 22<=total<128*1024**3:raise ValueError('Unexpected official ZIP size')
    reader=HttpRangeFile(url,total,budget=TAIL_LIMIT)
    reader.seek(max(0,total-TAIL_LIMIT))
    tail=reader.read(min(total,TAIL_LIMIT))
    result=parse_footer(tail,total)
    result.update(url=url,archive_bytes=total,etag=reader.etag,
                  footer_transport_bytes=reader.bytes_read,footer_sha256=hashlib.sha256(tail).hexdigest())
    return result


def main():
    root=Path('data/co3d_test_metadata')
    protocol_path=Path('results/co3d_seen41_protocol_20261002.json')
    protocol=json.loads(protocol_path.read_text())
    parent=json.loads(Path('results/co3d_test_selection_manifest.json').read_text())
    if sha(root/'references/links.json')!=parent['links_sha256']:raise ValueError('Pinned links changed')
    if shutil.disk_usage('results').free<1024**3+1024**2:raise RuntimeError('Preserve 1GiB reserve')
    links=json.loads((root/'references/links.json').read_text())['full']
    urls=[u for c in protocol['seen_categories'] for u in links[c][1:]]
    records={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures={executor.submit(probe,u):u for u in urls}
        for future in concurrent.futures.as_completed(futures):
            u=futures[future];records[u]=future.result()
            print(f'FOOTER {Path(u).name}: {records[u]["member_count"]} records, {records[u]["central_directory_bytes"]} directory bytes',flush=True)
    ordered=[records[u] for u in urls]
    result={'status':'seen41_all_zip_footer_size_preflight_not_directory_or_rgb_ready',
            'paper_mapping':'section 4.2 / Table 1 data budget guard diagnosis',
            'protocol_sha256':sha(protocol_path),'links_sha256':parent['links_sha256'],
            'source_script_sha256':sha(Path(__file__)),
            'category_count':len(protocol['seen_categories']),'archive_count':len(urls),
            'footer_byte_limit_per_zip':TAIL_LIMIT,
            'footer_transport_bytes':sum(r['footer_transport_bytes'] for r in ordered),
            'maximum_member_count':max(r['member_count'] for r in ordered),
            'maximum_central_directory_bytes':max(r['central_directory_bytes'] for r in ordered),
            'archives':ordered,'full_archive_sha_verified':False,'member_crc_verified':False,
            'formal_pose_metrics_available':False}
    save_identical('results/co3d_zip_footer_preflight_20261002.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='archives'},indent=2))


if __name__=='__main__':main()
