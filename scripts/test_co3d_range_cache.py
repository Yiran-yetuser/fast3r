"""Synthetic raw ZIP extraction/CRC/replay checks, not dataset coverage."""
import io
import json
import re
import unittest
import zipfile
from unittest.mock import patch

from co3d_range_cache import RecordedRanges, ReplayIndex, decode_member
from probe_co3d_zip_ranges import inventory
from test_co3d_zip_ranges import Response


class RangeCacheTests(unittest.TestCase):
    def archive(self, payload=b'original bytes', method=zipfile.ZIP_DEFLATED):
        stream=io.BytesIO()
        name='apple/s/images/frame000001.jpg'
        with zipfile.ZipFile(stream,'w',method) as z: z.writestr(name,payload)
        raw=stream.getvalue()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            e=z.getinfo(name)
            entry={k:getattr(e,k) for k in ('filename','header_offset','compress_size','file_size','CRC','compress_type','flag_bits')}
            entry['central_directory_offset']=z.start_dir
        return raw,entry

    def reader(self, raw):
        def opener(request,timeout):
            start,end=map(int,re.fullmatch(r'bytes=(\d+)-(\d+)',request.headers['Range']).groups())
            return Response(raw[start:end+1],start,end,len(raw))
        return RecordedRanges('https://example.invalid/x.zip',len(raw),opener,budget=1024**2)

    def test_stored_and_deflated_crc_bytes(self):
        for method in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED):
            raw,entry=self.archive(method=method)
            self.assertEqual(decode_member(self.reader(raw),entry),b'original bytes')

    def test_corrupt_crc_name_and_size_rejected(self):
        raw,entry=self.archive()
        for key,value in [('CRC',entry['CRC']+1),('filename','apple/s/images/frame000002.jpg'),('file_size',1)]:
            changed=dict(entry);changed[key]=value
            with self.assertRaises(ValueError): decode_member(self.reader(raw),changed)

    def test_index_replay_no_member_bytes_or_network(self):
        raw,entry=self.archive(payload=b'long original data'*10000)
        reader=self.reader(raw)
        inventory(reader,'apple',{'s':{1}})
        replay=ReplayIndex(len(raw),reader.blocks)
        with zipfile.ZipFile(replay) as z: self.assertEqual(z.getinfo(entry['filename']).CRC,entry['CRC'])
        replay.seek(0)
        # A short ZIP may be entirely in the footer read; use synthetic blocks
        # to ensure a request into an uncached member region fails closed.
        narrow=ReplayIndex(1000,[(900,b'x'*100)])
        with self.assertRaises(ValueError): narrow.read(1)

    def test_bounds_encryption_and_overlap_rejected(self):
        raw,entry=self.archive()
        for key,value in [('file_size',33*1024**2),('flag_bits',1),('central_directory_offset',31)]:
            changed=dict(entry);changed[key]=value
            with self.assertRaises(ValueError): decode_member(self.reader(raw),changed)

    def test_bounded_inflate_rejects_wrong_declared_size(self):
        raw,entry=self.archive(payload=b'a'*100000)
        changed=dict(entry);changed['file_size']=100
        changed['flag_bits'] |= 8
        # Align local bit3 to reach the inflate budget rather than header guard.
        raw=bytearray(raw);raw[6:8]=(changed['flag_bits']).to_bytes(2,'little')
        with self.assertRaises(ValueError): decode_member(self.reader(bytes(raw)),changed)


if __name__=='__main__': unittest.main()
