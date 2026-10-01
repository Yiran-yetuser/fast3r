"""Synthetic HTTP/tar fixtures, NOT new paper metrics."""
import gzip
import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import stream_re10k_missing_candidates as stream


class Response(io.BytesIO):
    status = 206

    def __init__(self, data, start, end, total, tag='pinned'):
        super().__init__(data)
        self.headers = {'Content-Range': f'bytes {start}-{end-1}/{total}', 'ETag': tag}


class StreamTests(unittest.TestCase):
    def test_exact_range_transport_and_sha(self):
        def opener(req, timeout):
            self.assertEqual(req.headers['Range'], 'bytes=0-3')
            return Response(b'data',0,4,4)
        reader = stream.RangeReader('https://example.test/file',4,4,opener)
        self.assertEqual(reader.read(2)+reader.read(2),b'data')
        self.assertEqual(reader.read(2),b'')
        self.assertEqual(reader.digest.hexdigest(),hashlib.sha256(b'data').hexdigest())
        reader.close()

    def test_prefix_is_not_eof(self):
        reader = stream.RangeReader('https://example.test/file',8,4,
            lambda *a,**k: Response(b'data',0,4,8))
        self.assertEqual(reader.read(4),b'data')
        with self.assertRaises(stream.ProbeLimit): reader.read(4)
        reader.close()

    def test_ignored_range_rejected(self):
        response = Response(b'data',0,4,4)
        response.status = 200
        reader = stream.RangeReader('https://example.test/file',4,4,lambda *a,**k:response)
        with self.assertRaises(ValueError): reader.read(4)
        reader.close()

    def test_interrupted_range_restarts_exact_offset(self):
        calls = []
        def opener(req, timeout):
            calls.append(req.headers['Range'])
            return Response(b'ab' if len(calls)==1 else b'cd',0 if len(calls)==1 else 2,4,4)
        with patch.object(stream.time,'sleep'):
            reader = stream.RangeReader('https://example.test/file',4,4,opener)
            self.assertEqual(reader.read(4)+reader.read(4),b'abcd')
            self.assertEqual(calls,['bytes=0-3','bytes=2-3'])
        reader.close()

    def test_changed_etag_rejected(self):
        calls = []
        def opener(req, timeout):
            calls.append(1)
            return Response(b'ab' if len(calls)==1 else b'cd',0 if len(calls)==1 else 2,4,4,
                            'old' if len(calls)==1 else 'new')
        with patch.object(stream.time,'sleep'):
            reader = stream.RangeReader('https://example.test/file',4,4,opener)
            reader.read(4)
            with self.assertRaises(ValueError): reader.read(4)
        reader.close()

    def test_traversal_link_and_member_limit_rejected(self):
        for name,kind,size in [('../outside',tarfile.REGTYPE,1),
                                ('link',tarfile.SYMTYPE,0),
                                (stream.PREFIX+'/'+'a'*16+'/0.png',tarfile.REGTYPE,stream.MAX_IMAGE+1)]:
            member = tarfile.TarInfo(name);member.type=kind;member.size=size
            with self.assertRaises(ValueError):stream.candidate_path(member)

    def png(self):
        image=io.BytesIO();Image.new('RGB',(32,24)).save(image,format='PNG')
        return image.getvalue()

    def test_resume_existing_bytes_and_low_space(self):
        with tempfile.TemporaryDirectory() as folder:
            cap=[0,1024**3]
            first=stream.store_png(folder,'a'*16,'0',self.png(),cap)
            self.assertEqual(stream.store_png(folder,'a'*16,'0',self.png(),cap),first)
            (Path(folder)/('a'*16)/'0.png').write_bytes(b'user-data')
            with self.assertRaises(ValueError):stream.store_png(folder,'a'*16,'0',self.png(),cap)
            with patch.object(stream.shutil,'disk_usage') as usage:
                usage.return_value.free=stream.RESERVE
                with self.assertRaises(RuntimeError):stream.store_png(folder,'b'*16,'0',self.png(),cap)

    def test_full_synthetic_source_and_no_promotion(self):
        raw=io.BytesIO();scene='a'*16
        with tarfile.open(fileobj=raw,mode='w') as tar:
            for timestamp in ['0','1','2']:
                payload=self.png();m=tarfile.TarInfo(stream.PREFIX+'/'+scene+'/'+timestamp+'.png')
                m.size=len(payload);tar.addfile(m,io.BytesIO(payload))
        data=gzip.compress(raw.getvalue())
        def opener(*a,**k):return Response(data,0,len(data),len(data))
        with tempfile.TemporaryDirectory() as folder,patch.object(stream,'EXPECTED_SHA',hashlib.sha256(data).hexdigest()):
            with stream.RangeReader('https://example.test/file',len(data),len(data),opener) as reader:
                report=stream.audit(reader,folder,{scene:{'0':{},'1':{}}},[scene],[0,1024**3])
            self.assertTrue(report['full_archive_sha_verified'])
            self.assertEqual(report['staged_frame_count'],2)
            self.assertEqual(report['extra_timestamps'],{scene:['2']})
            self.assertEqual(report['complete_candidate_timestamp_scene_ids'],[])
            self.assertFalse(report['full_prescribed_split_ready'])
            self.assertFalse(report['formal_pose_metrics_available'])

    def test_bounded_gzip_prefix_is_incomplete(self):
        payload=gzip.compress(b'\0'*10240)
        with tempfile.TemporaryDirectory() as folder:
            with stream.RangeReader('https://example.test/file',len(payload),10,
                 lambda *a,**k:Response(payload[:10],0,10,len(payload))) as reader:
                report=stream.audit(reader,folder,{},[],[0,1024**3])
            self.assertEqual(report['status'],'bounded_prefix_only')
            self.assertFalse(report['full_archive_sha_verified'])

    def test_full_source_sha_mismatch_not_accepted(self):
        payload=gzip.compress(b'\0'*10240)
        with tempfile.TemporaryDirectory() as folder:
            with stream.RangeReader('https://example.test/file',len(payload),len(payload),
                 lambda *a,**k:Response(payload,0,len(payload),len(payload))) as reader:
                with self.assertRaises(ValueError):stream.audit(reader,folder,{},[],[0,1024**3])


if __name__ == '__main__':unittest.main()
