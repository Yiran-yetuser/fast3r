"""Offline ZIP Range safety tests; synthetic data, not benchmark results."""
import io
import re
import unittest
import zipfile

from probe_co3d_zip_ranges import HttpRangeFile, inventory
from plan_co3d_seen41_storage import expected_paths


class Response(io.BytesIO):
    def __init__(self,data,start,end,total,tag='fixed',status=206):
        super().__init__(data)
        self.status=status
        self.headers={'Content-Range':f'bytes {start}-{end}/{total}','ETag':tag}


class ZipRangesTests(unittest.TestCase):
    def archive(self,names):
        target=io.BytesIO()
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
            for n in names:z.writestr(n,b'synthetic')
        return target.getvalue()

    def reader(self,payload,status=206,tag='fixed',budget=1024**2):
        def opener(request,timeout):
            start,end=map(int,re.fullmatch(r'bytes=(\d+)-(\d+)',request.headers['Range']).groups())
            return Response(payload[start:end+1],start,end,len(payload),tag,status)
        return HttpRangeFile('https://example.invalid/archive.zip',len(payload),opener,budget)

    def test_index_only_members_and_offsets(self):
        names=['apple/s/images/frame000001.jpg','apple/s/depths/frame000001.jpg.geometric.png',
               'apple/s/masks/frame000001.png','apple/s/images/frame000002.jpg']
        raw=self.archive(names)
        reader=self.reader(raw)
        result=inventory(reader,'apple',{'s':{1}})
        self.assertEqual(result['matched_member_counts'],{'images':1,'depths':1,'masks':1})
        self.assertEqual(result['zip_member_count'],4)
        self.assertTrue(reader.ranges)

    def test_range_ignored_and_budget_fail_closed(self):
        raw=self.archive(['apple/s/images/frame000001.jpg'])
        for reader in [self.reader(raw,status=200),self.reader(raw,budget=1)]:
            with self.assertRaises(ValueError):inventory(reader,'apple',{'s':{1}})

    def test_changed_etag(self):
        reader=self.reader(b'12345')
        reader.read(1)
        reader.etag='changed'
        with self.assertRaises(ValueError):reader.read(1)

    def test_traversal_and_wrong_category(self):
        for path in ['apple/../s/images/frame000001.jpg','/apple/s/images/frame000001.jpg',
                     'book/s/images/frame000001.jpg']:
            with self.assertRaises(ValueError):inventory(self.reader(self.archive([path])),'apple',{'s':{1}})

    def test_seek_bounds(self):
        reader=self.reader(b'abc')
        self.assertEqual(reader.seek(-1,2),2)
        self.assertEqual(reader.read(),b'c')
        for args in [(4,0),(-1,0),(0,3)]:
            with self.assertRaises(ValueError):reader.seek(*args)

    def test_cross_archive_duplicates_and_full_paths(self):
        paths=expected_paths('apple',{'s':[1]})
        self.assertEqual(len(paths),3)
        raw=self.archive(sorted(paths))
        matched=set()
        inventory(self.reader(raw),'apple',{'s':{1}},matched)
        self.assertEqual(matched,paths)
        with self.assertRaises(ValueError):inventory(self.reader(raw),'apple',{'s':{1}},matched)


if __name__=='__main__':unittest.main()
