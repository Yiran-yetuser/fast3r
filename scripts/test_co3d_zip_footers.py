"""Synthetic EOCD/ZIP64 parsing tests, not data/pose measurements."""
import io
import struct
import unittest
import zipfile

from audit_co3d_zip_footers import parse_footer


class FooterTests(unittest.TestCase):
    def classic(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:z.writestr('test',b'data')
        return b.getvalue()

    def zip64(self,count=500000,cd_size=60*1024**2):
        offset=20*1024**3
        rec=struct.pack('<4sQ2H2L4Q',b'PK\x06\x06',44,45,45,0,0,count,count,cd_size,offset-cd_size)
        loc=struct.pack('<4sLQL',b'PK\x06\x07',0,offset,1)
        end=struct.pack('<4s4H2LH',b'PK\x05\x06',0,0,65535,65535,0xffffffff,0xffffffff,0)
        return rec+loc+end,offset+len(rec+loc+end)

    def test_classic(self):
        data=self.classic();r=parse_footer(data,len(data))
        self.assertEqual(r['member_count'],1)
        self.assertFalse(r['zip64'])

    def test_large_zip64_without_reading_directory(self):
        tail,total=self.zip64();r=parse_footer(tail,total)
        self.assertEqual(r['member_count'],500000)
        self.assertEqual(r['central_directory_bytes'],60*1024**2)
        self.assertTrue(r['zip64'])

    def test_truncation_and_wrong_count(self):
        data=self.classic()
        for bad in (data[:-1],data+b'extra',b'notzip'):
            with self.assertRaises(ValueError):parse_footer(bad,len(bad))

    def test_multidisk_rejected(self):
        tail,total=self.zip64();bad=bytearray(tail)
        struct.pack_into('<L',bad,56+16,2)
        with self.assertRaises(ValueError):parse_footer(bytes(bad),total)

    def test_directory_outside_archive_rejected(self):
        tail,total=self.zip64();bad=bytearray(tail)
        struct.pack_into('<Q',bad,48,total+1)
        with self.assertRaises(ValueError):parse_footer(bytes(bad),total)


if __name__=='__main__':unittest.main()
