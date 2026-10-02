"""Immutable request envelopes: data result and after-state commit together."""
import json
import os
import tempfile
from pathlib import Path
from co3d_sampling_state import digest


def atomic_new(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    fd,tmp=tempfile.mkstemp(prefix='.request-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        os.link(tmp,path)  # exclusive, never overwrites historical commits
        directory=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        os.unlink(tmp)


def read_prefix(root, initial, validate_result):
    state=initial; previous=digest(initial);rows=[]
    files=sorted(Path(root).glob('request_*.json'))
    for i,path in enumerate(files):
        if path.name!=f'request_{i:03d}.json':raise ValueError('Noncontiguous journal')
        row=json.loads(path.read_text())
        if row['request']!=i or row['before_state_sha256']!=previous:
            raise ValueError('Broken request chain')
        after=row['after_state']
        if (after['identity']!=initial['identity'] or after['next_request']!=i+1
            or after['scope']!=initial['scope'] or after['version']!=initial['version']):
            raise ValueError('Changed after-state identity or cursor')
        validate_result(row['result'],i)
        state=after;previous=digest(after);rows.append(row)
    return rows,state


def commit(root, request, before, result, after):
    if before['next_request']!=request or after['next_request']!=request+1:
        raise ValueError('Wrong transaction cursor')
    if before['identity']!=after['identity']:raise ValueError('Identity changed')
    atomic_new(Path(root)/f'request_{request:03d}.json',{
        'request':request,'before_state_sha256':digest(before),
        'result':result,'after_state':after})
