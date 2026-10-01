#!/usr/bin/env python3
"""Fetch ONLY official metadata ZIPs, derive a provenance-pinned test selection.

This is a reconstruction of DUSt3R's published selection, not proof that the
unpublished Fast3R scene list/frame draws match. No RGB/depth archive downloads.
"""
import argparse
import concurrent.futures
import fcntl
import gzip
import hashlib
import io
import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

from prepare_re10k_rgb_from_archive import save_identical, sha

CO3D_REV = 'eb51d7583c56ff23dc918d9deafee50f4d8178c3'
DUST3R_REV = '4c24a6ebf04809f2cfe59915e51779c8984aaa40'
BASE = f'https://raw.githubusercontent.com/facebookresearch/co3d/{CO3D_REV}/co3d/'
DUST3R_SOURCE = ('https://raw.githubusercontent.com/naver/dust3r/' + DUST3R_REV
                  + '/datasets_preprocess/preprocess_co3d.py')
# Exact category order from the pinned public preprocessor; controls seed+index.
CATEGORIES = ['apple','backpack','ball','banana','baseballbat','baseballglove',
              'bench','bicycle','book','bottle','bowl','broccoli','cake','car','carrot',
              'cellphone','chair','couch','cup','donut','frisbee','hairdryer','handbag',
              'hotdog','hydrant','keyboard','kite','laptop','microwave','motorcycle',
              'mouse','orange','parkingmeter','pizza','plant','remote','sandwich',
              'skateboard','stopsign','suitcase','teddybear','toaster','toilet','toybus',
              'toyplane','toytrain','toytruck','tv','umbrella','vase','wineglass']
MAX_ZIP = 128*1024**2
MAX_LIST = 128*1024**2
RESERVE = 1024**3


def small_get(url, cap):
    with urllib.request.urlopen(url, timeout=45) as response:
        data = response.read(cap+1)
    if len(data)>cap:
        raise ValueError('Response exceeds bounded download budget')
    return data


def write_identical(path, payload):
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f'Existing metadata differs, preserved: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(path.parent).free < RESERVE+4*MAX_ZIP:
            raise RuntimeError('Metadata writes must preserve 1GiB plus worker headroom')
        with path.open('xb') as out: out.write(payload)


def select(category, category_index, setlists, sequence_data):
    """Same sorted quality-filtered sequences, Random(seed+category_index).

    ZIP order is recorded explicitly. Upstream os.listdir order is not recorded
    by the paper; frame-list ordering equivalence remains unverified.
    """
    rows = [row for data in setlists for row in data['test']]
    qualities = {s['sequence_name']: s.get('viewpoint_quality_score') for s in sequence_data}
    good = sorted({row[0] for row in rows if qualities.get(row[0]) is not None
                   and qualities[row[0]] > .5})
    chosen = good if len(good)<50 else random.Random(42+category_index).sample(good,50)
    result = {s:[] for s in chosen}
    for scene, frame_number, filepath in rows:
        if scene not in result: continue
        name = Path(filepath).name
        if (not name.startswith('frame') or not name.endswith('.jpg')
                or len(name)!=15 or not name[5:11].isdigit()
                or Path(filepath).parts != (category,scene,'images',name)):
            raise ValueError('Malformed official frame path')
        result[scene].append(int(name[5:11]))
    for scene, frames in result.items():
        if len(frames)!=len(set(frames)) or len(frames)<10:
            raise ValueError(f'Insufficient/duplicate test candidates for {category}/{scene}')
    return result


def category_metadata(category, index, url, expected_sha, root):
    target=root/'archives'/(category+'_000.zip')
    if target.exists():
        if target.stat().st_size>MAX_ZIP or sha(target)!=expected_sha:
            raise ValueError('Existing official metadata ZIP mismatch; no overwrite')
        data=target.read_bytes()
    else:
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=45) as r:
            size=int(r.headers['Content-Length'])
        if not 0<size<=MAX_ZIP:
            raise ValueError('First ZIP is not within metadata-only budget')
        if shutil.disk_usage(root).free < RESERVE+4*MAX_ZIP:
            raise RuntimeError('Metadata workers must reserve at least 1GiB')
        data=small_get(url,MAX_ZIP)
        if len(data)!=size or hashlib.sha256(data).hexdigest()!=expected_sha:
            raise ValueError('Official metadata length/SHA mismatch')
        write_identical(target,data)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or archive.testzip() is not None:
            raise ValueError('Duplicate ZIP paths or corrupt metadata CRC')
        if any(Path(n).is_absolute() or '..' in Path(n).parts or '\\' in n for n in names):
            raise ValueError('Unsafe official ZIP path')
        list_names=[n for n in names if n.startswith(category+'/set_lists/')
                    and 'fewview_train' in n and n.endswith('.json')]
        def read(name,cap):
            if archive.getinfo(name).file_size>cap:
                raise ValueError('Metadata member exceeds size budget')
            return archive.read(name)
        seq_name=category+'/sequence_annotations.jgz'
        packed=read(seq_name,8*1024**2)
        with gzip.GzipFile(fileobj=io.BytesIO(packed)) as g:
            seq_raw=g.read(32*1024**2+1)
        if len(seq_raw)>32*1024**2:raise ValueError('Oversized sequence metadata')
        seq=json.loads(seq_raw)
        lists=[json.loads(read(n,MAX_LIST)) for n in list_names]
        chosen=select(category,index,lists,seq)
        for n in list_names+[seq_name,category+'/LICENSE']:
            payload=read(n,MAX_LIST)
            write_identical(root/'files'/n,payload)
        return category,chosen,{'url':url,'archive_bytes':len(data),'archive_sha256':expected_sha,
               'all_metadata_member_crc_verified':True,'set_list_member_order':list_names,
               'selection_scope':'fewview_train/test only; empty list means no eligible test trajectory',
               'selected_sequence_count':len(chosen),'candidate_frame_count':sum(map(len,chosen.values())),
               'frame_annotations_member':category+'/frame_annotations.jgz',
               'geometry_annotations_preprocessed':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('data/co3d_test_metadata'))
    parser.add_argument('--output-json',type=Path,default=Path('results/co3d_test_selection_manifest.json'))
    args=parser.parse_args()
    args.data_root.mkdir(parents=True,exist_ok=True)
    with (args.data_root/'.metadata.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        links_raw=small_get(BASE+'links.json',1024**2)
        shas_raw=small_get(BASE+'co3d_sha256.json',1024**2)
        source=small_get(DUST3R_SOURCE,256*1024)
        # Do not execute remotely fetched Python; verify category literal only.
        import ast
        tree=ast.parse(source)
        categories=next(ast.literal_eval(n.value) for n in tree.body
                        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name)
                        and t.id=='CATEGORIES' for t in n.targets))
        if categories!=CATEGORIES:raise ValueError('Pinned preprocessor category order differs')
        for name,data in [('links.json',links_raw),('co3d_sha256.json',shas_raw),
                          ('preprocess_co3d_reference.py',source)]:
            write_identical(args.data_root/'references'/name,data)
        links,shas=json.loads(links_raw),json.loads(shas_raw)
        if set(links['full'])-{'METADATA'}!=set(CATEGORIES):
            raise ValueError('Unexpected official category universe')
        if shutil.disk_usage(args.data_root).free < RESERVE+len(CATEGORIES)*(MAX_ZIP+2*MAX_LIST):
            raise RuntimeError('Conservative metadata-only budget unavailable')
        chosen,records={},{}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures=[executor.submit(category_metadata,c,i,links['full'][c][0],
                      shas['full'][c+'_000.zip'],args.data_root) for i,c in enumerate(CATEGORIES)]
            for future in concurrent.futures.as_completed(futures):
                c,s,r=future.result();chosen[c]=s;records[c]=r
                print(f'{c}: {len(s)} selected test sequences',flush=True)
        chosen={c:chosen[c] for c in CATEGORIES}
        selected_path=args.data_root/'selected_seqs_test_reconstructed.json'
        save_identical(selected_path,chosen)
        report={'status':'official_metadata_selection_reconstructed_not_rgb_ready',
                'paper_mapping':'section 4.2 / Table 1 prerequisites only',
                'paper_categories_describe':'unseen trajectories from 41 object categories, NOT 41 unseen categories',
                'co3d_revision':CO3D_REV,'dust3r_revision':DUST3R_REV,
                'preprocessor_sha256':hashlib.sha256(source).hexdigest(),
                'links_sha256':hashlib.sha256(links_raw).hexdigest(),
                'checksums_sha256':hashlib.sha256(shas_raw).hexdigest(),
                'category_count_downloaded':len(CATEGORIES),
                'nonempty_selected_category_count':sum(bool(s) for s in chosen.values()),
                'selected_sequence_count':sum(len(s) for s in chosen.values()),
                'candidate_frame_count':sum(len(f) for s in chosen.values() for f in s.values()),
                'selected_manifest_sha256':sha(selected_path),'selected_by_category':chosen,
                'metadata_archives':{c:records[c] for c in CATEGORIES},
                'source_selection':{'split':'fewview_train lists / test key','quality_strictly_greater_than':.5,
                                    'max_sequences_per_category':50,'seed':42},
                'original_fast3r_scene_selection_equivalence':'unverified; author processed manifest absent',
                'rgb_depth_mask_and_camera_npz_ready':False,'formal_pose_metrics_available':False}
        save_identical(args.output_json,report)
        print('CO3D metadata selection complete; NOT a pose result',flush=True)


if __name__=='__main__':main()
