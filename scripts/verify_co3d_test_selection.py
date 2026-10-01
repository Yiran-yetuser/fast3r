#!/usr/bin/env python3
"""Read-only independent check of metadata selection, not RGB or pose metrics."""
import argparse
import gzip
import hashlib
import json
import random
import zipfile
from pathlib import Path

from prepare_co3d_test_metadata import CATEGORIES, CO3D_REV, DUST3R_REV
from prepare_re10k_rgb_from_archive import sha, save_identical


def verify(root, path):
    root, path=Path(root),Path(path)
    report=json.loads(path.read_text())
    checksums=json.loads((root/'references/co3d_sha256.json').read_text())['full']
    for file,key in [('links.json','links_sha256'),('co3d_sha256.json','checksums_sha256'),
                     ('preprocess_co3d_reference.py','preprocessor_sha256')]:
        if sha(root/'references'/file)!=report[key]:raise ValueError('Reference hash differs')
    if report['co3d_revision']!=CO3D_REV or report['dust3r_revision']!=DUST3R_REV:
        raise ValueError('Source revision differs')
    if sha(root/'selected_seqs_test_reconstructed.json')!=report['selected_manifest_sha256']:
        raise ValueError('Selected manifest hash differs')
    selected=json.loads((root/'selected_seqs_test_reconstructed.json').read_text())
    if selected!=report['selected_by_category'] or list(selected)!=CATEGORIES:
        raise ValueError('Category universe/order differs')
    for index,c in enumerate(CATEGORIES):
        record=report['metadata_archives'][c]
        archive=root/'archives'/(c+'_000.zip')
        if (sha(archive)!=checksums[c+'_000.zip'] or archive.stat().st_size!=record['archive_bytes']
                or record['archive_sha256']!=checksums[c+'_000.zip']):
            raise ValueError('Official archive hash/length differs')
        with zipfile.ZipFile(archive) as z:
            expected_lists=[n for n in z.namelist() if n.startswith(c+'/set_lists/')
                            and 'fewview_train' in n and n.endswith('.json')]
            if record['set_list_member_order']!=expected_lists:
                raise ValueError('Source set-list order differs')
            for member in expected_lists+[c+'/sequence_annotations.jgz']:
                if z.getinfo(member).file_size>128*1024**2:
                    raise ValueError('Metadata verification member too large')
                if sha(root/'files'/member)!=hashlib.sha256(z.read(member)).hexdigest():
                    raise ValueError('Saved selection metadata differs from official ZIP')
        with gzip.open(root/'files'/c/'sequence_annotations.jgz') as g:sequences=json.load(g)
        good={s['sequence_name'] for s in sequences if s.get('viewpoint_quality_score') is not None
              and s['viewpoint_quality_score']>.5}
        rows=[]
        for member in record['set_list_member_order']:
            rows.extend(json.loads((root/'files'/member).read_text())['test'])
        eligible=sorted({r[0] for r in rows}&good)
        chosen=eligible if len(eligible)<50 else random.Random(42+index).sample(eligible,50)
        if list(selected[c])!=chosen:raise ValueError('Sequence selection differs')
        frame_lists={s:[] for s in chosen}
        for scene,number,filename in rows:
            if scene in frame_lists:frame_lists[scene].append(int(Path(filename).name[5:11]))
        if frame_lists!=selected[c]:raise ValueError('Candidate frame lists/order differ')
        if record['selected_sequence_count']!=len(chosen):raise ValueError('Category count differs')
        if record['candidate_frame_count']!=sum(map(len,frame_lists.values())):
            raise ValueError('Category frame count differs')
        if any(len(f)<10 or len(f)!=len(set(f)) for f in frame_lists.values()):
            raise ValueError('Duplicate or insufficient frames')
    categories=sum(bool(v) for v in selected.values())
    sequences=sum(map(len,selected.values()))
    frames=sum(len(f) for s in selected.values() for f in s.values())
    if (categories!=report['nonempty_selected_category_count']
            or sequences!=report['selected_sequence_count'] or frames!=report['candidate_frame_count']):
        raise ValueError('Aggregate counts differ')
    return {'status':'metadata_sha_and_independent_selection_verified_not_rgb_ready',
            'selection_report_sha256':sha(path),'official_metadata_archive_count':len(CATEGORIES),
            'metadata_archive_bytes':sum(r['archive_bytes'] for r in report['metadata_archives'].values()),
            'nonempty_category_count':categories,'selected_sequence_count':sequences,
            'candidate_frame_count':frames,'empty_categories':[c for c,v in selected.items() if not v],
            'original_fast3r_split_equivalence_verified':False,'rgb_depth_and_pose_npz_ready':False,
            'formal_pose_metrics_available':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-json',type=Path,required=True)
    parser.add_argument('--summary-json',type=Path,default=Path('results/co3d_test_selection_summary_20261002.json'))
    args=parser.parse_args()
    result=verify('data/co3d_test_metadata','results/co3d_test_selection_manifest.json')
    save_identical(args.output_json,result)
    source=json.loads(Path('results/co3d_test_selection_manifest.json').read_text())
    summary={k:v for k,v in source.items() if k!='selected_by_category'}
    summary['full_selection_report_sha256']=result['selection_report_sha256']
    save_identical(args.summary_json,summary)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
