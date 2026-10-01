#!/usr/bin/env python3
"""Pin the PoseDiffusion seen-category protocol; derive a candidate, not a score.

Remote source is parsed as AST data, never imported/executed. Filtering preserves
the already verified DUSt3R seed+original-51-category-index scene selections.
This does NOT establish equivalence to Fast3R's unpublished processed manifest.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from prepare_co3d_test_metadata import CATEGORIES, small_get, write_identical
from prepare_re10k_rgb_from_archive import save_identical, sha

POSEDIFFUSION_REV = 'b138198e2891a0f1a1c3435614b9490adb7fd4d6'
SOURCE_URL = ('https://raw.githubusercontent.com/facebookresearch/PoseDiffusion/'
              + POSEDIFFUSION_REV + '/pose_diffusion/datasets/co3d_v2.py')
CONFIG_URL = ('https://raw.githubusercontent.com/facebookresearch/PoseDiffusion/'
              + POSEDIFFUSION_REV + '/cfgs/default_test.yaml')
SOURCE_SHA = '1d4fe0f2329ca80b117fa28306f7c04fb15e8f8284c4cc1017ab7d4699533ffb'
CONFIG_SHA = 'b5d1fe81a5ecf2a33e7ce5420bc86caf51dad8b106790d2f9950c93093a5d4e2'
EXCLUDED = ['ball','book','couch','frisbee','hotdog','kite','remote',
            'sandwich','skateboard','suitcase']


def parse_categories(payload):
    found = {}
    for node in ast.parse(payload).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('TRAINING_CATEGORIES','TEST_CATEGORIES'):
                    if target.id in found:
                        raise ValueError('Duplicate protocol constant')
                    found[target.id] = ast.literal_eval(node.value)
    seen, unseen = found.get('TRAINING_CATEGORIES'), found.get('TEST_CATEGORIES')
    if (not isinstance(seen,list) or not isinstance(unseen,list)
            or any(not isinstance(c,str) for c in seen+unseen)
            or len(seen)!=41 or len(set(seen))!=41 or unseen!=EXCLUDED
            or set(seen)&set(unseen) or set(seen+unseen)!=set(CATEGORIES)):
        raise ValueError('Expected explicit disjoint 41 seen / 10 unseen categories')
    return seen, unseen


def filter_selection(selected, seen):
    if list(selected)!=CATEGORIES:
        raise ValueError('Original 51-category universe/order changed')
    result = {c:selected[c] for c in seen}
    for category, sequences in result.items():
        if not isinstance(sequences,dict) or not sequences:
            raise ValueError('Missing seen-category sequences')
        for sequence, frames in sequences.items():
            if (not isinstance(sequence,str) or not isinstance(frames,list)
                    or len(frames)<10 or any(type(f) is not int for f in frames)
                    or len(set(frames))!=len(frames)):
                raise ValueError('Malformed/duplicate/insufficient candidate frames')
    return result


def audit(root, output):
    root = Path(root)
    report_path = Path('results/co3d_test_selection_manifest.json')
    verified = json.loads(Path('results/co3d_test_selection_verified_20261002.json').read_text())
    report = json.loads(report_path.read_text())
    selected_path = root/'selected_seqs_test_reconstructed.json'
    if (sha(report_path)!=verified['selection_report_sha256']
            or sha(selected_path)!=report['selected_manifest_sha256']):
        raise ValueError('Verified parent selection has changed')
    source_path = root/'references/posediffusion_co3d_v2_reference.py'
    config_path = root/'references/posediffusion_default_test_reference.yaml'
    for path,url in [(source_path,SOURCE_URL),(config_path,CONFIG_URL)]:
        if not path.exists():write_identical(path,small_get(url,1024**2))
        if path.stat().st_size>1024**2:raise ValueError('Protocol source exceeds bound')
    payload=source_path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=SOURCE_SHA or sha(config_path)!=CONFIG_SHA:
        raise ValueError('Pinned source SHA mismatch')
    # Inspect only the relevant plain YAML scalar, without instantiating configs.
    import yaml
    config=yaml.safe_load(config_path.read_text())
    if config['test']['category']!='seen' or config['test']['num_frames']!=10:
        raise ValueError('Pinned evaluation config does not select seen / 10 frames')
    seen,unseen=parse_categories(payload)
    candidate=filter_selection(json.loads(selected_path.read_text()),seen)
    candidate_path=root/'selected_seqs_test_seen41_candidate.json'
    save_identical(candidate_path,candidate)
    result={
        'status':'seen41_category_protocol_audited_scene_equivalence_unverified',
        'paper_mapping':'Fast3R section 4.2 / Table 1 CO3D unseen trajectories within 41 categories',
        'category_match_basis':'Inference from cited PoseDiffusion seen protocol, not author manifest confirmation',
        'posediffusion_revision':POSEDIFFUSION_REV,'source_url':SOURCE_URL,
        'source_sha256':sha(source_path),'config_url':CONFIG_URL,'config_sha256':sha(config_path),
        'seen_categories':seen,'excluded_unseen_categories':unseen,
        'category_count':len(candidate),'selected_sequence_count':sum(map(len,candidate.values())),
        'candidate_frame_count':sum(len(f) for s in candidate.values() for f in s.values()),
        'sequence_counts_by_category':{c:len(s) for c,s in candidate.items()},
        'frame_counts_by_category':{c:sum(map(len,s.values())) for c,s in candidate.items()},
        'parent_selection_report_sha256':sha(report_path),
        'parent_selected_manifest_sha256':sha(selected_path),
        'candidate_manifest_sha256':sha(candidate_path),
        'selection_operation':'Filter existing 51-category DUSt3R candidate; no reseeding or frame changes',
        'original_fast3r_split_equivalence_verified':False,
        'rgb_depth_mask_and_camera_npz_ready':False,'formal_pose_metrics_available':False,
        'remaining_checks':['original processed sequence/frame list and 100@ sampling scope',
                            'official ZIP member inventory and bounded RGB/depth/mask budget',
                            'camera conversion/crop and exact 10-view sampling audit']}
    save_identical(output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('data/co3d_test_metadata'))
    parser.add_argument('--output-json',type=Path,default=Path('results/co3d_seen41_protocol_20261002.json'))
    args=parser.parse_args()
    audit(args.data_root,args.output_json)
