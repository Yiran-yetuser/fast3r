"""Synthetic 100-request uninterrupted/resumed equivalence; no real data or GPU."""
import argparse
import json
import random
from pathlib import Path
from co3d_sampling_state import capture, restore, digest
from test_co3d_sampling_state import fixture, draw
from fast3r_hf_re10k_pose_eval import sha, save_new

OUTPUT=Path('results/co3d_sampling_resume_synthetic_20261002.json')


def audit():
    mapping=list(range(100));bindings={'synthetic':True,'workers':0}
    d=fixture();random.seed(314)
    prefix=[draw(d,i) for i in range(37)]
    state=capture(d,mapping,37,bindings)
    suffix=[draw(d,i) for i in range(37,100)]
    final=capture(d,mapping,100,bindings)
    other=fixture();random.seed(999)
    cursor=restore(other,mapping,state,bindings)
    replay=[draw(other,i) for i in range(cursor,100)]
    end=capture(other,mapping,100,bindings)
    if suffix!=replay or final!=end: raise ValueError('Resume diverged')
    return {'status':'synthetic_sampling_resume_verified_not_real_CO3D',
        'source_sha256':{p:sha(p) for p in ('scripts/co3d_sampling_state.py',
            'scripts/test_co3d_sampling_state.py',__file__,
            'fast3r/dust3r/datasets/co3d_multiview.py')},
        'sample_count':len(prefix)+len(suffix),'resume_boundary':cursor,
        'suffix_request_count':len(suffix),'suffix_exact_equal':suffix==replay,
        'final_state_exact_equal':final==end,'boundary_state':state,
        'final_state_sha256':digest(final),'suffix_draws_sha256':digest(suffix),
        'invalid_scenes':[list(s) for s in sorted(other.invalid_scene_tracker)],
        'invalid_frame_flags':sum(sum(flags) for r in other.invalidate.values() for flags in r.values()),
        'real_RGB_or_depth_loaded':False,'model_forward_count':0,
        'formal_pose_metrics_available':False,'full_paper_completed':False,
        'limitations':['Synthetic invalid-depth fixture, not actual candidate draws',
            'Single-worker completed-request boundary only; not GPU RNG/model resume',
            'Real 100@ wrapper mapping and data runner not yet connected',
            'Caller must bind code/data/seed identities and commit result with state atomically']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-only',action='store_true')
    args=p.parse_args();result=audit()
    if args.verify_only:
        if result!=json.loads(OUTPUT.read_text()):raise ValueError('Archived synthetic audit changed')
    else:save_new(OUTPUT,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('boundary_state','source_sha256')},indent=2))
