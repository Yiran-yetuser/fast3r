"""JSON-only request-boundary state for single-worker CO3D sampling.

Not a GPU/model checkpoint or multiworker resume. Restore rejects changed pool,
combinations, wrapper mapping and caller protocol before mutating live state.
"""
import copy
import hashlib
import json
import random
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                     allow_nan=False).encode()).hexdigest()


def identity(dataset, mapping, bindings):
    return {'pool_sha256':digest([[list(s),dataset.scenes[s]] for s in dataset.scene_list]),
            'combinations_sha256':digest(dataset.combinations),
            'mapping':[int(x) for x in mapping], 'bindings':bindings,
            'dataset_seed':getattr(dataset,'seed',None), 'num_views':dataset.num_views}


def capture(dataset, mapping, next_request, bindings):
    if type(next_request) is not int or not 0 <= next_request <= len(mapping):
        raise ValueError('Invalid request boundary')
    rows=[]
    for scene in dataset.scene_list:
        rows.append({'scene':list(scene),'resolutions':[
            {'resolution':list(res),'invalid':list(flags)}
            for res,flags in sorted(dataset.invalidate[scene].items())]})
    return json.loads(json.dumps({'version':1,'identity':identity(dataset,mapping,bindings),
        'next_request':next_request,'invalidate':rows,
        'invalid_scene_tracker':[list(s) for s in sorted(dataset.invalid_scene_tracker)],
        'python_random_state':random.getstate(),
        'dataset_rng':copy.deepcopy(dataset._rng.bit_generator.state) if hasattr(dataset,'_rng') else None,
        'scope':'single_worker_completed_request_boundary_sampling_only'},allow_nan=False))


def tuples(value):
    return tuple(tuples(x) for x in value) if isinstance(value,list) else value


def restore(dataset, mapping, state, bindings):
    if state.get('version')!=1 or state.get('identity')!=identity(dataset,mapping,bindings):
        raise ValueError('Sampling identity changed')
    n=state['next_request']
    if type(n) is not int or not 0<=n<=len(mapping): raise ValueError('Invalid cursor')
    if state.get('scope')!='single_worker_completed_request_boundary_sampling_only':
        raise ValueError('Invalid checkpoint scope')
    rows=state['invalidate']
    if [tuple(r['scene']) for r in rows]!=dataset.scene_list: raise ValueError('Scene order changed')
    invalid={}
    for row in rows:
        scene=tuple(row['scene']); resolutions={}
        for item in row['resolutions']:
            res=tuple(item['resolution']); flags=item['invalid']
            if (len(res)!=2 or any(type(x) is not int or x<=0 for x in res)
                or res in resolutions or len(flags)!=len(dataset.scenes[scene])
                or any(type(x) is not bool for x in flags)):
                raise ValueError('Invalid frame mask state')
            resolutions[res]=flags.copy()
        invalid[scene]=resolutions
    tracker=[tuple(s) for s in state['invalid_scene_tracker']]
    if len(tracker)!=len(set(tracker)) or not set(tracker)<=set(dataset.scene_list):
        raise ValueError('Invalid scene tracker')
    py_state=tuples(state['python_random_state']); random.Random().setstate(py_state)
    rng_state=state['dataset_rng']; rng=None
    if rng_state is not None:
        if rng_state.get('bit_generator')!='PCG64': raise ValueError('Unsupported bit generator')
        rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(rng_state)
    # All structural/RNG checks passed; commit changes only now.
    dataset.invalidate=invalid;dataset.invalid_scene_tracker=set(tracker)
    if rng is not None: dataset._rng=rng
    elif hasattr(dataset,'_rng'): del dataset._rng
    random.setstate(py_state)
    return n
