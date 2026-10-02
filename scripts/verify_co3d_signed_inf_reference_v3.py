"""Actual failed-frame reference equivalence; offline fixture, not benchmark."""
import gzip
import io
import json
import os
import random
import warnings
import zipfile
from pathlib import Path
import numpy as np
import torch
import co3d_lazy_dataset_v3 as v2
from audit_co3d_camera_metadata import ROOT
from prepare_re10k_rgb_from_archive import sha, save_identical
from probe_co3d_preprocess import exclusive_bytes

WORK = Path('data/co3d_signed_inf_reference_v3')
OUTPUT = Path('results/co3d_signed_inf_reference_v3_verified_20261002.json')
CATEGORY = 'stopsign'
SCENE = '599_92267_182752'
MEMBER = f'{CATEGORY}/{SCENE}/images/frame000042.jpg'


def verify():
    torch.set_num_threads(2)
    p = v2.LazyPreparer()
    annotation = p.annotations(CATEGORY)[MEMBER]
    raw_root = WORK/'raw'
    for key in ('image', 'depth', 'mask'):
        member = annotation[key]['path']
        source = Path('data/co3d_range_cache/raw')/member
        record = json.loads(Path(str(source)+'.json').read_text())
        if (sha(source) != record['sha256'] or source.stat().st_size != record['entry']['file_size']
                or not record['member_crc_verified']):
            raise ValueError('Cached source identity changed')
        target = raw_root/member
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != sha(source): raise ValueError('Fixture source differs')
        else: os.link(source, target)
    with zipfile.ZipFile(ROOT/'archives'/f'{CATEGORY}_000.zip') as z:
        packed = z.read(CATEGORY+'/sequence_annotations.jgz')
    with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
        raw = stream.read(32*1024**2+1)
    if len(raw) > 32*1024**2: raise ValueError('Sequence bound')
    sequences = [s for s in json.loads(raw) if s['sequence_name'] == SCENE]
    if len(sequences) != 1 or sequences[0]['viewpoint_quality_score'] <= .5:
        raise ValueError('Sequence quality/identity')
    exclusive_bytes(raw_root/CATEGORY/'frame_annotations.jgz',
                    gzip.compress(json.dumps([annotation]).encode(), mtime=0))
    exclusive_bytes(raw_root/CATEGORY/'sequence_annotations.jgz',
                    gzip.compress(json.dumps(sequences).encode(), mtime=0))
    exclusive_bytes(raw_root/CATEGORY/'set_lists/fixture_fewview_train.json',
                    json.dumps({'test': [[SCENE, annotation['frame_number'], MEMBER]]}).encode())
    ref_root = WORK/'reference_processed'
    marker = WORK/'reference_complete.json'
    expected_marker = {'reference_sha256': p.fingerprint['preprocessor_sha256'],
                       'fixture_sha256': sha(raw_root/CATEGORY/'frame_annotations.jgz')}
    saved_random = random.getstate()
    try:
        if not marker.exists():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                selected = p.ns['prepare_sequences'](CATEGORY, str(raw_root), str(ref_root),
                                                    512, 'test', .5, 50, 42, False)
            if selected != {SCENE: [42]}: raise ValueError('Reference fixture selection differs')
            save_identical(marker, expected_marker)
        elif json.loads(marker.read_text()) != expected_marker:
            raise ValueError('Reference fixture identity differs')
    finally: random.setstate(saved_random)
    # Cache fetch must be a hit: this proof never requires a network request.
    def forbidden_network(*args, **kwargs):
        raise AssertionError('Offline proof attempted network')
    from unittest.mock import patch
    with patch('co3d_range_cache.HttpRangeFile', forbidden_network), patch('co3d_range_cache.RecordedRanges', forbidden_network):
        record = p.ensure(CATEGORY, SCENE, 42)
    byte_checks = {}
    for key in ('image', 'depth', 'mask'):
        member = annotation[key]['path']
        if sha(ref_root/member) != sha(p.root/member): raise ValueError('Reference bytes differ')
        byte_checks[key] = sha(p.root/member)
    rel = Path(MEMBER).with_suffix('.npz')
    with np.load(ref_root/rel, allow_pickle=False) as ref, np.load(p.root/rel, allow_pickle=False) as adapted:
        if set(ref.files) != set(adapted.files): raise ValueError('NPZ keys differ')
        for key in ref.files:
            if ref[key].dtype != adapted[key].dtype or not np.array_equal(ref[key], adapted[key]):
                raise ValueError('Reference NPZ arrays/dtypes differ')
        if not np.isposinf(adapted['maximum_depth']): raise ValueError('Inf metadata lost')
    selected = json.loads(p.cache.manifest.read_text())
    dataset = v2.StrictLazyCo3d(selected, p)
    pool = dataset.scenes[CATEGORY, SCENE]
    index = pool.index(42); resolution = (512, 384)
    dataset.invalidate[CATEGORY, SCENE][resolution] = [False]*len(pool)
    rng = np.random.default_rng(777)
    rng_state = rng.bit_generator.state
    loaded = v2.STRICT_LOAD(dataset, CATEGORY, SCENE, pool, index, resolution, rng)
    if loaded is not None or not dataset.invalidate[CATEGORY, SCENE][resolution][index]:
        raise ValueError('Strict original loader did not invalidate zero reference depth')
    dataset.ROOT = str(ref_root)
    dataset.invalidate[CATEGORY, SCENE][resolution][index] = False
    rng.bit_generator.state = rng_state
    released_loaded = v2.released.Co3d_Multiview._load_view_data(
        dataset, CATEGORY, SCENE, pool, index, resolution, rng)
    if released_loaded is not None or not dataset.invalidate[CATEGORY, SCENE][resolution][index]:
        raise ValueError('Released original loader/reference fixture disagrees')
    if p.cache.network_bytes != 0: raise ValueError('Unexpected network bytes')
    report = {'status': 'real_signed_inf_frame_reference_bytes_npz_and_original_loader_verified',
        'paper_mapping': 'section 4.2 / Table 1 CO3D input preparation only',
        'member': MEMBER, 'v3_fingerprint': p.fingerprint,
        'verification_code_sha256': sha(__file__), 'processed_modality_sha256': byte_checks,
        'image_depth_mask_bytes_equal': True, 'npz_arrays_and_dtypes_equal': True,
        'maximum_depth_preserved': 'positive_infinity',
        'original_loader_returns_none_and_invalidates': True,
        'strict_loader_returns_none_and_invalidates': True,
        'prepared_record': record, 'network_bytes': 0, 'model_forward_count': 0,
        'continuous_sampling_resumed': False, 'prior_prefix_v2_equivalence_verified': False,
        'formal_pose_metrics_available': False, 'full_paper_completed': False}
    save_identical(OUTPUT, report)
    return report


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2, allow_nan=False))
