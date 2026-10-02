#!/usr/bin/env python3
"""Fail-closed lazy CO3D data wiring on the complete original candidate pool.

Version 2 preserves positive-Inf reference quantization and NPZ metadata.
NaN, negative depth and IO failures still abort. V1 files remain immutable.
No replacement sparse split, eviction or model inference. Released sampling,
oversampling, scene retry, crop and base __getitem__ remain inherited. The only
loader semantic change is propagation of IO/decode errors instead of catch-all
None. True all-zero masked depth still returns None and is traced explicitly.
"""
import ast
import copy
import gzip
import hashlib
import io
import json
import random
import warnings
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

from fast3r.dust3r.datasets import co3d_multiview as released
from fast3r.dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from fast3r.dust3r.datasets.utils import cropping
from co3d_range_cache import RawCache, RESERVE
from audit_co3d_camera_metadata import ROOT, MAX_JSON, MAX_PACKED, annotation_mapping
from probe_co3d_preprocess import preprocessing_namespace, reference_modules, save_processed, MAX_PIXELS
from prepare_re10k_rgb_from_archive import sha, save_identical

LOADER = Path('fast3r/dust3r/datasets/co3d_multiview.py')
LOADER_SHA = 'd307fd6e75b97f47a2efeefcd4ef5fd6f4fce44536b0d3824fa43928539e02f0'
MAX_PROCESSED = 512 * 1024**2


def uncaught_source_loader():
    """Extract the local pinned method verbatim, removing ONLY its catch-all.

    Hash and AST shape guards make source drift explicit. No downloaded code is
    executed here; no change is made to the repository's historical loader.
    """
    if sha(LOADER) != LOADER_SHA:
        raise ValueError('Released loader changed; review adaptation before use')
    tree = ast.parse(LOADER.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Co3d_Multiview')
    method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_load_view_data'))
    if (len(method.body) != 2 or not isinstance(method.body[1], ast.Try)
            or method.body[1].orelse or method.body[1].finalbody
            or len(method.body[1].handlers) != 1
            or not isinstance(method.body[1].handlers[0].type, ast.Name)
            or method.body[1].handlers[0].type.id != 'Exception'):
        raise ValueError('Unexpected source catch-all shape')
    method.body = [method.body[0], *method.body[1].body]
    method.name = '_uncaught_load_view_data'
    ns = dict(vars(released))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(LOADER), 'exec'), ns)
    return ns[method.name]


STRICT_LOAD = uncaught_source_loader()


def process_frame_allow_zero(paths, annotation, ns):
    # Copyright (C) 2024-present Naver Corporation. All rights reserved.
    # Derived from pinned DUSt3R preprocessing, CC BY-NC-SA 4.0.
    # See NOTICES_CO3D_PREPROCESSING.md. Positive-depth arithmetic/order unchanged.
    with PIL.Image.open(paths['images']) as im:
        if im.width * im.height > MAX_PIXELS: raise ValueError('RGB pixel bound')
        image = im.convert('RGB')
    with PIL.Image.open(paths['depths']) as im:
        if im.width * im.height > MAX_PIXELS: raise ValueError('Depth pixel bound')
        depth = np.asarray(im, dtype=np.uint16).view(np.float16).astype(np.float32)
    with PIL.Image.open(paths['masks']) as im:
        if im.width * im.height > MAX_PIXELS: raise ValueError('Mask pixel bound')
    mask = plt.imread(paths['masks'])
    if (list(depth.shape) != annotation['image']['size'] or mask.shape != depth.shape
            or image.size != depth.shape[::-1] or np.isnan(depth).any()
            or not np.isfinite(mask).all() or (depth < 0).any() or (mask < 0).any() or (mask > 1).any()
            or annotation['depth']['scale_adjustment'] != 1.):
        raise ValueError('Malformed depth/mask/geometry; never sample past errors')
    v = annotation['viewpoint']
    if v['intrinsics_format'] != 'ndc_isotropic': raise ValueError('Unexpected K format')
    R, t, K = ns['opencv_from_cameras_projection'](np.array(v['R']), np.array(v['T']),
        np.array(v['focal_length']), np.array(v['principal_point']), np.array(annotation['image']['size']))
    K = K.numpy(); H, W = depth.shape
    cx, cy = K[:2, 2].round().astype(int)
    mx, my = min(cx, W-cx), min(cy, H-cy)
    if min(mx, my) <= 0: raise ValueError('Invalid principal-point crop')
    bbox = (cx-mx, cy-my, cx+mx, cy+my)
    image, dm, K = cropping.crop_image_depthmap(image, np.stack((depth, mask), axis=-1), K, bbox)
    scale = (384/min(H, W)) + 1e-8
    target = np.floor(np.array([W, H])*scale).astype(int)
    if max(target) < 512:
        scale = (512/max(H, W)) + 1e-8
        target = np.floor(np.array([W, H])*scale).astype(int)
    image, dm, K = cropping.rescale_image_depthmap(image, dm, K, target)
    depth, mask = dm[:, :, 0], dm[:, :, 1]
    maximum = np.max(depth)
    if np.isnan(maximum) or maximum < 0: raise ValueError('Invalid max depth')
    # Preserve pinned reference arithmetic, including Inf NPZ metadata. Do not
    # replace raw Inf pixels. NumPy converts Inf/Inf NaNs to zero uint16 here.
    with warnings.catch_warnings(), np.errstate(invalid='ignore', divide='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)
        quantized = (depth/maximum*65535).astype(np.uint16)
    if np.isposinf(maximum) and quantized.any():
        raise ValueError('Unexpected nonzero reference Inf quantization')
    w2c = np.eye(4, dtype=np.float32); w2c[:3, :3] = R; w2c[:3, 3] = t
    pose = np.linalg.inv(w2c)
    return image, quantized, (mask*255).astype(np.uint8), K, pose, maximum, {
        'raw_size_hw': [H, W], 'processed_size_hw': [image.height, image.width],
        'raw_depth_max': audit_maximum(maximum), 'zero_maximum_depth': bool(maximum == 0),
        'positive_inf_reference_quantization': bool(np.isposinf(maximum))}


def audit_maximum(value):
    if np.isposinf(value): return 'positive_infinity'
    if not np.isfinite(value) or value < 0: raise ValueError('Invalid audited maximum')
    return float(value)


class LazyPreparer:
    """Separate raw/processed roots; bounded journals, no overwrite of history."""
    def __init__(self, root=Path('data/co3d_lazy_v2_processed')):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache = RawCache()
        self.categories = {}
        parent = json.loads(Path('results/co3d_test_selection_manifest.json').read_text())
        self.parent = parent
        crop, crop_path, geometry_path = reference_modules()
        reference = ROOT/'references/preprocess_co3d_reference.py'
        if sha(reference) != parent['preprocessor_sha256']: raise ValueError('Preprocessing reference changed')
        probe = json.loads(Path('results/co3d_preprocess_probe_20261002.json').read_text())
        for path, key in [(crop_path, 'cropping_reference_sha256'), (geometry_path, 'geometry_reference_sha256')]:
            if sha(path) != probe['fingerprint'][key]: raise ValueError('Reviewed reference helpers changed')
        self.ns = preprocessing_namespace(crop)
        self.fingerprint = {'protocol_version': 'reference_positive_inf_v2',
            'lazy_code_sha256': sha(Path(__file__)),
            'frozen_v1_sha256': sha(Path('scripts/co3d_lazy_dataset.py')),
            'candidate_sha256': sha(self.cache.manifest), 'cache': self.cache.fingerprint,
            'preprocessor_sha256': sha(reference), 'local_cropping_sha256': sha(Path(cropping.__file__)),
            'runtime': {'numpy': np.__version__, 'opencv': cv2.__version__, 'pillow': PIL.__version__}}
        self.prepared_records = {}

    def budget(self, additional=0):
        used = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        if used + additional > MAX_PROCESSED or shutil.disk_usage(self.root).free < RESERVE + additional:
            raise RuntimeError('Processed 512MiB bound or 1GiB reserve; no automatic eviction')

    def annotations(self, category):
        if category in self.categories: return self.categories[category]
        archive = ROOT/'archives'/f'{category}_000.zip'
        meta = self.parent['metadata_archives'][category]
        if archive.stat().st_size > MAX_PACKED or sha(archive) != meta['archive_sha256']:
            raise ValueError('Metadata ZIP identity changed')
        with zipfile.ZipFile(archive) as z:
            name = category+'/frame_annotations.jgz'
            if z.getinfo(name).file_size > MAX_PACKED: raise ValueError('Packed annotation bound')
            packed = z.read(name)
            rows = []
            for member in meta['set_list_member_order']:
                rows.extend(json.loads(z.read(member))['test'])
        with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream: raw = stream.read(MAX_JSON+1)
        if len(raw) > MAX_JSON: raise ValueError('Annotation JSON bound')
        mapping = annotation_mapping(category, self.cache.selected[category], rows)
        wanted = set(mapping.values()); found = {}
        for a in json.loads(raw):
            key = (a['sequence_name'], a['frame_number'])
            if key not in wanted: continue
            path = a['image']['path']
            if key in found or mapping.get(path) != key: raise ValueError('Annotation ID/path mismatch')
            stem = Path(path).stem
            if (a['depth']['path'] != f'{category}/{key[0]}/depths/{stem}.jpg.geometric.png'
                    or a['mask']['path'] != f'{category}/{key[0]}/masks/{stem}.png'):
                raise ValueError('Raw modality paths mismatch')
            found[key] = a
        if set(found) != wanted: raise ValueError('Missing candidate annotations')
        self.categories[category] = {path: found[key] for path, key in mapping.items()}
        return self.categories[category]

    def ensure(self, category, scene, frame):
        path = f'{category}/{scene}/images/frame{frame:06d}.jpg'
        annotation = self.annotations(category)[path]
        raw, members = {}, []
        for kind, key in [('images', 'image'), ('depths', 'depth'), ('masks', 'mask')]:
            member = annotation[key]['path']; raw[kind], record = self.cache.fetch(member)
            members.append({'path': member, 'sha256': record['sha256'], 'bytes': record['entry']['file_size'],
                'member_crc_verified': record['member_crc_verified'], 'full_archive_sha_verified': False})
        values = process_frame_allow_zero(raw, annotation, self.ns)
        # Conservative per-frame allowance covers encodings/journal/NPZ before writes.
        record_path = self.root/(path+'.prepared.json')
        if not record_path.exists(): self.budget(8*1024**2)
        targets, metadata = save_processed(self.root, annotation, values)
        record = {'fingerprint': self.fingerprint, 'image_path': path,
            'annotation_frame_number': annotation['frame_number'], 'raw_members': members,
            'processed_sha256': {str(p.relative_to(self.root)): sha(p) for p in [*targets, metadata]},
            'maximum_depth': audit_maximum(values[5]), 'stats': values[-1]}
        save_identical(record_path, record)
        self.budget()
        self.prepared_records[path] = record
        print('LAZY PREPARED', path, flush=True)
        return record


class StrictLazyCo3d(released.Co3d_Multiview):
    def __init__(self, selected, preparer, combination_seed=42, dataset_seed=777):
        BaseStereoViewDataset.__init__(self, split='test', resolution=(512,384), seed=dataset_seed)
        self.num_views = 10; self.mask_bg = True; self.preparer = preparer; self.ROOT = str(preparer.root)
        self.scenes = {(c,s): list(f) for c,scenes in selected.items() for s,f in scenes.items()}
        self.scene_list = list(self.scenes)
        self.invalidate = {s: {} for s in self.scene_list}; self.invalid_scene_tracker = set()
        saved_state = random.getstate()
        try:
            random.seed(combination_seed); self._generate_combinations(100,360,100)
        finally: random.setstate(saved_state)
        self.trace = []; self.pool_attempts = []; self.active_request = None

    def _fetch_views_for_pool(self, obj, instance, image_pool, resolution, rng):
        self.pool_attempts.append({'category': obj, 'scene': instance, 'candidate_pool_length': len(image_pool)})
        return super()._fetch_views_for_pool(obj, instance, image_pool, resolution, rng)

    def _load_view_data(self, obj, instance, image_pool, im_idx, resolution, rng):
        event = {'category': obj, 'scene': instance, 'pool_index': int(im_idx),
                 'frame_number': int(image_pool[im_idx]), 'request': self.active_request}
        self.trace.append(event)
        try:
            self.preparer.ensure(obj, instance, image_pool[im_idx])
            result = STRICT_LOAD(self, obj, instance, image_pool, im_idx, resolution, rng)
            event['status'] = 'valid' if result is not None else 'zero_masked_depth_after_crop'
            if result is not None:
                if not all(np.isfinite(result[k]).all() for k in ('depthmap','camera_pose','camera_intrinsics')):
                    raise ValueError('Nonfinite loaded geometry')
                event['image_size_wh'] = list(result['img'].size)
                event['positive_depth_fraction'] = float(np.mean(result['depthmap'] > 0))
            return result
        except Exception as error:
            event['status'] = 'hard_error'; event['error_type'] = type(error).__name__
            raise  # never resample through an IO/CRC/GT failure
