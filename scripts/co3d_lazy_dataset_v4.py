"""V4 finite signed depth reference arithmetic; immutable V1-V3 lineage.

Raw finite signed pixels are not filled/deleted: exact reference uint16 cast
is retained and audited. NaN, negative processed maximum and unproved finite-
maximum/-Inf cases still abort. No original sampler or loader changes.
"""
from pathlib import Path
import warnings
import cv2
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from fast3r.dust3r.datasets.utils import cropping
from probe_co3d_preprocess import MAX_PIXELS, save_processed
from prepare_re10k_rgb_from_archive import sha, save_identical
from co3d_lazy_dataset_v3 import LazyPreparer as FrozenPreparer, StrictLazyCo3d, STRICT_LOAD, released, audit_maximum


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
            or not np.isfinite(mask).all()  or (mask < 0).any() or (mask > 1).any()
            or annotation['depth']['scale_adjustment'] != 1.):
        raise ValueError('Malformed depth/mask/geometry; never sample past errors')
    raw_negative_inf_count = int(np.isneginf(depth).sum())
    raw_negative_count = int((depth < 0).sum())
    raw_positive_inf_count = int(np.isposinf(depth).sum())
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
    if raw_negative_inf_count and not np.isposinf(maximum):
        raise ValueError('Negative Inf with finite maximum is not yet verified')
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
        'positive_inf_reference_quantization': bool(np.isposinf(maximum)),
        'raw_negative_depth_count': raw_negative_count,
        'processed_negative_depth_count': int((depth < 0).sum()),
        'negative_quantized_positive_count': int((quantized[depth < 0] > 0).sum()), 'raw_positive_inf_count': raw_positive_inf_count}


class LazyPreparer(FrozenPreparer):
    def __init__(self, root=Path('data/co3d_lazy_v4_processed')):
        super().__init__(root)
        self.fingerprint.update(protocol_version='reference_finite_signed_v4',
            frozen_v3_sha256=sha(Path('scripts/co3d_lazy_dataset_v3.py')),
            lazy_code_sha256=sha(Path(__file__)))

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
