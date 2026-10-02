#!/usr/bin/env python3
"""CPU pixel-geometry audit and fixed-frame landscape diagnostic inputs.

No historical code/data/weights are changed. The alternative crop removes
ONLY the automatic portrait/square target-resolution reversal from the pinned
base method. No new frames, resampling, downloads or GT focal substitution.
"""
import argparse
import ast
import copy
import inspect
import json
import textwrap
from pathlib import Path

import cv2
import numpy as np
import torch
from fast3r.dust3r.datasets.base import base_stereo_view_dataset as base
from co3d_lazy_dataset_v4 import STRICT_LOAD
from diagnose_co3d_candidate_pnp import load_fixed_request
from fast3r_hf_co3d_100_pose_eval import (CHECKPOINT, PREP_ROOT, PREFIX_PROOF,
    build_dataset, tensor_sha)
from fast3r_hf_co3d_pose_smoke import signature
from fast3r_hf_re10k_pose_eval import sha
from verify_co3d_candidate_pose_report import REPORT, SUMMARY

REQUESTS = (0, 2, 3)
BASE_SHA = 'e3aa91f3722ce54b022bc65b282adc642e20d676640fd0c60eacc3b290e46a62'
OUTPUT = Path('results/co3d_portrait_geometry_audit_20261003.json')


def force_landscape_method():
    """AST-guarded local alternative, not a mutation of the released class."""
    if sha(base.__file__) != BASE_SHA:
        raise ValueError('Pinned base crop changed')
    source = textwrap.dedent(inspect.getsource(base.BaseStereoViewDataset._crop_resize_if_necessary))
    tree = ast.parse(source); method = tree.body[0]
    expected = ast.dump(ast.parse('H > 1.1 * W', mode='eval').body)
    nodes = [n for n in method.body if isinstance(n, ast.If) and ast.dump(n.test) == expected]
    if (len(nodes) != 1 or len(nodes[0].orelse) != 1 or not isinstance(nodes[0].orelse[0], ast.If)
            or 'rng.integers(2)' not in ast.unparse(nodes[0])):
        raise ValueError('Unexpected portrait/square crop conditional')
    method.body.remove(nodes[0]); method.name = '_force_landscape_crop'
    ns = dict(vars(base)); exec(compile(tree, base.__file__, 'exec'), ns)
    return ns[method.name]


class FixedLandscapeViews(base.BaseStereoViewDataset):
    _crop_resize_if_necessary = force_landscape_method()

    def __init__(self, original_views, preparer):
        super().__init__(split='test', resolution=(512, 384), seed=777)
        self.original_views = original_views; self.preparer = preparer
        self.ROOT = str(preparer.root); self.mask_bg = True; self.invalidate = {}

    def _get_views(self, idx, resolution, rng):
        result = []
        for v in self.original_views:
            category, scene = v['label'].split('/')
            frame = int(Path(v['instance']).stem.removeprefix('frame'))
            self.preparer.ensure(category, scene, frame)
            self.invalidate[category, scene] = {resolution: {}}
            view = STRICT_LOAD(self, category, scene, [frame], 0, resolution, rng)
            if view is None:
                raise ValueError('Fixed frame became invalid after alternative crop; never replace or drop it')
            result.append(view)
        return result


def load_context():
    """Check frozen manifests/hashes, not rerun the archived 100-input replay."""
    baseline = json.loads(REPORT.read_text()); summary = json.loads(SUMMARY.read_text())
    proof = json.loads(PREFIX_PROOF.read_text())
    if (sha(REPORT) != summary['complete_report_sha256'] or baseline['formal_Table1_result']
            or proof['verified_request_count'] != 100 or not proof['all_100_requests_prepared']
            or sha(PREFIX_PROOF) != baseline['protocol']['full_prefix_proof_sha256']):
        raise ValueError('Archived scope/baseline/prefix proof changed')
    initial = json.loads((PREP_ROOT/'initial.json').read_text())
    if sha(PREP_ROOT/'initial.json') != proof['initial_sha256']:
        raise ValueError('Initial journal changed')
    rows = []
    for i in range(100):
        p = PREP_ROOT/f'request_{i:03d}.json'
        if sha(p) != proof['request_sha256'][p.name]: raise ValueError('Prepared transaction changed')
        rows.append(json.loads(p.read_text()))
    for p, h in {**initial['identity']['bindings']['code_sha256'],
                 **baseline['protocol']['source_code_sha256']}.items():
        if sha(p) != h: raise ValueError('Archived source changed: '+p)
    for p, k in [('model.safetensors', 'checkpoint_weight_sha256'), ('config.json', 'checkpoint_config_sha256')]:
        if sha(CHECKPOINT/p) != baseline['protocol'][k]: raise ValueError('Public checkpoint changed')
    return baseline, initial, rows, proof, build_dataset(initial, rows)


def fixed_inputs(i, context):
    baseline, initial, rows, proof, (dataset, wrapper, mapping, bindings) = context
    native = load_fixed_request(i, initial, rows, dataset, wrapper, mapping, bindings)
    # Same selected frame order and duplicates. Base __getitem__ RNG here is
    # independent: changed square-crop decisions may consume different RNG.
    # It is NOT claimed to reproduce the sampler after-state under a new crop.
    forced = FixedLandscapeViews(native, dataset.preparer)[mapping[i]]
    if ([(v['label'], v['instance']) for v in forced] != [(v['label'], v['instance']) for v in native]
            or any(not np.array_equal(a['camera_pose'], b['camera_pose']) for a, b in zip(native, forced))
            or any(v['true_shape'].tolist() != [384, 512] for v in forced)
            or [tensor_sha(v['img']) for v in native] != baseline['requests'][i]['input_tensor_sha256']):
        raise ValueError('Fixed frame/GT/input identity or forced crop shape mismatch')
    return native, forced


def token_layout_probe():
    """Actual published patch class, synthetic labelled patches, no weights/model."""
    from fast3r.dust3r.patch_embed import PatchEmbedDust3R
    grid = torch.arange(24*32, dtype=torch.float32).reshape(24, 32)
    img = grid.repeat_interleave(16, 0).repeat_interleave(16, 1)[None, None].repeat(1, 3, 1, 1)
    patch = PatchEmbedDust3R(img_size=512, patch_size=16, in_chans=3, embed_dim=1)
    with torch.no_grad():
        patch.proj.weight.zero_(); patch.proj.weight[0, 0, 0, 0] = 1; patch.proj.bias.zero_()
        a, pa = patch(img, true_shape=torch.tensor([[384, 512]]))
        b, pb = patch(img, true_shape=torch.tensor([[512, 384]]))
    # The DPT head uses (true_H/16, true_W/16), while this encoder ignores true_shape.
    reshaped = b[0, :, 0].reshape(32, 24); transposed = grid.T
    return {'patch_class': type(patch).__name__, 'fixture_not_trained_model': True,
        'encoder_tokens_and_positions_ignore_true_shape': bool(torch.equal(a, b) and torch.equal(pa, pb)),
        'encoder_token_grid_hw': [24, 32], 'portrait_head_token_grid_hw': [32, 24],
        'head_reshape_is_not_spatial_transpose': not torch.equal(reshaped, transposed),
        'token_label_mismatch_fraction': float((reshaped != transposed).float().mean())}


def project(points, w2c, K):
    camera = points.reshape(-1, 3) @ w2c[:3, :3].T + w2c[:3, 3]
    homogeneous = camera @ K.T
    return homogeneous[:, :2]/homogeneous[:, 2:], camera[:, 2]


def synthetic_geometry_probe():
    from fast3r.dust3r.cloud_opt.init_im_poses import fast_pnp
    H, W, f = 80, 60, 100.
    y, x = np.mgrid[:H, :W]; z = 3.+.3*np.sin(x*.18)+.5*np.cos(y*.16)
    camera = np.stack([(x-W/2)*z/f, (y-H/2)*z/f, z], -1)
    R = cv2.Rodrigues(np.array([.2, -.1, .3]))[0]; t = np.array([.3, -.4, 1.])
    world = (camera-t) @ R; w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = t
    K = np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1.]])
    swapped = world.swapaxes(0, 1); yt, xt = np.mgrid[:W, :H]
    pixels = np.stack([xt, yt], -1).reshape(-1, 2)
    exact, _ = project(swapped, w2c, K[[1, 0, 2]])
    standard = np.array([[f, 0, H/2], [0, f, W/2], [0, 0, 1.]])
    incorrect, _ = project(swapped, w2c, standard)
    rows = []
    for name, pts in [('native_portrait', world), ('transposed_pixels_standard_K', swapped)]:
        cv2.setRNGSeed(42)
        focal, pose = fast_pnp(torch.from_numpy(pts.astype(np.float32)), f,
            torch.ones(pts.shape[:2], dtype=torch.bool), 'cpu', niter_PnP=100)
        if pose is None: raise ValueError('Synthetic PnP failed unexpectedly')
        estimate = np.linalg.inv(pose.numpy()); uv, depth = project(pts, estimate,
            K if name == 'native_portrait' else standard)
        yy, xx = np.mgrid[:pts.shape[0], :pts.shape[1]]
        residual = np.linalg.norm(uv-np.stack([xx, yy], -1).reshape(-1, 2), axis=1)
        angle = np.rad2deg(np.arccos(np.clip((np.trace(estimate[:3, :3]@R.T)-1)/2, -1, 1)))
        rows.append({'branch': name, 'rotation_error_deg': float(angle),
            'median_reprojection_error_px': float(np.median(residual)),
            'max_reprojection_error_px': float(residual.max()),
            'positive_depth_fraction': float(np.mean(depth > 0)), 'PnP_returned_success': True})
    return {'scope': 'synthetic_perfect_world_points_not_model_or_benchmark',
        'row_swap_matrix_determinant': -1.,
        'swapped_K_exact_max_pixel_error': float(np.max(np.abs(exact-pixels))),
        'standard_K_true_pose_median_pixel_error': float(np.median(np.linalg.norm(incorrect-pixels, axis=1))),
        'PnP': rows, 'negative_depth_ambiguity': 'A=-S is a proper rotation with reversed z; a projectively equivalent behind-camera solution exists. This is not the measured positive-depth solution above.'}


def audit():
    torch.set_num_threads(2)
    baseline = json.loads(REPORT.read_text()); config = json.loads((CHECKPOINT/'config.json').read_text())
    if config['encoder_args']['patch_embed_cls'] != 'PatchEmbedDust3R' or config['head_args']['landscape_only']:
        raise ValueError('This audit targets the observed public HF architecture')
    counts = {'all_portrait': 0, 'all_landscape': 0, 'mixed': 0}; portrait_views = swapped_K = 0
    for i, r in enumerate(baseline['requests']):
        p = json.loads((PREP_ROOT/f'request_{i:03d}.json').read_text())
        flags = [v['true_shape_hw'][0] > v['true_shape_hw'][1] for v in p['result']['returned_views']]
        counts['all_portrait' if all(flags) else 'all_landscape' if not any(flags) else 'mixed'] += 1
        for flag, v in zip(flags, p['result']['returned_views']):
            K = np.asarray(v['camera_intrinsics']); portrait_views += int(flag)
            swapped_K += int(np.linalg.det(K) < 0)
    paths = [__file__, base.__file__, 'fast3r/dust3r/patch_embed.py', 'fast3r/dust3r/heads/dpt_head.py',
        'fast3r/dust3r/utils/misc.py', 'fast3r/models/fast3r.py',
        'fast3r/models/multiview_dust3r_module.py', 'fast3r/dust3r/cloud_opt/init_im_poses.py']
    return {'status': 'published_HF_portrait_pixel_and_token_geometry_audited',
        'baseline_report_sha256': sha(REPORT), 'checkpoint_config_sha256': sha(CHECKPOINT/'config.json'),
        'source_sha256': {str(p): sha(p) for p in paths}, 'candidate_request_orientation_counts': counts,
        'portrait_view_count': portrait_views, 'negative_determinant_loaded_K_count': swapped_K,
        'public_HF_encoder_patch_class': config['encoder_args']['patch_embed_cls'],
        'public_HF_head_landscape_only': config['head_args']['landscape_only'],
        'token_probe': token_layout_probe(), 'geometry_probe': synthetic_geometry_probe(),
        'interpretation': 'Two distinct portrait interface mismatches: encoder/head token-grid ordering and transposed-pixel intrinsics versus positive-diagonal PnP K. Synthetic evidence is not proof that these fully explain measured model errors.',
        'GT_used_by_prediction': False, 'network_bytes': 0, 'trained_model_forward_count': 0,
        'formal_Table1_result': False, 'full_paper_completed': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--emit-proof-patch', action='store_true')
    args = p.parse_args(); result = audit()
    if args.emit_proof_patch:
        if OUTPUT.exists(): raise FileExistsError('Preserve historical audit')
        print('*** Begin Patch\n*** Add File: '+str(OUTPUT))
        for line in (json.dumps(result, indent=2, allow_nan=False)+'\n').splitlines(): print('+'+line)
        print('*** End Patch')
    else: print(json.dumps(result, indent=2, allow_nan=False))
