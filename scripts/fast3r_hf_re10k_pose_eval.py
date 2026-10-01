#!/usr/bin/env python3
"""Portable single-GPU RE10K pose evaluation, with strict split and resume.

Uses the public HF checkpoint and released geometry/metric implementations.
Seeded per-scene frame draws are a documented adaptation, not the unpublished
paper draws. PnP identity fallbacks are counted, never silently discarded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

import numpy as np
from PIL import Image

from check_pose_data import check_re10k, read_re10k_metadata, read_split

METRICS = ['RRA_at_5', 'RRA_at_15', 'RRA_at_30',
           'RTA_at_5', 'RTA_at_15', 'RTA_at_30', 'mAA_30']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def scene_seed(seed, name):
    return (seed + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')) % (2**32)


def frame_selection(root, name, num_views, seed):
    # Match upstream lexical frame ordering; fixed per-scene PRNG makes resume stable.
    frames = sorted((Path(root) / name).glob('*.jpg'))
    if len(frames) < num_views:
        raise ValueError(f'Need {num_views} distinct frames for {name}')
    return sorted(random.Random(scene_seed(seed, name)).sample(frames, num_views))


def crop_resize(image, intrinsics, target=(512, 288)):
    """Same principal-point crop/resize sequence as the upstream RE10K script."""
    from fast3r.dust3r.datasets.utils import cropping
    width, height = image.size
    cx, cy = (int(round(x)) for x in intrinsics[:2, 2])
    mx, my = min(cx, width-cx), min(cy, height-cy)
    if min(mx, my) <= 0:
        raise ValueError('Principal point lies outside RGB image')
    image, _, intrinsics = cropping.crop_image_depthmap(
        image, None, intrinsics, (cx-mx, cy-my, cx+mx, cy+my))
    if image.height > image.width:
        target = target[::-1]
    image, _, intrinsics = cropping.rescale_image_depthmap(image, None, intrinsics, np.array(target))
    camera = cropping.camera_matrix_of_crop(intrinsics, image.size, target, offset_factor=.5)
    bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, camera, target)
    image, _, intrinsics = cropping.crop_image_depthmap(image, None, intrinsics, bbox)
    return image, intrinsics


def load_views(paths, metadata):
    import torch
    from fast3r.dust3r.datasets.utils.transforms import ImgNorm
    model_views, gt, cameras = [], [], []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert('RGB')
            width, height = image.size
            record = metadata[path.stem]
            fx, fy, cx, cy = record['intrinsics_normalized']
            camera = np.array([[fx*width, 0, cx*width], [0, fy*height, cy*height],
                               [0, 0, 1]], dtype=np.float32)
            image, camera = crop_resize(image, camera)
            model_views.append({'img': ImgNorm(image).unsqueeze(0),
                                'true_shape': torch.tensor([[image.height, image.width]])})
        gt.append(record['camera_pose'])
        cameras.append(camera.tolist())
    # GT is kept OUTSIDE model inputs and PnP focal estimation.
    return model_views, np.asarray(gt, dtype=np.float32), cameras


def predict_poses(preds, seed, niter=100):
    import cv2
    import torch
    from fast3r.dust3r.cloud_opt.init_im_poses import fast_pnp
    from fast3r.models.multiview_dust3r_module import estimate_focal
    focal = estimate_focal(preds[0]['pts3d_in_other_view'].float(),
                           preds[0]['conf'].float(), min_conf_thr_percentile=10)
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError('Invalid estimated focal length')
    poses, failures = [], []
    for i, pred in enumerate(preds):
        points = pred['pts3d_in_other_view'][0].float().cpu()
        confidence = pred['conf'][0].float().cpu()
        if not torch.isfinite(points).all() or not torch.isfinite(confidence).all():
            raise ValueError('Nonfinite prediction')
        # Actual upstream mask is conf > 1; percentile argument there is ignored.
        mask = confidence > 1.0
        cv2.setRNGSeed(int((seed+i) % (2**31-1)))
        found_focal, pose = fast_pnp(points, focal, mask, 'cpu', niter_PnP=niter)
        if pose is None or found_focal is None:
            failures.append(i)
            poses.append(np.eye(4, dtype=np.float32))  # explicit upstream fallback
        else:
            pose = pose.cpu().numpy()
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError('Invalid PnP camera pose')
            poses.append(pose)
    return np.asarray(poses), focal, failures


def pose_metrics(predicted, gt):
    import torch
    from fast3r.eval.cam_pose_metric import camera_to_rel_deg, calculate_auc
    predicted, gt = [torch.tensor(x, dtype=torch.float32) for x in (predicted, gt)]
    r, t = camera_to_rel_deg(predicted, gt, 'cpu', len(gt))
    if not torch.isfinite(r).all() or not torch.isfinite(t).all():
        raise ValueError('Nonfinite relative pose error')
    metrics = {f'RRA_at_{threshold}': (r < threshold).float().mean().item()
               for threshold in (5, 15, 30)}
    metrics.update({f'RTA_at_{threshold}': (t < threshold).float().mean().item()
                    for threshold in (5, 15, 30)})
    metrics['mAA_30'] = calculate_auc(r, t, max_threshold=30).item()
    return metrics, {'rotation_deg': r.tolist(), 'translation_deg': t.tolist()}


def validate_row(row, name, signature, inputs):
    if row.get('scene') != name or row.get('protocol_sha256') != signature or row.get('inputs') != inputs:
        raise ValueError(f'Resume input/protocol changed: {name}')
    metrics = row.get('metrics', {})
    if set(metrics) != set(METRICS) or not all(np.isfinite(x) and 0 <= x <= 1 for x in metrics.values()):
        raise ValueError(f'Invalid resumed metrics: {name}')
    count = len(inputs['frames'])
    if np.asarray(row.get('predicted_c2w')).shape != (count, 4, 4):
        raise ValueError(f'Invalid resumed poses: {name}')
    if not np.isfinite(np.asarray(row['predicted_c2w'])).all():
        raise ValueError(f'Nonfinite resumed poses: {name}')
    if row.get('pair_count') != count*(count-1)//2:
        raise ValueError(f'Invalid resumed pair count: {name}')
    failures = row.get('pnp_failed_view_indices', [])
    if len(failures) != len(set(failures)) or any(not isinstance(i, int) or not 0 <= i < count for i in failures):
        raise ValueError(f'Invalid PnP failure indices: {name}')
    recalculated, errors = pose_metrics(row['predicted_c2w'], row['gt_c2w'])
    if any(abs(metrics[key]-recalculated[key]) > 1e-7 for key in METRICS):
        raise ValueError(f'Resumed metrics disagree with saved poses: {name}')
    if any(not np.allclose(errors[key], row['relative_errors'][key], atol=1e-7, rtol=0)
           for key in errors):
        raise ValueError(f'Resumed pairwise errors disagree: {name}')


def aggregate(rows, names):
    if len(rows) != len(names) or len({row['scene'] for row in rows}) != len(names) or set(
            row['scene'] for row in rows) != set(names):
        raise ValueError('Report does not cover the complete prescribed split')
    return {key: statistics.mean(row['metrics'][key] for row in rows) for key in METRICS}


def save_new(path, value):
    # Exclusive creation prevents replacing history. A torn row is rejected on resume.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def run(args):
    names = read_split(args.split_file)
    if args.benchmark_scope == 'prescribed_full' and sha(args.split_file) != (
            '6d55e2006523c2388f40f3d6a2a6bb2ef062790edb5dde9a10c1bf90f18c4796'):
        raise ValueError('Full benchmark requires the exact prescribed 1832-ID split; use adaptation_smoke for tests')
    preflight = check_re10k(args.data_root, args.metadata_root, args.split_file,
                            num_views=args.num_views, decode=True)
    if preflight['status'] != 'ready':
        raise ValueError(f"Full split is incomplete: {len(preflight['issues'])} scenes; "
                         f"first issues: {preflight['issues'][:3]}")
    if args.dry_run:
        paths = frame_selection(args.data_root, names[0], args.num_views, args.seed)
        views, gt, _ = load_views(paths, read_re10k_metadata(args.metadata_root/(names[0]+'.txt')))
        print(json.dumps({'mode': 'dry_run', 'expected_scene_count': len(names),
                          'first_scene': names[0], 'timestamps': [p.stem for p in paths],
                          'image_shapes': [list(v['img'].shape) for v in views],
                          'gt_shape': list(gt.shape), 'model_loaded': False}))
        return
    import torch
    from fast3r.models.fast3r import Fast3R
    from fast3r.dust3r.inference_multiview import inference
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use host execution')
    if args.output_json.exists():
        raise FileExistsError('Final report exists; verify it, do not overwrite history')
    protocol = {'dataset': 'RealEstate10K', 'benchmark_scope': args.benchmark_scope,
                'split_sha256': sha(args.split_file),
                'expected_scene_count': len(names), 'num_views': args.num_views, 'seed': args.seed,
                'sampling': 'upstream lexical frame ordering, per-scene seeded random.sample',
                'resolution': [512, 288], 'precision': '16-mixed',
                'head_chunk_size': args.head_chunk_size, 'niter_PnP': 100,
                'focal': 'first_view_from_global_head; confidence percentile10',
                'pnp_mask': 'conf > 1.0 (actual upstream implementation)',
                'pnp_execution': 'sequential with per-view OpenCV RNG seed; upstream uses threads',
                'pnp_failure': 'retain identity fallback and count failed views',
                'metric': 'released torch camera_to_rel_deg + calculate_auc; equal scene mean',
                'checkpoint_weight_sha256': sha(args.checkpoint_dir/'model.safetensors'),
                'checkpoint_config_sha256': sha(args.checkpoint_dir/'config.json'),
                'runner_sha256': sha(__file__),
                'metric_sha256': sha('fast3r/eval/cam_pose_metric.py'),
                'geometry_sha256': sha('fast3r/models/multiview_dust3r_module.py'),
                'paper_equivalence': 'not established: public checkpoint mapping/frame draws differ'}
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    run_file = args.progress_dir/'run.json'
    if run_file.exists():
        if json.loads(run_file.read_text()) != protocol:
            raise ValueError('Resume protocol changed; choose a new progress directory')
    else:
        save_new(run_file, protocol)
    rows, model = [], None
    for name in names:
        seed = scene_seed(args.seed, name)
        paths = frame_selection(args.data_root, name, args.num_views, args.seed)
        inputs = {'metadata_sha256': sha(args.metadata_root/(name+'.txt')),
                  'frames': [{'timestamp': p.stem, 'sha256': sha(p)} for p in paths]}
        row_path = args.progress_dir/(name+'.json')
        if row_path.exists():
            row = json.loads(row_path.read_text())
            validate_row(row, name, signature, inputs)
        else:
            if model is None:
                model = Fast3R.from_pretrained(str(args.checkpoint_dir)).cuda().eval()
                model.set_max_parallel_views_for_head(args.head_chunk_size)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            views, gt, cameras = load_views(paths, read_re10k_metadata(args.metadata_root/(name+'.txt')))
            with torch.inference_mode():
                output = inference(views, model, torch.device('cuda'), dtype='16-mixed', verbose=False)
                poses, focal, failures = predict_poses(output['preds'], seed)
                metrics, errors = pose_metrics(poses, gt)
            row = {'scene': name, 'seed': seed, 'protocol_sha256': signature, 'inputs': inputs,
                   'metrics': metrics, 'relative_errors': errors, 'predicted_c2w': poses.tolist(),
                   'gt_c2w': gt.tolist(), 'cropped_gt_intrinsics': cameras,
                   'estimated_focal': focal, 'pnp_failed_view_indices': failures,
                   'pair_count': args.num_views*(args.num_views-1)//2}
            validate_row(row, name, signature, inputs)
            save_new(row_path, row)
            print(json.dumps({'scene': name, 'completed': len(rows)+1, 'expected': len(names),
                              'pnp_failures': failures, 'metrics': metrics}), flush=True)
            del output, views
        rows.append(row)
    means = aggregate(rows, names)
    failed_views = sum(len(row['pnp_failed_view_indices']) for row in rows)
    save_new(args.output_json, {'status': 'complete', 'protocol': protocol,
                                'protocol_sha256': signature, 'scene_count': len(rows),
                                'pnp_failed_view_count': failed_views,
                                'aggregate_mean': means, 'scenes': rows,
                                'paper_result_matched': False})
    print('POSE EVALUATION COMPLETE: full supplied split; paper equivalence not claimed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data/RealEstate10K/videos/test'))
    parser.add_argument('--metadata-root', type=Path, default=Path('data/RealEstate10K/test'))
    parser.add_argument('--split-file', type=Path, default=Path('scripts/re10k_test_1800.txt'))
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints/Fast3R_ViT_Large_512'))
    parser.add_argument('--progress-dir', type=Path, default=Path('results/re10k_pose_seed42_progress'))
    parser.add_argument('--output-json', type=Path, default=Path('results/re10k_pose_seed42.json'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-views', type=int, default=10)
    parser.add_argument('--head-chunk-size', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--benchmark-scope', choices=['prescribed_full', 'adaptation_smoke'],
                        default='prescribed_full', help='Smoke subsets are never a full Table1 benchmark')
    args = parser.parse_args()
    if args.head_chunk_size < 1 or args.num_views < 2:
        parser.error('Chunk size must be >=1 and views >=2')
    if args.dry_run:
        run(args)
    else:
        import fcntl
        args.progress_dir.mkdir(parents=True, exist_ok=True)
        with (args.progress_dir/'.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args)


if __name__ == '__main__':
    main()
