#!/usr/bin/env python3
"""ONE archived CO3D mapped draw with public HF weights, NEVER full Table1.

Read-only prepared data; no network. Only RGB tensors + true_shape go into the
network/orientation correction. GT K/depth/poses stay outside focal/PnP. Keep
duplicates, all 45 pairs and explicit identity fallback for failed PnP.
"""
import argparse
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from co3d_lazy_dataset import StrictLazyCo3d
from fast3r_hf_re10k_pose_eval import predict_poses, pose_metrics, validate_row, save_new
from prepare_re10k_rgb_from_archive import sha, save_identical

DATA_REPORT = Path('results/co3d_lazy_draw0_v2_20261002.json')
REFERENCE_REPORT = Path('results/co3d_lazy_draw0_verified_20261002.json')
PREFLIGHT = Path('results/co3d_pose_draw0_input_preflight_v2_20261002.json')


class ReadOnlyPrepared:
    def __init__(self, report):
        self.root = Path('data/co3d_lazy_processed')
        self.rows = {r['image_path']: r for r in report['prepared_frames']}

    def ensure(self, category, scene, frame):
        row = self.rows[f'{category}/{scene}/images/frame{frame:06d}.jpg']
        for name, expected in row['processed_sha256'].items():
            if sha(self.root/name) != expected: raise ValueError('Prepared byte identity changed')
        return row  # no missing-frame download, no sparse-pool substitution


def model_inputs(views):
    return [{'img': v['img'].unsqueeze(0),
             'true_shape': torch.from_numpy(v['true_shape'].copy()).unsqueeze(0)} for v in views]


def correct_orientation(preds, inputs):
    """Use the actual released correction once, restore B=1 tensors for runner."""
    from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
    corrected = [{k:p[k] for k in ('pts3d_in_other_view','conf')} for p in preds]
    MultiViewDUSt3RLitModule.correct_preds_orientation(corrected, inputs)
    for pred, view in zip(corrected,inputs):
        # Released evaluator corrects portrait head outputs back to loader landscape,
        # NOT to original true_shape. Public HF landscape_only=False emits portrait.
        H,W = [int(x) for x in view['img'].shape[-2:]]
        for key in ('pts3d_in_other_view','conf'): pred[key] = torch.stack(pred[key])
        if pred['conf'].shape != (1,H,W) or pred['pts3d_in_other_view'].shape != (1,H,W,3):
            raise ValueError('Wrong corrected prediction orientation')
    return corrected


def load_archived_draw():
    report = json.loads(DATA_REPORT.read_text()); ref = json.loads(REFERENCE_REPORT.read_text())
    if ref['probe_report_sha256'] != sha(DATA_REPORT) or not report['original_loader_base_outputs_exactly_equal']:
        raise ValueError('Archived draw verification changed')
    f = report['fingerprint']
    expected_files = {f['preparer']['lazy_code_sha256']:'scripts/co3d_lazy_dataset.py',
        f['released_loader_sha256']:'fast3r/dust3r/datasets/co3d_multiview.py',
        f['base_loader_sha256']:'fast3r/dust3r/datasets/base/base_stereo_view_dataset.py',
        f['wrapper_sha256']:'fast3r/dust3r/datasets/base/easy_dataset.py',
        ref['verifier_sha256']:'scripts/verify_co3d_lazy_sample.py'}
    for h,p in expected_files.items():
        if sha(p) != h: raise ValueError(f'Archived code identity changed: {p}')
    candidate = Path('data/co3d_test_metadata/selected_seqs_test_seen41_candidate.json')
    if sha(candidate) != f['preparer']['candidate_sha256']: raise ValueError('Candidate changed')
    for row in report['prepared_frames']:
        for member in row['raw_members']:
            raw = Path('data/co3d_range_cache/raw')/member['path']
            if raw.stat().st_size != member['bytes'] or sha(raw) != member['sha256']:
                raise ValueError('Raw saved bytes changed')
    s = report['scope']; selected = json.loads(candidate.read_text())
    dataset = StrictLazyCo3d(selected,ReadOnlyPrepared(report),s['combination_seed'],s['dataset_seed'])
    random.seed(s['python_request_seed']); views = dataset[s['base_index']]
    if len(views) != 10 or len(report['returned_views']) != 10: raise ValueError('Invalid view count')
    for view,row in zip(views,report['returned_views']):
        if (view['label'] != row['label'] or view['instance'] != row['instance']
                or list(view['img'].shape) != row['img_shape'] or view['rng'] != row['rng']
                or view['true_shape'].tolist() != row['true_shape_hw']
                or not np.array_equal(view['camera_pose'],np.asarray(row['camera_pose'],np.float32))
                or not np.array_equal(view['camera_intrinsics'],np.asarray(row['camera_intrinsics'],np.float32))):
            raise ValueError('Realized archived inputs changed')
    if [t['frame_number'] for t in dataset.trace] != [t['frame_number'] for t in report['actual_load_trace']]:
        raise ValueError('Actual sampling trace changed')
    return report,views


def protocol(checkpoint,seed,chunk):
    audit=json.loads(Path('results/protocol_audit_20261001.json').read_text())['checkpoint']
    weights=sha(checkpoint/'model.safetensors'); config=sha(checkpoint/'config.json')
    if weights != audit['local_weight_sha256'] or config != audit['local_config_sha256']:
        raise ValueError('Audited public checkpoint changed')
    return {'dataset':'CO3D','benchmark_scope':'one_draw_adaptation_smoke_not_Table1',
        'draw_report_sha256':sha(DATA_REPORT),'reference_report_sha256':sha(REFERENCE_REPORT),
        'runner_sha256':sha(Path(__file__)),'pose_helper_sha256':sha(Path('scripts/fast3r_hf_re10k_pose_eval.py')),
        'module_sha256':sha(Path('fast3r/models/multiview_dust3r_module.py')),
        'metric_sha256':sha(Path('fast3r/eval/cam_pose_metric.py')),
        'so3_sha256':sha(Path('fast3r/utils/so3_utils.py')),
        'pnp_sha256':sha(Path('fast3r/dust3r/cloud_opt/init_im_poses.py')),
        'model_sha256':sha(Path('fast3r/models/fast3r.py')),
        'inference_sha256':sha(Path('fast3r/dust3r/inference_multiview.py')),
        'checkpoint_weight_sha256':weights,'checkpoint_config_sha256':config,
        'head_chunk_size':chunk,'precision':'16-mixed','inference_seed':seed,
        'orientation':'HF landscape_only=False emits true_shape; released correction once to loader img shape',
        'network_input_keys':['img','true_shape'],'gt_used_by_network_or_PnP':False,
        'focal':'first_view global-head + confidence percentile10; no GT intrinsics',
        'pnp':'released fast_pnp, conf>1,100 iterations; sequential per-view OpenCV seeds',
        'fallback':'identity retained and counted','metric_device':'cpu',
        'pair_count':45,'paper_checkpoint_and_draw_mapping_verified':False}


def signature(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def validate_smoke(row, inputs, proto, gt):
    name=inputs['frames'][0]['label']
    validate_row(row,name,signature(proto),inputs)
    if row['protocol'] != proto or not np.array_equal(np.asarray(row['gt_c2w'],np.float32),gt):
        raise ValueError('Saved protocol or GT changed')
    if row['formal_pose_metrics_available'] or row['all_100_draws_executed']:
        raise ValueError('Smoke misrepresented as full benchmark')
    if row['model_inference_completed'] is not True: raise ValueError('No completed model inference')


def run(args):
    torch.set_num_threads(2)
    report,views=load_archived_draw(); inputs={'frames':report['returned_views']}
    seed=args.seed+int(report['scope']['base_index'])
    proto=protocol(args.checkpoint_dir,seed,args.head_chunk_size)
    gt=np.stack([v['camera_pose'] for v in views]).astype(np.float32)
    model_views=model_inputs(views)
    if args.verify_only:
        validate_smoke(json.loads(args.output_json.read_text()),inputs,proto,gt)
        print('SAVED CO3D ONE-DRAW POSE SMOKE VERIFIED; no inference rerun',flush=True);return
    if args.dry_run:
        preflight={'status':'one_draw_input_and_public_checkpoint_verified_no_inference',
            'protocol':proto,'protocol_sha256':signature(proto),
            'frame_names':[[v['label'],v['instance']] for v in views],
            'input_tensor_shapes':[list(v['img'].shape) for v in model_views],
            'original_true_shapes':[v['true_shape'].tolist() for v in model_views],
            'gt_shape':list(gt.shape),'model_loaded':False,'network_bytes':0,
            'gt_used_by_network_or_PnP':False,'formal_pose_metrics_available':False}
        save_identical(PREFLIGHT,preflight);print(json.dumps(preflight,indent=2),flush=True);return
    if args.output_json.exists(): raise FileExistsError('Existing output; verify, never overwrite')
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    if free < 10240: raise RuntimeError('GPU free memory below10240MiB; queue instead of competing')
    if shutil.disk_usage(Path('results')).free < 1024**3+1024**2: raise RuntimeError('Preserve1GiB reserve')
    from fast3r.models.fast3r import Fast3R
    from fast3r.dust3r.inference_multiview import inference
    if not torch.cuda.is_available():raise RuntimeError('Host CUDA unavailable')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=Fast3R.from_pretrained(str(args.checkpoint_dir)).cuda().eval()
    model.set_max_parallel_views_for_head(args.head_chunk_size)
    # Initialization must not advance the recorded inference PRNG state.
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    with torch.inference_mode():
        output=inference(model_views,model,torch.device('cuda'),dtype='16-mixed',verbose=False)
        torch.cuda.synchronize();forward_seconds=time.perf_counter()-start
        raw_prediction_shapes=[list(p['conf'].shape) for p in output['preds']]
        corrected=correct_orientation(output['preds'],model_views)
        poses,focal,failures=predict_poses(corrected,seed,niter=100)
        metrics,errors=pose_metrics(poses,gt)
        self_metrics,self_errors=pose_metrics(gt,gt)
    repeated=[list(pair) for pair in __import__('itertools').combinations(range(len(views)),2)
              if views[pair[0]]['instance']==views[pair[1]]['instance'] and views[pair[0]]['label']==views[pair[1]]['label']]
    row={'status':'one_draw_public_HF_pose_smoke_complete_not_Table1','scene':views[0]['label'],
        'protocol':proto,'protocol_sha256':signature(proto),'inputs':inputs,'seed':seed,
        'metrics':metrics,'relative_errors':errors,'predicted_c2w':poses.tolist(),'gt_c2w':gt.tolist(),
        'pair_count':45,'pnp_failed_view_indices':failures,'estimated_focal':float(focal),
        'raw_prediction_shapes':raw_prediction_shapes,
        'corrected_prediction_shapes':[list(p['conf'].shape) for p in corrected],
        'repeated_camera_view_pairs':repeated,'gt_self_metric_diagnostic':{'metrics':self_metrics,'errors':self_errors},
        'model_inference_completed':True,'formal_pose_metrics_available':False,'all_100_draws_executed':False,
        'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
            'forward_seconds':forward_seconds,'cuda_max_memory_allocated_bytes':torch.cuda.max_memory_allocated()},
        'note':'One mapped sample only, not paper aggregate; duplicate zero-baseline pairs retained. GT self metric is numerical diagnostic, not prediction performance. Shared-state continuous100@ not executed.'}
    validate_smoke(row,inputs,proto,gt)
    # Pre-serialize before exclusive save to prevent another NumPy torn report.
    json.dumps(row,allow_nan=False);save_new(args.output_json,row)
    print('CO3D ONE-DRAW POSE SMOKE COMPLETE',json.dumps({'metrics':metrics,'PnP_failures':failures}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir',type=Path,default=Path('checkpoints/Fast3R_ViT_Large_512'))
    p.add_argument('--output-json',type=Path,default=Path('results/co3d_pose_draw0_seed42_20261002.json'))
    p.add_argument('--seed',type=int,default=42);p.add_argument('--head-chunk-size',type=int,default=2)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--verify-only',action='store_true')
    run(p.parse_args())
