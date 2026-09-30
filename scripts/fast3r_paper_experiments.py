#!/usr/bin/env python3
"""Run seeded DTU head/view ablations using the released Fast3R weights.

Paper mapping: section 4.3/Table 4, section 5.1/Figure 5, section 5.4/Table 5.
Uniform-view sampling is a local extension; it is not the official stride-5 run.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

from fast3r.data.components.spann3r_datasets.dtu import DTU
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from fast3r_hf_dtu_eval import make_eval_views, make_model_views


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def summarize(rows):
    groups = {}
    for row in rows:
        if row['status'] != 'completed':
            continue
        for head, metrics in row['metrics'].items():
            key = f"{row['view_count']}_views_{head}"
            groups.setdefault(key, []).append(metrics)
    return {
        key: {'scene_count': len(values), 'aggregate_mean': {
            metric: float(np.mean([v[metric] for v in values]))
            for metric in values[0]
        }} for key, values in groups.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints/Fast3R_ViT_Large_512'))
    parser.add_argument('--data-root', type=Path, default=Path('data/dtu_test_mvsnet_release'))
    parser.add_argument('--output-json', type=Path, default=Path('results/dtu_paper_experiments.json'))
    parser.add_argument('--view-counts', type=int, nargs='+', default=[3, 5, 10, 20])
    parser.add_argument('--scenes', nargs='+')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--head-chunk-size', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if min(args.view_counts) < 2 or args.head_chunk_size < 1 or args.repeats < 1:
        parser.error('view counts >= 2, head chunk and repeats >= 1 required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run this command on the host outside the sandbox')
    if args.output_json.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output_json}; choose a new output path')
    seed_all(args.seed)
    dataset = DTU(split='test', ROOT=str(args.data_root), resolution=512,
                  num_seq=1, full_video=True, kf_every=1, seed=args.seed)
    dataset.scene_list = sorted(dataset.scene_list)
    names = args.scenes or dataset.scene_list
    if set(names) - set(dataset.scene_list):
        raise ValueError('Unknown scenes')
    print('Loading checkpoint', flush=True)
    model = Fast3R.from_pretrained(str(args.checkpoint_dir)).cuda().eval()
    model.set_max_parallel_views_for_head(args.head_chunk_size)
    evaluator = MultiViewDUSt3RLitModule.load_for_inference(model)
    report = {
        'status': 'running', 'seed': args.seed,
        'checkpoint_dir': str(args.checkpoint_dir),
        'config_sha256': hashlib.sha256((args.checkpoint_dir/'config.json').read_bytes()).hexdigest(),
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda, 'python': platform.python_version(),
        'precision': '16-mixed', 'resolution': 512, 'head_chunk_size': args.head_chunk_size,
        'sampling': 'Uniform indices over the loader-ordered full 49-view sequence; first anchor preserved. Different view counts also change the evaluated GT point union.',
        'alignment_confidence_percentile': 85, 'metric_confidence_percentile': 0,
        'paper_mapping': ['4.1/Table 2 (local timing adaptation)', '4.3/Table 4 (DTU extension)',
                          '5.1/Figure 5 (uniform-view adaptation)', '5.4/Table 5 (DTU paired head comparison)'],
        'scope_note': 'Released HF checkpoint. No retraining or new training checkpoints. Local hardware, sampling and DPT chunking differ from paper.',
        'scenes_expected': names, 'view_counts_expected': args.view_counts,
        'performance': [], 'rows': [], 'aggregate': {},
    }
    write_report(args.output_json, report)
    for name in names:
        raw = dataset[dataset.scene_list.index(name)]
        for count in args.view_counts:
            indices = np.linspace(0, len(raw)-1, count).round().astype(int).tolist()
            if len(set(indices)) != count:
                raise ValueError(f'Cannot choose {count} distinct views from {len(raw)}')
            selected = [raw[i] for i in indices]
            row = {'scene': name, 'view_count': count, 'input_labels': [v['label'] for v in selected],
                   'status': 'running', 'metrics': {}}
            result = None
            try:
                # Stable seed per scene/count, independent of loop order or warmups.
                trial_seed = args.seed + int(name.removeprefix('scan')) * 1000 + count
                seed_all(trial_seed)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                with torch.inference_mode():
                    result, profile = inference(make_model_views(selected), model, torch.device('cuda'),
                                                dtype='16-mixed', verbose=False, profiling=True)
                torch.cuda.synchronize()
                row.update(seed=trial_seed, inference_wall_seconds=time.perf_counter()-start,
                           forward_seconds=profile['total_time'],
                           peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                           peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                for pred in result['preds']:
                    for key in ['pts3d_in_other_view', 'pts3d_local', 'conf', 'conf_local']:
                        if key in pred and not torch.isfinite(pred[key]).all():
                            raise ValueError(f'Nonfinite output {name}/{count}/{key}')
                # Both modes use exactly the same forward predictions at 10 views.
                for head in (['local', 'global'] if count == 10 else ['local']):
                    evaluator.evaluate_reconstruction(make_eval_views(selected), result['preds'], 'dtu',
                        use_pts3d_from_local_head=head == 'local',
                        min_conf_thr_percentile_for_local_alignment_and_icp=85,
                        min_conf_thr_percentile_for_metric_cacluation=0)
                    values = evaluator.reconstruction_metrics_per_epoch.pop('dtu')[name]
                    if not values or not all(np.isfinite(v) for v in values.values()):
                        raise ValueError('Missing or nonfinite metrics')
                    row['metrics'][head] = {k: float(v) for k,v in values.items()}
                row['status'] = 'completed'
                # First scene: warm up once, then repeat synchronized inference.
                if name == names[0]:
                    del result
                    result = None
                    samples = []
                    for repeat in range(args.repeats + 1):
                        seed_all(trial_seed)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        start = time.perf_counter()
                        with torch.inference_mode():
                            timed_result, prof = inference(make_model_views(selected), model, torch.device('cuda'),
                                dtype='16-mixed', verbose=False, profiling=True)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter()-start
                        if repeat > 0:
                            samples.append({'forward_seconds': prof['total_time'],
                                'inference_wall_seconds': elapsed,
                                'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                                'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30})
                        del timed_result
                    report['performance'].append({'scene': name, 'view_count': count,
                        'warmup': 1, 'repeats': args.repeats, 'samples': samples})
            except torch.cuda.OutOfMemoryError as exc:
                row.update(status='oom', error=str(exc))
                torch.cuda.empty_cache()
            finally:
                del result
                gc.collect()
            report['rows'].append(row)
            report['aggregate'] = summarize(report['rows'])
            write_report(args.output_json, report)
            print(f"PROGRESS {name} views={count} {row['status']}", flush=True)
        del raw
    report['status'] = 'completed' if all(r['status']=='completed' for r in report['rows']) else 'completed_with_oom'
    write_report(args.output_json, report)
    print('REPORT', args.output_json, report['status'], flush=True)


if __name__ == '__main__':
    main()
