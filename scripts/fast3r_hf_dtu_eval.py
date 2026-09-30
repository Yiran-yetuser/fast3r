#!/usr/bin/env python3
"""Evaluate local Hugging Face Fast3R weights on prepared reconstruction data.

The official ``fast3r/eval.py`` entry point expects a Lightning ``last.ckpt``
directory.  The released Hugging Face model is an inference checkpoint instead,
so this script keeps the weights in their native format and reuses the
repository's official reconstruction metric implementation directly.

This is useful for reproducing the reconstruction numbers when only the public
Hugging Face checkpoint is available.  It intentionally does not claim to
replace the paper's original training checkpoint or every benchmark split.
"""

from __future__ import annotations

import argparse
import json
import random
import platform
import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fast3r.data.components.spann3r_datasets.dtu import DTU
from fast3r.data.components.spann3r_datasets.nrgbd import NRGBD
from fast3r.data.components.spann3r_datasets.seven_scenes import SevenScenes
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("dtu", "nrgbd", "7scenes"), default="dtu")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for Python, NumPy, dataset and randomized image-index embeddings.",
    )
    parser.add_argument(
        "--head", choices=("local", "global"), default="local",
        help="Aligned local pointmap or raw global pointmap (paper section 5.4).",
    )
    parser.add_argument(
        "--head-chunk-size", type=int, default=2,
        help="Max views processed together by DPT heads (lower reduces VRAM).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/Fast3R_ViT_Large_512"),
        help="Local Hugging Face checkpoint directory.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Directory containing dtu_test_mvsnet_release/.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("demo_outputs/paper_eval/dtu_hf_metrics.json"),
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="Inference device. CPU is intended for dry-run/debug only.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Optional scene names, for example scan1 scan12.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="Evaluate at most this many scenes; 0 means all selected scenes.",
    )
    parser.add_argument(
        "--kf-every",
        type=int,
        default=None,
        help="Temporal stride used by the official DTU configuration.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Long-side image resolution used by the official DTU configuration.",
    )
    parser.add_argument(
        "--alignment-confidence-percentile",
        type=float,
        default=85.0,
        help="Confidence percentile used for local-to-global alignment and ICP.",
    )
    parser.add_argument(
        "--metric-confidence-percentile",
        type=float,
        default=0.0,
        help="Confidence percentile retained for metric calculation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect DTU scenes and sample tensors without loading the model.",
    )
    return parser.parse_args()


def _as_batched_tensor(value: Any) -> Any:
    """Add the batch dimension expected by Fast3R/Lightning evaluation."""

    if torch.is_tensor(value):
        return value.unsqueeze(0)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).unsqueeze(0)
    return value


def make_model_views(raw_views: list[dict[str, Any]]) -> list[dict[str, torch.Tensor]]:
    """Keep only image inputs needed by the inference function."""

    return [
        {
            "img": _as_batched_tensor(view["img"]),
            "true_shape": _as_batched_tensor(view["true_shape"]),
        }
        for view in raw_views
    ]


def make_eval_views(raw_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Batch GT views while preserving labels and dataset metadata."""

    batched_views: list[dict[str, Any]] = []
    for view in raw_views:
        batched = dict(view)
        for key, value in list(batched.items()):
            if key in {"dataset", "label", "instance"} and isinstance(value, str):
                batched[key] = [value]
            elif key not in {"idx", "rng"}:
                batched[key] = _as_batched_tensor(value)
        batched_views.append(batched)
    return batched_views


def select_scene_indices(dataset: DTU, scenes: list[str] | None, max_scenes: int) -> list[int]:
    if scenes is None:
        indices = list(range(len(dataset.scene_list)))
    else:
        missing = sorted(set(scenes) - set(dataset.scene_list))
        if missing:
            raise ValueError(f"Unknown DTU scenes: {missing}; available examples: {dataset.scene_list[:5]}")
        indices = [dataset.scene_list.index(scene) for scene in scenes]
    if max_scenes > 0:
        indices = indices[:max_scenes]
    if not indices:
        raise ValueError("No DTU scenes selected")
    return indices


def run_dry_run(dataset: DTU, scene_indices: list[int]) -> None:
    print(f"{type(dataset).__name__} scenes available: {len(dataset.scene_list)}")
    print(f"Scenes selected: {[dataset.scene_list[index] for index in scene_indices]}")
    for scene_index in scene_indices:
        views = dataset[scene_index]
        valid_ratio = float(np.mean([view["valid_mask"].mean() for view in views]))
        print(
            f"{dataset.scene_list[scene_index]}: views={len(views)}, "
            f"image_shape={tuple(views[0]['img'].shape)}, "
            f"valid_depth_ratio={valid_ratio:.3f}"
        )


def load_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    specs = {"dtu": (DTU, "dtu_test_mvsnet_release", 5),
             "nrgbd": (NRGBD, "neural_rgbd", 40),
             "7scenes": (SevenScenes, "7_scenes_processed", 20)}
    dataset_cls, subdir, default_stride = specs[args.dataset]
    if args.kf_every is None:
        args.kf_every = default_stride
    if args.head_chunk_size < 1 or args.kf_every < 1:
        raise ValueError("Head chunk size and kf-every must be positive")
    dtu_root = args.data_root / subdir
    if not dtu_root.is_dir():
        raise FileNotFoundError(f"{args.dataset} directory not found: {dtu_root}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Use --dry-run for data checks "
            "or run on a machine with the required GPU."
        )

    dataset = dataset_cls(
        split="test",
        ROOT=str(dtu_root),
        resolution=args.resolution,
        num_seq=1,
        full_video=True,
        kf_every=args.kf_every,
        seed=args.seed,
    )
    dataset.scene_list = sorted(dataset.scene_list)
    if args.dataset == "nrgbd":
        dataset.scene_list = [name for name in dataset.scene_list
                              if (dtu_root/name/"poses.txt").is_file()
                              and (dtu_root/name/"images").is_dir()
                              and (dtu_root/name/"depth").is_dir()]
        if not dataset.scene_list:
            raise ValueError("No prepared Neural RGB-D sequences found")
    scene_indices = select_scene_indices(dataset, args.scenes, args.max_scenes)
    if args.dry_run:
        run_dry_run(dataset, scene_indices)
        return {
            "mode": "dry_run",
            "data_root": str(dtu_root),
            "scenes": [dataset.scene_list[index] for index in scene_indices],
        }

    if not args.checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Hugging Face checkpoint directory not found: {args.checkpoint_dir}")

    print(f"Loading Fast3R checkpoint from {args.checkpoint_dir}")
    model = Fast3R.from_pretrained(str(args.checkpoint_dir)).to(device).eval()
    model.set_max_parallel_views_for_head(args.head_chunk_size)
    lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)
    precision = "16-mixed" if device.type == "cuda" else "32"
    profiling = device.type == "cuda"

    scene_reports: list[dict[str, Any]] = []
    for scene_index in scene_indices:
        scene_name = dataset.scene_list[scene_index]
        raw_views = dataset[scene_index]
        model_views = make_model_views(raw_views)
        eval_views = make_eval_views(raw_views)

        # Repeatable irrespective of scene selection/order or model initialization.
        scene_seed = args.seed + int.from_bytes(hashlib.sha256(scene_name.encode()).digest()[:4], 'little')
        if args.dataset == 'dtu':
            scene_seed = args.seed + int(scene_name.removeprefix('scan'))
        torch.manual_seed(scene_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(scene_seed)
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        with torch.inference_mode():
            result = inference(
                model_views,
                model,
                device,
                dtype=precision,
                verbose=False,
                profiling=profiling,
            )
        predictions, profile = result if profiling else (result, None)

        lit_module.evaluate_reconstruction(
            eval_views,
            predictions["preds"],
            dataset_name=args.dataset,
            use_pts3d_from_local_head=args.head == "local",
            min_conf_thr_percentile_for_local_alignment_and_icp=args.alignment_confidence_percentile,
            min_conf_thr_percentile_for_metric_cacluation=args.metric_confidence_percentile,
        )
        metrics_by_scene = lit_module.reconstruction_metrics_per_epoch.pop(args.dataset, {})
        metrics = metrics_by_scene.get(scene_name, {})
        if not metrics or not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError(f"Missing or nonfinite metrics for {scene_name}")
        report = {
            "scene": scene_name,
            "views": len(raw_views),
            "seed": scene_seed,
            "wall_time_seconds": time.perf_counter() - start,
            "forward_time_seconds": profile.get("total_time") if profile else None,
            "metrics": {key: float(value) for key, value in metrics.items()},
        }
        scene_reports.append(report)
        print(json.dumps(report, ensure_ascii=False))

    metric_names = sorted({key for report in scene_reports for key in report["metrics"]})
    aggregate = {
        key: float(np.mean([report["metrics"][key] for report in scene_reports if key in report["metrics"]]))
        for key in metric_names
    }
    return {
        "mode": "hf_checkpoint_official_metrics",
        "dataset": args.dataset,
        "checkpoint_dir": str(args.checkpoint_dir),
        "data_root": str(dtu_root),
        "device": str(device),
        "precision": precision,
        "seed": args.seed,
        "head": args.head,
        "head_chunk_size": args.head_chunk_size,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "resolution": args.resolution,
        "kf_every": args.kf_every,
        "alignment_confidence_percentile": args.alignment_confidence_percentile,
        "metric_confidence_percentile": args.metric_confidence_percentile,
        "scene_count": len(scene_reports),
        "aggregate_mean": aggregate,
        "paper_distance_multiplier": 1 if args.dataset == "dtu" else 100,
        "scenes": scene_reports,
        "scope_note": (
            "Metrics reuse fast3r.models.multiview_dust3r_module.MultiViewDUSt3RLitModule "
            "and are evaluated with the public Hugging Face inference checkpoint; "
            "this is distinct from running fast3r/eval.py with the original Lightning checkpoint."
        ),
    }


def main() -> None:
    args = parse_args()
    report = evaluate(args)
    load_report(args.output_json, report)
    print(f"Report written to {args.output_json}")


if __name__ == "__main__":
    main()
