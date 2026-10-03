#!/usr/bin/env python3
"""Run a bounded DTU confidence-threshold sensitivity study on one forward pass.

This is a diagnostic for the local Hugging Face checkpoint and the retained
DTU test data. It is not used to select a best score or claim paper-level
reproduction. Every threshold case reuses the same input views, GT, and raw
predictions from one model forward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fast3r.data.components.spann3r_datasets.dtu import DTU
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from fast3r_hf_dtu_eval import make_eval_views, make_model_views


SCENE = "scan1"
RESOLUTION = 512
KF_EVERY = 1
MIN_FREE_GPU_MIB = 10_240
EXPECTED_LABELS = [
    "scan1/00000048.jpg",
    "scan1/00000043.jpg",
    "scan1/00000037.jpg",
    "scan1/00000032.jpg",
    "scan1/00000027.jpg",
    "scan1/00000021.jpg",
    "scan1/00000016.jpg",
    "scan1/00000011.jpg",
    "scan1/00000005.jpg",
    "scan1/00000000.jpg",
]
THRESHOLD_CASES = [
    {"name": "baseline", "alignment_percentile": 85.0, "metric_percentile": 0.0},
    {"name": "alignment_0", "alignment_percentile": 0.0, "metric_percentile": 0.0},
    {"name": "alignment_50", "alignment_percentile": 50.0, "metric_percentile": 0.0},
    {"name": "alignment_95", "alignment_percentile": 95.0, "metric_percentile": 0.0},
    {"name": "metric_25", "alignment_percentile": 85.0, "metric_percentile": 25.0},
    {"name": "metric_50", "alignment_percentile": 85.0, "metric_percentile": 50.0},
    {"name": "metric_75", "alignment_percentile": 85.0, "metric_percentile": 75.0},
]
RAW_PREDICTION_KEYS = ("pts3d_local", "conf_local", "pts3d_in_other_view", "conf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints/Fast3R_ViT_Large_512",
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "results/diagnostics/dtu_scan1_threshold_sensitivity_seed42_v1.json",
    )
    parser.add_argument("--head-chunk-size", type=int, default=2)
    return parser.parse_args()


def _tensor_bytes(value: Any) -> tuple[bytes, str, tuple[int, ...]]:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
    else:
        tensor = torch.as_tensor(np.ascontiguousarray(value)).contiguous()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    return raw, str(tensor.dtype), tuple(tensor.shape)


def tensor_fields_sha256(views: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for view_index, view in enumerate(views):
        for key in fields:
            if key not in view:
                continue
            raw, dtype, shape = _tensor_bytes(view[key])
            digest.update(f"{view_index}:{key}:{dtype}:{shape}".encode("utf-8"))
            digest.update(raw)
    return digest.hexdigest()


def prediction_sha256(preds: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for view_index, pred in enumerate(preds):
        for key in RAW_PREDICTION_KEYS:
            raw, dtype, shape = _tensor_bytes(pred[key])
            digest.update(f"{view_index}:{key}:{dtype}:{shape}".encode("utf-8"))
            digest.update(raw)
    return digest.hexdigest()


def checkpoint_manifest(checkpoint_dir: Path) -> list[dict[str, Any]]:
    files = sorted(path for path in checkpoint_dir.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No checkpoint files under {checkpoint_dir}")
    manifest: list[dict[str, Any]] = []
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        manifest.append({
            "path": str(path.relative_to(checkpoint_dir)),
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        })
    return manifest


def source_manifest() -> dict[str, str]:
    paths = (
        Path(__file__),
        REPO_ROOT / "scripts/fast3r_hf_dtu_eval.py",
        REPO_ROOT / "fast3r/models/multiview_dust3r_module.py",
        REPO_ROOT / "fast3r/data/components/spann3r_datasets/dtu.py",
    )
    return {
        str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def check_gpu_capacity(device: torch.device) -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_mib = int(free_bytes // (1024 * 1024))
    total_mib = int(total_bytes // (1024 * 1024))
    if free_mib < MIN_FREE_GPU_MIB:
        raise RuntimeError(
            f"Refusing to start inference: only {free_mib} MiB GPU memory is free; "
            f"this bounded run requires at least {MIN_FREE_GPU_MIB} MiB free."
        )
    return {"free_mib_before_model_load": free_mib, "total_mib": total_mib}


def write_report_new(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_json.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_json}")
    if args.head_chunk_size < 1:
        raise ValueError("head-chunk-size must be positive")
    if not args.checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint_dir}")
    data_path = args.data_root / "dtu_test_mvsnet_release"
    if not data_path.is_dir():
        raise FileNotFoundError(f"DTU test data not found: {data_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = DTU(
        split="test",
        ROOT=str(data_path),
        resolution=RESOLUTION,
        num_seq=1,
        full_video=True,
        kf_every=KF_EVERY,
        seed=args.seed,
    )
    dataset.scene_list = sorted(dataset.scene_list)
    if SCENE not in dataset.scene_list:
        raise ValueError(f"{SCENE} is missing from {data_path}")
    all_views = dataset[dataset.scene_list.index(SCENE)]
    selected_indices = np.linspace(0, len(all_views) - 1, 10).round().astype(int).tolist()
    raw_views = [all_views[index] for index in selected_indices]
    labels = [str(view["label"]) for view in raw_views]
    if labels != EXPECTED_LABELS:
        raise ValueError(
            f"Fixed input labels changed: expected {EXPECTED_LABELS}, got {labels}"
        )
    model_views = make_model_views(raw_views)
    eval_views = make_eval_views(raw_views)
    input_sha = tensor_fields_sha256(model_views, ("img", "true_shape"))
    ground_truth_sha = tensor_fields_sha256(eval_views, ("img", "pts3d", "valid_mask"))
    labels_sha = hashlib.sha256("\n".join(labels).encode("utf-8")).hexdigest()

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this execution context")
    gpu_capacity = check_gpu_capacity(device)
    checkpoint_files = checkpoint_manifest(args.checkpoint_dir)

    print(f"Loading Fast3R checkpoint: {args.checkpoint_dir}", flush=True)
    model = Fast3R.from_pretrained(str(args.checkpoint_dir)).to(device).eval()
    model.set_max_parallel_views_for_head(args.head_chunk_size)
    lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)
    scene_seed = args.seed + int(SCENE.removeprefix("scan")) * 1000 + 10
    torch.manual_seed(scene_seed)
    torch.cuda.manual_seed_all(scene_seed)
    torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    with torch.inference_mode():
        predictions, profile = inference(
            model_views,
            model,
            device,
            dtype="16-mixed",
            verbose=False,
            profiling=True,
        )
    forward_seconds = time.perf_counter() - start
    raw_prediction_sha_before = prediction_sha256(predictions["preds"])

    cases: list[dict[str, Any]] = []
    for case in THRESHOLD_CASES:
        print(f"Evaluating threshold case: {case['name']}", flush=True)
        lit_module.evaluate_reconstruction(
            eval_views,
            predictions["preds"],
            dataset_name="dtu",
            use_pts3d_from_local_head=True,
            min_conf_thr_percentile_for_local_alignment_and_icp=case["alignment_percentile"],
            min_conf_thr_percentile_for_metric_cacluation=case["metric_percentile"],
        )
        per_dataset = lit_module.reconstruction_metrics_per_epoch.pop("dtu", {})
        scene_metrics = per_dataset.get(SCENE)
        if not scene_metrics or not all(np.isfinite(value) for value in scene_metrics.values()):
            raise ValueError(f"Missing or nonfinite metrics for {SCENE}/{case['name']}")
        cases.append({
            **case,
            "head": "local",
            "metrics": {key: float(value) for key, value in scene_metrics.items()},
        })

    raw_prediction_sha_after = prediction_sha256(predictions["preds"])
    if raw_prediction_sha_before != raw_prediction_sha_after:
        raise RuntimeError("Raw prediction tensors changed during threshold evaluation")

    report = {
        "mode": "single_forward_threshold_sensitivity_diagnostic",
        "dataset": "dtu",
        "split": "test / dtu_test_mvsnet_release",
        "scene": SCENE,
        "seed": args.seed,
        "scene_seed": scene_seed,
        "resolution": RESOLUTION,
        "kf_every": KF_EVERY,
        "selection_protocol": "full stride1 sequence, rounded linspace 10 views; historical paper-experiment protocol",
        "selected_indices": selected_indices,
        "head_chunk_size": args.head_chunk_size,
        "precision": "16-mixed",
        "head": "local",
        "view_count": len(raw_views),
        "input_labels": labels,
        "input_labels_sha256": labels_sha,
        "model_inputs_sha256": input_sha,
        "ground_truth_img_pts_valid_mask_sha256": ground_truth_sha,
        "raw_prediction_sha256_before_evaluation": raw_prediction_sha_before,
        "raw_prediction_sha256_after_evaluation": raw_prediction_sha_after,
        "forward_count": 1,
        "same_forward_for_all_cases": True,
        "forward_wall_seconds": forward_seconds,
        "profile_total_seconds": profile.get("total_time") if isinstance(profile, dict) else None,
        "peak_allocated_mib": int(torch.cuda.max_memory_allocated(device) // (1024 * 1024)),
        "gpu_capacity_before_model_load": gpu_capacity,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "git_revision": git_revision(),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "checkpoint_files": checkpoint_files,
        "source_sha256": source_manifest(),
        "threshold_cases": cases,
        "scope_note": (
            "Single-scene sensitivity diagnostic on retained DTU test data using the public "
            "Hugging Face inference checkpoint and the repository metric implementation. "
            "Thresholds are compared on one unchanged forward and are not selected to fit "
            "the paper score; this does not replace the original Lightning checkpoint or "
            "the full benchmark evaluation."
        ),
    }
    write_report_new(args.output_json, report)
    print(f"Report written to {args.output_json}", flush=True)
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
