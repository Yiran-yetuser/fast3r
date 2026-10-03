#!/usr/bin/env python3
"""Audit a historical engineering proposal, NOT the released CO3D sampler.

This is deliberately metadata-only: it does not fetch RGB/depth/masks, load a
model, run PnP, or claim the author's Table 1 protocol.  The plan is a stable
engineering proposal only. Its distinct-sequence/random.sample policy differs
from 1000 @ Co3d_Multiview. Cache ceilings are NOT a measured storage budget.
The original JSON is preserved; do not use this proposal for a paper run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path


ALL_MANIFEST = Path("data/co3d_test_metadata/selected_seqs_test_reconstructed.json")
SEEN41_MANIFEST = Path("data/co3d_test_metadata/selected_seqs_test_seen41_candidate.json")
NOMINAL_DRAWS = Path("data/co3d_test_metadata/nominal_all_sequences_draws_seed42.json")
ARCHIVES = Path("data/co3d_test_metadata/archives")
OUT = Path("results/co3d_51_1000_protocol_budget_20261003.json")

REQUESTS = 1000
VIEWS_PER_REQUEST = 10
RAW_CACHE_BOUND = 2 * 1024**3
PROCESSED_CACHE_BOUND = 512 * 1024**2
RESERVE = 1024**3


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest_stats(manifest: dict) -> dict:
    return {
        "category_count": len(manifest),
        "sequence_count": sum(len(scenes) for scenes in manifest.values()),
        "candidate_frame_count": sum(
            len(frames) for scenes in manifest.values() for frames in scenes.values()
        ),
    }


def build_plan(manifest: dict, requests: int = REQUESTS, seed: int = 42) -> list[dict]:
    """Build the non-source engineering proposal; never reads image members."""
    rows = [
        {"category": category, "scene": scene, "frame_numbers": list(frames)}
        for category, scenes in manifest.items()
        for scene, frames in scenes.items()
    ]
    if requests <= 0 or requests > len(rows):
        raise ValueError("request count must be within the candidate sequence pool")
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    # Guarantee one sequence from every non-empty category before filling the
    # remaining request slots from the same seeded permutation.
    first_by_category = {}
    for row in shuffled:
        first_by_category.setdefault(row["category"], row)
    chosen = list(first_by_category.values())
    chosen_ids = {(r["category"], r["scene"]) for r in chosen}
    chosen.extend(
        row for row in shuffled if (row["category"], row["scene"]) not in chosen_ids
    )
    chosen = chosen[:requests]
    plan = []
    for request, row in enumerate(chosen):
        frames = row["frame_numbers"]
        if len(frames) < VIEWS_PER_REQUEST:
            raise ValueError("candidate sequence has fewer than 10 frames")
        selected = rng.sample(frames, VIEWS_PER_REQUEST)
        plan.append(
            {
                "request": request,
                "category": row["category"],
                "scene": row["scene"],
                "frame_numbers": selected,
                "distinct_frame_count": len(set(selected)),
            }
        )
    return plan


def audit(root: Path = Path("."), requests: int = REQUESTS) -> dict:
    all_manifest = json.loads((root / ALL_MANIFEST).read_text())
    seen41_manifest = json.loads((root / SEEN41_MANIFEST).read_text())
    nominal = json.loads((root / NOMINAL_DRAWS).read_text())
    all_stats = manifest_stats(all_manifest)
    seen41_stats = manifest_stats(seen41_manifest)
    extra_categories = sorted(set(all_manifest) - set(seen41_manifest))
    if not extra_categories or any(all_manifest[c] == {} for c in extra_categories):
        # Empty categories would make the 51-category statement misleading.
        raise ValueError("official 51-category manifest has an unexpected empty category")
    if any(all_manifest[c] != seen41_manifest[c] for c in seen41_manifest):
        raise ValueError("the seen41 candidate is not an exact prefix of the 51-category manifest")
    plan = build_plan(all_manifest, requests=requests)
    if len(plan) != requests or len({(r["category"], r["scene"]) for r in plan}) != requests:
        raise ValueError("request plan is not one request per distinct sequence")
    if any(r["distinct_frame_count"] != VIEWS_PER_REQUEST for r in plan):
        raise ValueError("request plan contains duplicate frame numbers")
    archive_bytes = sum(p.stat().st_size for p in (root / ARCHIVES).glob("*.zip"))
    usage = shutil.disk_usage(root)
    return {
        "status": "51_category_1000_request_metadata_plan_verified_not_rgb_or_pose_ready",
        "paper_mapping": "section 4.2 / Table 1 data feasibility and resume planning only",
        "source_files_sha256": {
            str(ALL_MANIFEST): sha(root / ALL_MANIFEST),
            str(SEEN41_MANIFEST): sha(root / SEEN41_MANIFEST),
            str(NOMINAL_DRAWS): sha(root / NOMINAL_DRAWS),
        },
        "official_51_manifest": all_stats,
        "archived_seen41_candidate": seen41_stats,
        "additional_categories_not_in_seen41": extra_categories,
        "additional_sequence_count": all_stats["sequence_count"] - seen41_stats["sequence_count"],
        "additional_frame_count": all_stats["candidate_frame_count"] - seen41_stats["candidate_frame_count"],
        "nominal_draw_scope": nominal["scope"],
        "nominal_draw_row_count": len(nominal["rows"]),
        "fixed_plan": {
            "seed": 42,
            "request_count": requests,
            "views_per_request": VIEWS_PER_REQUEST,
            "distinct_sequence_count": len(plan),
            "category_counts": dict(sorted(Counter(r["category"] for r in plan).items())),
            "unique_frame_slots": len({(r["category"], r["scene"], f) for r in plan for f in r["frame_numbers"]}),
            "request_plan_sha256": hashlib.sha256(
                json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "storage_envelope": {
            "official_metadata_archive_bytes": archive_bytes,
            "raw_cache_bound_bytes": RAW_CACHE_BOUND,
            "processed_cache_bound_bytes": PROCESSED_CACHE_BOUND,
            "required_free_reserve_bytes": RESERVE,
            "bounded_workspace_allowance_bytes": RAW_CACHE_BOUND + PROCESSED_CACHE_BOUND + RESERVE,
            "current_filesystem_total_bytes": usage.total,
            "current_filesystem_free_bytes": usage.free,
            "free_after_required_reserve_bytes": usage.free - RESERVE,
            "network_bytes_transferred": 0,
            "model_forward_count": 0,
        },
        "author_protocol_equivalence_verified": False,
        "rgb_depth_mask_camera_gt_ready": False,
        "formal_Table1_result": False,
        "full_paper_completed": False,
        "note": (
            "The plan is a deterministic engineering checkpoint, not the author's exact split, "
            "not a dataset download, and not a pose score. If a future lazy run exceeds the "
            "cache envelope it must stop and preserve the checkpoint rather than evicting data."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=OUT)
    parser.add_argument("--requests", type=int, default=REQUESTS)
    args = parser.parse_args()
    result = audit(requests=args.requests)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    # Never replace an earlier disk snapshot or result with a later rerun.
    with args.output_json.open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "official_51_manifest", "archived_seen41_candidate", "fixed_plan", "storage_envelope")}, indent=2))


if __name__ == "__main__":
    main()
