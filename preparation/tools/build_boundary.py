#!/usr/bin/env python3
"""Stage 2: scale-based boundary pseudo-label extraction (paper Sec. 3.4).

Writes ``<output_root>/<split>/<scene>/boundary.npy``, an int64 ``(N,)`` array
in ``{0, 1}``. Only columns ``0:3`` (the aggregated Gaussian scale ``S'``) of
``features.npy`` are read.

Example:
    python tools/build_boundary.py --dataset scannet \\
        --config configs/scannet.yaml \\
        --features-root /data/features/mah_k20 \\
        --output-root /data/boundary/mah_k20_scale07_sem004 \\
        --method both --eta 0.7 --semantic-radius 0.04
"""

import argparse
import functools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g2p_prep import (  # noqa: E402
    get_dataset,
    load_config,
    load_scene,
    merge_boundaries,
    read_scene_list,
    scale_boundary,
    semantic_boundary,
)
from g2p_prep.datasets import SAMPLE_META  # noqa: E402
from g2p_prep.runner import report, run_scenes  # noqa: E402


def process_scene(
    scene,
    split,
    spec,
    scene_root,
    features_root,
    output_root,
    method,
    eta,
    semantic_radius,
    overwrite,
):
    output_dir = os.path.join(output_root, split, scene)
    output_path = os.path.join(output_dir, "boundary.npy")
    if os.path.exists(output_path) and not overwrite:
        return "skipped"

    scene_dir = os.path.join(scene_root, split, scene)
    if not os.path.exists(os.path.join(scene_dir, "coord.npy")):
        raise FileNotFoundError(f"scene not found: {scene_dir}")

    scene_data = load_scene(scene_dir, spec.required_assets)
    points = scene_data["coord"]
    labels = scene_data[spec.label_key]

    boundary_scale = None
    if method in ("scale", "both"):
        features_path = os.path.join(features_root, split, scene, "features.npy")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"features.npy not found: {features_path}")
        features = np.load(features_path)
        if len(features) != len(points):
            raise RuntimeError(
                f"features.npy has {len(features)} rows for {len(points)} points"
            )
        boundary_scale = scale_boundary(
            features, labels, eta=eta, excluded_classes=spec.excluded_classes
        )

    boundary_sem = None
    if method in ("semantic", "both"):
        boundary_sem = semantic_boundary(points, labels, radius=semantic_radius)

    boundary = merge_boundaries(boundary_scale, boundary_sem)

    os.makedirs(output_dir, exist_ok=True)
    np.save(output_path, boundary)
    return "done"


def parse_args():
    parser = argparse.ArgumentParser(
        description="G2P stage 2: boundary pseudo-label extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="scannet", choices=["scannet"])
    parser.add_argument("--config", required=True, help="Preparation YAML config.")
    parser.add_argument("--features-root", required=True, help="Root of stage-1 features.npy.")
    parser.add_argument("--output-root", required=True, help="Where boundary.npy is written.")
    parser.add_argument("--scene-root", default=None, help="Override scene_root from the config.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--scenes", nargs="+", default=None, help="Process only these scene ids.")
    parser.add_argument("--sample-subset", action="store_true", help="Use the small ScanNet debugging split lists.")
    parser.add_argument(
        "--method",
        default="both",
        choices=["scale", "semantic", "both"],
        help="scale = B_scale only, semantic = B_sem only, both = their union (paper default).",
    )
    parser.add_argument("--eta", type=float, default=0.7, help="Scale pruning ratio eta of Eq. (5).")
    parser.add_argument("--semantic-radius", type=float, default=0.04, help="Semantic boundary radius r_s in metres.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true", help="Recompute scenes that already have boundary.npy.")
    parser.add_argument("--continue-on-error", action="store_true", help="Exit 0 even when some scenes failed.")
    return parser.parse_args()


def main():
    args = parse_args()
    spec = get_dataset(args.dataset)
    config = load_config(args.config, overrides={"scene_root": args.scene_root})

    print(
        f"dataset={args.dataset}  method={args.method}  eta={args.eta}  "
        f"r_s={args.semantic_radius}  excluded={spec.excluded_classes}"
    )
    print(f"features: {args.features_root}\noutput: {args.output_root}")

    timings = {}
    failures = []
    for split in args.splits:
        if args.scenes:
            scenes = list(args.scenes)
        else:
            meta_name = (
                SAMPLE_META[split]
                if args.sample_subset
                else (spec.train_meta if split == "train" else spec.val_meta)
            )
            scenes = read_scene_list(os.path.join(config["meta_root"], meta_name))

        worker = functools.partial(
            process_scene,
            split=split,
            spec=spec,
            scene_root=config["scene_root"],
            features_root=args.features_root,
            output_root=args.output_root,
            method=args.method,
            eta=args.eta,
            semantic_radius=args.semantic_radius,
            overwrite=args.overwrite,
        )
        elapsed, skipped, split_failures = run_scenes(
            worker, scenes, args.num_workers, desc=f"{split} scenes"
        )
        timings[split] = (elapsed, len(scenes), skipped)
        failures.extend(split_failures)

    sys.exit(report(timings, failures, args.continue_on_error))


if __name__ == "__main__":
    main()
