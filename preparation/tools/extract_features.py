#!/usr/bin/env python3
"""Stage 1: Gaussian-to-Point feature augmentation (paper Sec. 3.3).

Writes ``<output_root>/<split>/<scene>/features.npy``, a float32 ``(N, 7)``
array laid out as ``scale S'(3) | opacity alpha'(1) | rotation direction(3)``.
``N`` matches the scene's ``coord.npy``.

Example:
    python tools/extract_features.py --dataset scannet \\
        --config configs/scannet.yaml \\
        --output-root /data/features/mah_k20 \\
        --k 20 --radius 0.06 --num-workers 24
"""

import argparse
import functools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g2p_prep import (  # noqa: E402
    augment_euclidean,
    augment_mahalanobis,
    get_dataset,
    load_3dgs_ply,
    load_config,
    load_scene,
    preprocess_3dgs_attributes,
    read_scene_list,
    resolve_gs_path,
)
from g2p_prep.datasets import SAMPLE_META  # noqa: E402
from g2p_prep.runner import report, run_scenes  # noqa: E402


def process_scene(
    scene,
    split,
    spec,
    scene_root,
    gs_root,
    gs_ply_template,
    output_root,
    k_neighbors,
    radius,
    metric,
    include_sh,
    fallback,
    overwrite,
):
    output_dir = os.path.join(output_root, split, scene)
    output_path = os.path.join(output_dir, "features.npy")
    if os.path.exists(output_path) and not overwrite:
        return "skipped"

    scene_dir = os.path.join(scene_root, split, scene)
    gs_path = resolve_gs_path(gs_root, scene, gs_ply_template)
    if not os.path.exists(os.path.join(scene_dir, "coord.npy")):
        raise FileNotFoundError(f"scene not found: {scene_dir}")
    if not os.path.exists(gs_path):
        raise FileNotFoundError(f"Gaussian PLY not found: {gs_path}")

    scene_data = load_scene(scene_dir, spec.required_assets)
    points = scene_data["coord"]

    gaussian_points, _, raw_features = load_3dgs_ply(gs_path, include_sh=include_sh)
    gaussian_features = preprocess_3dgs_attributes(raw_features, include_sh=include_sh)

    if metric == "mahalanobis":
        features = augment_mahalanobis(
            points,
            gaussian_points,
            gaussian_features,
            k_neighbors=k_neighbors,
            radius=radius,
            include_sh=include_sh,
            fallback=fallback,
        )
    else:
        features = augment_euclidean(
            points,
            gaussian_points,
            gaussian_features,
            k_neighbors=k_neighbors,
            include_sh=include_sh,
        )

    if len(features) != len(points):
        raise RuntimeError(
            f"produced {len(features)} rows for {len(points)} points"
        )

    os.makedirs(output_dir, exist_ok=True)
    np.save(output_path, features.astype(np.float32))
    return "done"


def parse_args():
    parser = argparse.ArgumentParser(
        description="G2P stage 1: Gaussian-to-Point feature augmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="scannet", choices=["scannet"])
    parser.add_argument("--config", required=True, help="Preparation YAML config.")
    parser.add_argument("--output-root", required=True, help="Where features.npy is written.")
    parser.add_argument("--scene-root", default=None, help="Override scene_root from the config.")
    parser.add_argument("--gs-root", default=None, help="Override gs_root from the config.")
    parser.add_argument(
        "--gs-ply-template",
        default=None,
        help="Gaussian PLY sub-path relative to gs_root; '{scene}' is substituted. "
             "Overrides gs_ply_template from the config and the dataset default.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--scenes", nargs="+", default=None, help="Process only these scene ids.")
    parser.add_argument("--sample-subset", action="store_true", help="Use the small ScanNet debugging split lists.")
    parser.add_argument("--k", type=int, default=20, dest="k_neighbors", help="k in Eq. (1).")
    parser.add_argument("--radius", type=float, default=0.06, help="Candidate search radius r_g in metres.")
    parser.add_argument("--metric", default="mahalanobis", choices=["mahalanobis", "euclidean"])
    parser.add_argument(
        "--fallback",
        default="nearest",
        choices=["nearest", "knn"],
        help="Handling of points with too few Gaussians inside r_g. "
             "'nearest' follows Alg. 1; 'knn' reproduces the original artifacts.",
    )
    parser.add_argument("--include-sh", action="store_true", help="Also aggregate the 48-dim SH block.")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true", help="Recompute scenes that already have features.npy.")
    parser.add_argument("--continue-on-error", action="store_true", help="Exit 0 even when some scenes failed.")
    return parser.parse_args()


def main():
    args = parse_args()
    spec = get_dataset(args.dataset)
    config = load_config(
        args.config,
        overrides={
            "scene_root": args.scene_root,
            "gs_root": args.gs_root,
            "gs_ply_template": args.gs_ply_template,
        },
    )
    gs_ply_template = config.get("gs_ply_template") or spec.gs_ply_template

    print(
        f"dataset={args.dataset}  metric={args.metric}  k={args.k_neighbors}  "
        f"r_g={args.radius}  fallback={args.fallback}  include_sh={args.include_sh}"
    )
    print(f"scenes:    {config['scene_root']}")
    print(f"gaussians: {config['gs_root']}/{gs_ply_template}")
    print(f"output:    {args.output_root}")

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
            gs_root=config["gs_root"],
            gs_ply_template=gs_ply_template,
            output_root=args.output_root,
            k_neighbors=args.k_neighbors,
            radius=args.radius,
            metric=args.metric,
            include_sh=args.include_sh,
            fallback=args.fallback,
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
