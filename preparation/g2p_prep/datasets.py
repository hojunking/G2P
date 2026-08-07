"""Dataset registry for the G2P preparation stage.

This release covers ScanNet v2. The registry exists because benchmarks differ
in only four respects, so adding one is a table entry rather than a new script:

1. the semantic-label asset name inside a Pointcept scene directory,
2. the sub-path of the 3D Gaussian Splatting PLY inside the GS root,
3. the train/val split files under ``meta/``,
4. the background classes excluded when building the object-focused point
   cloud ``P'_obj`` of Eq. (5).

Everything else in the pipeline is dataset agnostic, so those four items are
tabulated here instead of being duplicated across per-dataset scripts.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of one benchmark."""

    name: str
    label_key: str
    """Name of the semantic-label ``.npy`` inside a Pointcept scene directory."""

    required_assets: Tuple[str, ...]
    """Assets that must exist for a scene to be processed."""

    gs_ply_template: str
    """Sub-path of the Gaussian PLY relative to the GS root. ``{scene}`` is
    substituted with the scene id."""

    train_meta: str
    val_meta: str

    excluded_classes: List[int] = field(default_factory=list)
    """Background / ignore class ids removed before scale-based boundary
    extraction (Sec. 3.4). Typically ignore, floor and wall."""


DATASETS = {
    "scannet": DatasetSpec(
        name="scannet",
        label_key="segment20",
        required_assets=("coord", "color", "normal", "segment20"),
        gs_ply_template="{scene}/ckpts/point_cloud_30000.ply",
        train_meta="scannetv2_train.txt",
        val_meta="scannetv2_val.txt",
        # -1 ignore, 0 wall, 1 floor
        excluded_classes=[-1, 0, 1],
    ),
}

# Small debugging subsets.
SAMPLE_META = {
    "train": "train100_samples.txt",
    "val": "valid20_samples.txt",
}


def get_dataset(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {sorted(DATASETS)}"
        )
    return DATASETS[name]
