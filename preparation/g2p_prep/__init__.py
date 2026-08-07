"""G2P preparation stage: Gaussian-to-Point augmentation and boundary extraction."""

from .augment import augment_euclidean, augment_mahalanobis
from .boundary import (
    merge_boundaries,
    save_boundary_ply,
    scale_boundary,
    semantic_boundary,
)
from .config import load_config
from .datasets import DATASETS, SAMPLE_META, DatasetSpec, get_dataset
from .gaussian import (
    build_covariance,
    preprocess_3dgs_attributes,
    quaternion_to_direction,
    weighted_direction,
)
from .io import load_3dgs_ply, load_scene, read_scene_list, resolve_gs_path

__all__ = [
    "DATASETS",
    "SAMPLE_META",
    "DatasetSpec",
    "augment_euclidean",
    "augment_mahalanobis",
    "build_covariance",
    "get_dataset",
    "load_3dgs_ply",
    "load_config",
    "load_scene",
    "merge_boundaries",
    "preprocess_3dgs_attributes",
    "quaternion_to_direction",
    "read_scene_list",
    "resolve_gs_path",
    "save_boundary_ply",
    "scale_boundary",
    "semantic_boundary",
    "weighted_direction",
]
