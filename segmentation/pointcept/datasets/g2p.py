"""ScanNet v2 datasets that additionally load the G2P preparation-stage assets.

Two assets are produced offline by ``preparation/`` and consumed here:

``features.npy``  float32 ``(N, 7)`` -- ``scale(3) | opacity(1) | rotation(3)``,
    the Gaussian attributes aggregated onto each point (paper Sec. 3.3).
    ``--include-sh`` extends it to ``(N, 55)``.
``boundary.npy``  int64 ``(N,)`` in ``{0, 1}`` -- the scale-based boundary
    pseudo-labels (paper Sec. 3.4).

Which columns of ``features.npy`` are exposed is controlled by ``features_flag``.
The paper uses ``["opacity"]``: the appearance encoder is trained on
``(mu^p, c, alpha')`` and the same opacity channel feeds the distillation
teacher.
"""

import os

import numpy as np

from pointcept.utils.logger import get_root_logger

from .builder import DATASETS
from .scannet import ScanNetDataset

# Column layout of features.npy, in the order the preparation stage writes it.
FEATURE_LAYOUT = (
    ("scale", 3),
    ("opacity", 1),
    ("rotation", 3),
    ("sh", 48),
)


class GaussianAssetMixin:
    """Adds ``features`` and ``boundary`` loading to a Pointcept dataset.

    Args:
        features_root: root of the stage-1 output. ``None`` leaves whatever the
            base dataset loaded untouched.
        features_flag: which blocks of ``features.npy`` to expose, e.g.
            ``["opacity"]``. Empty means none.
        sh_degree: spherical-harmonics degree to keep when ``"sh"`` is
            requested; 0 disables it.
        boundary_root: root of the stage-2 output. ``None`` fills ``boundary``
            with zeros, which disables boundary supervision.
    """

    def __init__(
        self,
        features_root=None,
        features_flag=None,
        sh_degree=0,
        boundary_root=None,
        **kwargs,
    ):
        self.features_root = features_root
        self.features_flag = features_flag or []
        self.sh_degree = sh_degree
        self.boundary_root = boundary_root
        # Set before super().__init__, which builds the data list and may
        # already touch get_data.
        super().__init__(**kwargs)

    def get_data(self, idx):
        data_dict = super().get_data(idx)
        scene_name = self.get_data_name(idx)
        num_points = data_dict["coord"].shape[0]

        if self.features_root is not None:
            data_dict["features"] = self._load_features(scene_name, num_points)
        if self.loads_boundary:
            data_dict["boundary"] = self._load_boundary(scene_name, num_points)
        return data_dict

    # Subclasses that do not use boundary supervision set this to False.
    loads_boundary = True

    def _load_features(self, scene_name, num_points):
        logger = get_root_logger()
        empty = np.zeros((num_points, 0), dtype=np.float32)

        if not self.features_flag:
            logger.warning(
                f"features_root is set but features_flag is empty; "
                f"no Gaussian features loaded for {scene_name}."
            )
            return empty

        path = os.path.join(self.features_root, self.split, scene_name, "features.npy")
        if not os.path.exists(path):
            logger.warning(f"features.npy not found at {path}; using no Gaussian features.")
            return empty

        features = np.load(path).astype(np.float32)
        if len(features) != num_points:
            raise ValueError(
                f"{path} has {len(features)} rows but scene {scene_name} has "
                f"{num_points} points. Regenerate the preparation-stage assets."
            )

        selected = []
        offset = 0
        for name, dim in FEATURE_LAYOUT:
            if name == "sh":
                if not self.sh_degree:
                    break
                dim = min(dim, 3 * (self.sh_degree + 1) ** 2)
            if name in self.features_flag:
                if offset + dim > features.shape[1]:
                    raise ValueError(
                        f"{path} has {features.shape[1]} columns, too few for "
                        f"features_flag={self.features_flag} (need {offset + dim} "
                        f"for '{name}'). Regenerate with --include-sh if you want SH."
                    )
                selected.append(features[:, offset:offset + dim])
            offset += dim

        return np.concatenate(selected, axis=-1) if selected else empty

    def _load_boundary(self, scene_name, num_points):
        logger = get_root_logger()
        if self.boundary_root is None:
            logger.warning(
                f"boundary_root is not set; boundary supervision is disabled "
                f"for {scene_name}."
            )
            return np.zeros(num_points, dtype=np.int32)

        path = os.path.join(self.boundary_root, self.split, scene_name, "boundary.npy")
        if not os.path.exists(path):
            logger.warning(f"boundary.npy not found at {path}; filling with zeros.")
            return np.zeros(num_points, dtype=np.int32)

        boundary = np.load(path).reshape(-1).astype(np.int32)
        if len(boundary) != num_points:
            raise ValueError(
                f"{path} has {len(boundary)} rows but scene {scene_name} has "
                f"{num_points} points. Regenerate the preparation-stage assets."
            )
        return boundary


@DATASETS.register_module()
class ScanNetDatasetBoundary(GaussianAssetMixin, ScanNetDataset):
    """ScanNet v2 with Gaussian features and boundary pseudo-labels.

    Used for the main G2P training stage: the backbone consumes
    ``(coord, color, normal)`` while ``features`` feeds the distillation teacher
    and ``boundary`` supervises the boundary head.
    """


@DATASETS.register_module()
class ScanNetDistillDataset(GaussianAssetMixin, ScanNetDataset):
    """ScanNet v2 with Gaussian features only.

    Used to pre-train the appearance encoder (teacher) on
    ``(mu^p, c, alpha') in R^7``; no boundary supervision is involved.
    """

    loads_boundary = False
