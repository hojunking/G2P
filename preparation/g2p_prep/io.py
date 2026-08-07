"""I/O helpers for Pointcept scenes and 3D Gaussian Splatting PLY files."""

import os

import numpy as np
from plyfile import PlyData

SH_REST_DIM = 45
SH_DC_DIM = 3


def load_scene(scene_dir, required_assets, optional_assets=("segment200", "instance")):
    """Load a Pointcept-format scene directory.

    Args:
        scene_dir: directory holding ``coord.npy``, ``color.npy``, ...
        required_assets: asset names that must be present.
        optional_assets: asset names filled with -1 when missing.

    Returns:
        dict mapping asset name to array.
    """
    data = {}
    for key in required_assets:
        path = os.path.join(scene_dir, f"{key}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{key}.npy not found in {scene_dir}")
        data[key] = np.load(path)

    reference = data[required_assets[-1]]
    for key in optional_assets:
        if key in data:
            continue
        path = os.path.join(scene_dir, f"{key}.npy")
        if os.path.exists(path):
            data[key] = np.load(path)
        else:
            data[key] = np.full_like(reference, -1, dtype=np.int64)
    return data


def load_3dgs_ply(path, include_sh=False):
    """Read a 3D Gaussian Splatting PLY.

    Returns:
        points: ``(M, 3)`` Gaussian centroids ``mu^g``.
        normals: ``(M, 3)`` per-vertex normals as stored in the PLY (3DGS
            writes zeros here; kept for compatibility).
        raw_features: ``(M, 8)`` laid out as
            ``scale(3) | opacity(1) | rotation quaternion(4)``, or ``(M, 56)``
            with ``f_dc(3) | f_rest(45)`` appended when ``include_sh``.
            Values are the raw, un-activated parameters stored by 3DGS.
    """
    with open(path, "rb") as f:
        ply = PlyData.read(f)
    vertex = ply["vertex"]

    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
    normals = np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], axis=-1)

    raw_attr = [
        np.vstack([vertex[f"scale_{i}"] for i in range(3)]).T,
        vertex["opacity"][:, None],
        np.vstack([vertex[f"rot_{i}"] for i in range(4)]).T,
    ]
    if include_sh:
        sh_dc = np.vstack([vertex[f"f_dc_{i}"] for i in range(SH_DC_DIM)]).T
        sh_rest = np.vstack([vertex[f"f_rest_{i}"] for i in range(SH_REST_DIM)]).T
        raw_attr.extend([sh_dc, sh_rest])

    raw_features = np.hstack(raw_attr)
    return points, normals, raw_features


def resolve_gs_path(gs_root, scene, template):
    """Build the Gaussian PLY path for one scene."""
    return os.path.join(gs_root, template.format(scene=scene))


def read_scene_list(path):
    """Read a newline-delimited scene-id list, ignoring blanks and comments."""
    with open(path) as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]
