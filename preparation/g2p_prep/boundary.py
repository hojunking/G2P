"""Gaussian scale-based boundary extraction (paper Sec. 3.4, Eq. 4-5).

Small aggregated Gaussian scales concentrate along object boundaries and thin
structures, while large scales dominate smooth planar regions. We therefore
prune the largest-scale points of the object-focused subset and treat the
remainder as boundary candidates ``B_scale``. These are unioned with the
conventional semantic boundary ``B_sem`` to form the final pseudo-label
``B = B_scale u B_sem``.
"""

import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree


def scale_boundary(features, labels, eta, excluded_classes):
    """``B_scale`` of Eq. (5).

    Args:
        features: ``(N, >=3)`` augmented features; columns 0:3 hold ``S'``.
        labels: ``(N,)`` semantic labels used to build ``P'_obj``.
        eta: pruning ratio. ``eta = 0.7`` removes the 70% of points with the
            largest ``||S'||_2`` and keeps the remaining 30% as boundary
            candidates.
        excluded_classes: background/ignore class ids removed from ``P'_obj``.

    Returns:
        ``(N,)`` int64 array in ``{0, 1}``.
    """
    num_points = features.shape[0]
    boundary = np.zeros(num_points, dtype=np.int64)
    if eta <= 0:
        return boundary

    if labels.ndim > 1:
        labels = labels[:, 0]

    # Eq. (4): scale magnitude.
    scales = np.linalg.norm(features[:, 0:3], axis=1)

    # P'_obj: drop background classes before computing the threshold, so that
    # large planar regions do not dominate the scale distribution.
    inclusion_mask = ~np.isin(labels, excluded_classes)
    scales_obj = scales[inclusion_mask]
    if scales_obj.size == 0:
        return boundary

    # tau_eta: the (1 - eta) percentile of the object-focused scale magnitudes.
    threshold = np.percentile(scales_obj, 100 * (1.0 - eta))
    boundary[(scales <= threshold) & inclusion_mask] = 1
    return boundary


def semantic_boundary(points, labels, radius):
    """``B_sem``: a point is a boundary if any neighbour within ``r_s`` has a
    different semantic label.

    Returns:
        ``(N,)`` int8 array in ``{0, 1}``.
    """
    if points is None or labels is None:
        raise ValueError("Points and labels cannot be None.")
    if len(points) != len(labels):
        raise ValueError("Points and labels must have the same length.")

    tree = KDTree(points)
    neighbor_indices = tree.query_radius(points, r=radius)

    boundary = np.zeros(len(points), dtype=np.int8)
    for i in range(len(points)):
        neighbors = neighbor_indices[i]
        if len(neighbors) <= 1:
            continue
        if np.any(labels[neighbors] != labels[i]):
            boundary[i] = 1
    return boundary


def merge_boundaries(*boundaries):
    """Union of the available boundary cues, as int64 ``{0, 1}``."""
    present = [b for b in boundaries if b is not None]
    if not present:
        raise ValueError("At least one boundary source is required.")

    merged = present[0].astype(bool)
    for b in present[1:]:
        merged = np.logical_or(merged, b)
    return merged.astype(np.int64)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def save_boundary_ply(
    points,
    boundary,
    output_path,
    boundary_hex="#E43636",
    non_boundary_hex="#808080",
):
    """Write a PLY colouring boundary points red and the rest grey."""
    colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    colors[boundary == 1] = hex_to_rgb(boundary_hex)
    colors[boundary == 0] = hex_to_rgb(non_boundary_hex)

    vertex = np.empty(
        points.shape[0],
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = (
        colors[:, 0], colors[:, 1], colors[:, 2],
    )

    PlyData([PlyElement.describe(vertex, "vertex")], text=True).write(output_path)
