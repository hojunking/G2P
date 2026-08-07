"""Gaussian-to-Point feature augmentation (paper Sec. 3.3, Eq. 1-3, Alg. 1).

For every point ``p_i`` we
  1. collect candidate Gaussians whose centroids lie within a Euclidean radius
     ``r_g`` of ``p_i``;
  2. rank the candidates by Mahalanobis distance under each Gaussian's own
     covariance and keep the ``k`` closest;
  3. weight them by the normalised inverse Mahalanobis distance (Eq. 1);
  4. aggregate scale ``S'`` and opacity ``alpha'`` (Eq. 2).

Two fallbacks exist for points with too few candidates inside ``r_g``; see
:data:`FALLBACK_MODES`.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .gaussian import SH_DIM, build_covariance, weighted_direction


FALLBACK_MODES = ("nearest", "knn")
"""How to handle points with too few Gaussians inside ``r_g``.

``nearest`` (default)
    Only a point with *zero* candidates falls back, and then to the single
    Euclidean nearest Gaussian. This is what Alg. 1 of the paper specifies.

``knn``
    A point with fewer than ``k`` candidates falls back to the ``k`` Euclidean
    nearest Gaussians, weighted by Euclidean distance. This reproduces the
    original experimental artifacts bit-exactly.

On ScanNet the two agree on ~96% of points; the ~4% with a sparse Gaussian
neighbourhood get a different neighbour set.
"""


def _split_attributes(features, include_sh):
    idx = 0
    scale = features[:, idx:idx + 3]; idx += 3
    opacity = features[:, idx:idx + 1]; idx += 1
    rotation = features[:, idx:idx + 4]; idx += 4
    sh = None
    if include_sh and features.shape[1] >= idx + SH_DIM:
        sh = features[:, idx:idx + SH_DIM]
    return scale, opacity, rotation, sh


def augment_mahalanobis(
    points,
    gaussian_points,
    gaussian_features,
    k_neighbors=20,
    radius=0.06,
    include_sh=False,
    fallback="nearest",
):
    """Covariance-aware Gaussian-to-Point augmentation (the paper default).

    Args:
        points: ``(N, 3)`` point-cloud coordinates ``mu^p``.
        gaussian_points: ``(M, 3)`` Gaussian centroids ``mu^g``.
        gaussian_features: ``(M, 8)`` activated attributes from
            :func:`g2p_prep.gaussian.preprocess_3dgs_attributes`.
        k_neighbors: ``k`` in Eq. (1). Paper default 20.
        radius: ``r_g`` candidate search radius in metres. Paper default 0.06.
        include_sh: additionally aggregate the 48-dim SH block.
        fallback: one of :data:`FALLBACK_MODES`.

    Returns:
        ``(N, 7)`` array laid out as
        ``scale S'(3) | opacity alpha'(1) | rotation direction(3)``,
        extended to ``(N, 55)`` when ``include_sh``.
    """
    if fallback not in FALLBACK_MODES:
        raise ValueError(f"fallback must be one of {FALLBACK_MODES}, got {fallback!r}")

    scale_g, opacity_g, rotation_g, sh_g = _split_attributes(
        gaussian_features, include_sh
    )

    # --- Stage 1: pre-compute the inverse covariance of every Gaussian. ---
    num_gaussians = len(gaussian_points)
    inv_covariances = np.zeros((num_gaussians, 3, 3))
    for j in range(num_gaussians):
        sigma = build_covariance(np.maximum(scale_g[j], 1e-8), rotation_g[j])
        inv_covariances[j] = np.linalg.inv(sigma + np.eye(3) * 1e-8)

    # --- Stage 2: radius candidates, then Mahalanobis ranking. ---
    radius_nbrs = NearestNeighbors(radius=radius, algorithm="ball_tree").fit(
        gaussian_points
    )
    candidates_per_point = radius_nbrs.radius_neighbors(
        points, return_distance=False
    )
    fallback_nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(gaussian_points)

    # Points with too few candidates fall back to Euclidean matching. Under
    # "knn" the threshold is k neighbours, under "nearest" it is a single one.
    fallback_threshold = k_neighbors if fallback == "knn" else 1
    fallback_k = k_neighbors if fallback == "knn" else 1

    final_indices = []
    final_distances = []
    for i in range(len(points)):
        candidates = candidates_per_point[i]

        if len(candidates) < fallback_threshold:
            dist, ind = fallback_nbrs.kneighbors(
                [points[i]], n_neighbors=fallback_k
            )
            final_distances.append(dist[0])
            final_indices.append(ind[0])
            continue

        diff = points[i] - gaussian_points[candidates]
        temp = np.einsum("mjk,mk->mj", inv_covariances[candidates], diff)
        mahalanobis = np.sqrt(np.einsum("mj,mj->m", diff, temp))

        top_k = np.argsort(mahalanobis)[:k_neighbors]
        final_indices.append(candidates[top_k])
        final_distances.append(mahalanobis[top_k])

    # --- Weights (Eq. 1) and attribute aggregation (Eq. 2). ---
    num_points = len(points)
    scale_aug = np.zeros((num_points, 3), dtype=np.float32)
    opacity_aug = np.zeros((num_points, 1), dtype=np.float32)
    rotation_aug = np.zeros((num_points, 3), dtype=np.float32)
    sh_aug = (
        np.zeros((num_points, SH_DIM), dtype=np.float32)
        if include_sh and sh_g is not None
        else None
    )

    for i in range(num_points):
        inds = final_indices[i]
        weights = 1.0 / (final_distances[i] + 1e-8)
        weights = weights / np.sum(weights)

        scale_aug[i] = np.sum(scale_g[inds] * weights[:, np.newaxis], axis=0)
        opacity_aug[i] = np.sum(opacity_g[inds] * weights[:, np.newaxis], axis=0)
        rotation_aug[i] = weighted_direction(rotation_g[inds], weights)
        if sh_aug is not None:
            sh_aug[i] = np.sum(sh_g[inds] * weights[:, np.newaxis], axis=0)

    blocks = [scale_aug, opacity_aug, rotation_aug]
    if sh_aug is not None:
        blocks.append(sh_aug)
    return np.concatenate(blocks, axis=-1)


def augment_euclidean(
    points,
    gaussian_points,
    gaussian_features,
    k_neighbors=20,
    include_sh=False,
):
    """Isotropic Euclidean-kNN augmentation.

    Ablation baseline for Tab. 10 ("Metric: Euclidean"). Uses plain kNN with
    inverse-Euclidean-distance weights instead of the covariance-aware
    matching of :func:`augment_mahalanobis`.
    """
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm="auto").fit(
        gaussian_points
    )
    distances, indices = nbrs.kneighbors(points)

    weights = 1.0 / (distances + 1e-8)
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    scale_g, opacity_g, rotation_g, sh_g = _split_attributes(
        gaussian_features, include_sh
    )

    scale_aug = np.sum(scale_g[indices] * weights[..., np.newaxis], axis=1)
    opacity_aug = np.sum(opacity_g[indices] * weights[..., np.newaxis], axis=1)

    rotation_neighbors = rotation_g[indices]
    rotation_aug = np.zeros((len(points), 3), dtype=np.float32)
    for i in range(len(points)):
        rotation_aug[i] = weighted_direction(rotation_neighbors[i], weights[i])

    blocks = [scale_aug, opacity_aug, rotation_aug]
    if include_sh and sh_g is not None:
        blocks.append(np.sum(sh_g[indices] * weights[..., np.newaxis], axis=1))
    return np.concatenate(blocks, axis=-1)
