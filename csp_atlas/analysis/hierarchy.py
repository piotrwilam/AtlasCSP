"""Hierarchical clustering on (1 − Jaccard) distance — paper §6.

Wraps scipy's linkage with a fixed, documented choice (Ward + 1-Jaccard)
so the dendrogram is reproducible across scipy versions. If a scipy
upgrade changes tie-breaking, the locked dendrogram numbers will fail
and we'll catch it.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster as _scipy_fcluster
from scipy.cluster.hierarchy import linkage as _scipy_linkage
from scipy.spatial.distance import squareform


def ward_linkage_from_jaccard(jaccard_matrix: np.ndarray) -> np.ndarray:
    """Ward-linkage hierarchical clustering on `1 − Jaccard` distance.

    Returns the `(n-1, 4)` scipy linkage matrix.
    """
    assert jaccard_matrix.ndim == 2 and jaccard_matrix.shape[0] == jaccard_matrix.shape[1], (
        f"expected square 2-D matrix, got shape {jaccard_matrix.shape}"
    )
    assert np.allclose(jaccard_matrix, jaccard_matrix.T), "matrix must be symmetric"

    distance_matrix = 1.0 - jaccard_matrix
    np.fill_diagonal(distance_matrix, 0.0)  # guard against rounding-induced ~0
    condensed = squareform(distance_matrix, checks=False)
    return _scipy_linkage(condensed, method="ward")


def cut_dendrogram_at_k_clusters(
    linkage: np.ndarray,
    labels: list[str],
    k: int,
) -> dict[str, int]:
    """Cut a Ward linkage dendrogram into exactly `k` flat clusters.

    Used in §6 to identify the atomicity super-cluster.
    """
    assert linkage.ndim == 2 and linkage.shape[1] == 4, (
        f"linkage must be (n-1, 4), got shape {linkage.shape}"
    )
    assert linkage.shape[0] == len(labels) - 1, (
        f"linkage has {linkage.shape[0]} rows but labels has {len(labels)} entries "
        f"(expected {linkage.shape[0] + 1})"
    )
    assert k >= 1, f"k must be ≥ 1, got {k}"
    assert k <= len(labels), (
        f"cannot cut into more clusters ({k}) than leaves ({len(labels)})"
    )
    cluster_ids = _scipy_fcluster(linkage, t=k, criterion="maxclust")
    return {label: int(cid) for label, cid in zip(labels, cluster_ids)}
