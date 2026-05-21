"""Jaccard similarity for set-valued neuron masks.

Operates on Python `set[int]` (the form returned by
`csp_atlas.io.load_universal_masks`). There is a separate
array-based Jaccard in `circuits.extraction.metrics` for the boolean-mask
inputs used during the binarisation/marginalisation stage; the two are
kept distinct because their inputs are not interchangeable.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def jaccard_sets(a: set[int], b: set[int]) -> float:
    """Jaccard = |A ∩ B| / |A ∪ B|. Returns 0.0 if both empty."""
    assert isinstance(a, set), f"a must be a set, got {type(a).__name__}"
    assert isinstance(b, set), f"b must be a set, got {type(b).__name__}"
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def pairwise_jaccard_matrix(
    masks: dict[str, set[int]],
) -> tuple[list[str], np.ndarray]:
    """Pairwise Jaccard similarity over a dict of named sets.

    Returns `(names, matrix)` where `names` is the sorted key list
    (deterministic order) and `matrix` is `(n, n)` symmetric with
    diagonal = 1 for non-empty sets, 0 otherwise.
    """
    assert isinstance(masks, dict), f"masks must be a dict, got {type(masks).__name__}"
    assert all(isinstance(v, set) for v in masks.values()), (
        "all values in masks must be sets"
    )
    names = sorted(masks.keys())
    n = len(names)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i, j in combinations(range(n), 2):
        sim = jaccard_sets(masks[names[i]], masks[names[j]])
        matrix[i, j] = sim
        matrix[j, i] = sim
    for i, name in enumerate(names):
        matrix[i, i] = 1.0 if masks[name] else 0.0
    return names, matrix
