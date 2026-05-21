"""Pure-compute primitives applied to the frozen artifacts.

Submodules:
    jaccard        — set-based Jaccard similarity + pairwise matrix
    decomposition  — three-way A\\B / A∩B / B\\A + concept fraction
    hierarchy      — Ward linkage on (1 − Jaccard) + dendrogram cut
    modularity     — §6.1 relaxed-modularity score (drives the atomicity finding)

All functions are pure (no I/O, no model). Inputs come from `csp_atlas.io.*`
loaders; outputs feed the experiment scripts in `experiments/`.
"""

from csp_atlas.analysis.decomposition import concept_fraction, decompose_sets
from csp_atlas.analysis.hierarchy import (
    cut_dendrogram_at_k_clusters,
    ward_linkage_from_jaccard,
)
from csp_atlas.analysis.jaccard import jaccard_sets, pairwise_jaccard_matrix
from csp_atlas.analysis.modularity import (
    DEFAULT_P_VALUES,
    DEFAULT_SIGNIFICANCE,
    compute_modularity_scores,
    trimmed_mean_jaccard,
)

__all__ = [
    "DEFAULT_P_VALUES",
    "DEFAULT_SIGNIFICANCE",
    "compute_modularity_scores",
    "concept_fraction",
    "cut_dendrogram_at_k_clusters",
    "decompose_sets",
    "jaccard_sets",
    "pairwise_jaccard_matrix",
    "trimmed_mean_jaccard",
    "ward_linkage_from_jaccard",
]
