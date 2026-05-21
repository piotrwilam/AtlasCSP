"""Golden-numbers tests: lock every numeric claim in the CSP-Atlas paper
to the digits reported. Any drift > 0.001 fails loudly.

These tests require the frozen artifacts at `$CSP_ATLAS_DATA_ROOT`.
Skip on CI / fresh clones without local data:
    pytest -m "not requires_data"
"""

from __future__ import annotations

import pytest

from csp_atlas.io import (
    N_LAYERS,
    load_concept_inventory,
    load_modularity_scores,
    load_universal_masks,
)
from csp_atlas.paths import DATA_ROOT

pytestmark = pytest.mark.requires_data


# §3.2 — concept space: 43 AST + 63 builtin = 106 testable concepts.
def test_concept_inventory_counts() -> None:
    fname = DATA_ROOT / "13_object_masks_eps0.5_cons0.8.h5"
    if not fname.exists():
        pytest.skip(f"frozen artifact not present at {fname}")
    ast, builtin = load_concept_inventory(eps=0.5, cons=0.8)
    assert len(ast) == 43, f"AST node count drifted: {len(ast)}, expected 43"
    assert len(builtin) == 63, f"builtin count drifted: {len(builtin)}, expected 63"
    assert len(ast) + len(builtin) == 106


# §7.1 parameter-stability: at every one of the 9 (ε, cons) settings,
# all 106 universal circuits are non-empty.
PARAM_SWEEP = [
    (eps, cons) for eps in (0.001, 0.1, 0.5) for cons in (0.2, 0.5, 0.8)
]


@pytest.mark.parametrize("eps,cons", PARAM_SWEEP)
def test_all_106_universals_non_empty_at_every_setting(eps: float, cons: float) -> None:
    """The §7.1 stability finding: 106/106 non-empty across the 3×3 sweep."""
    fname = DATA_ROOT / f"13_object_masks_eps{eps}_cons{cons}.h5"
    if not fname.exists():
        pytest.skip(f"frozen artifact not present at {fname}")
    masks = load_universal_masks(eps=eps, cons=cons)
    assert len(masks) == 106, (
        f"universal-circuit count drifted at ({eps}, {cons}): "
        f"got {len(masks)}, expected 106"
    )
    # Each concept must be non-empty at AT LEAST one layer.
    empty = [n for n, per_layer in masks.items()
             if not any(s for s in per_layer.values())]
    assert not empty, (
        f"§7.1 violated at ({eps}, {cons}): {len(empty)} concepts entirely empty: "
        f"{empty[:5]}"
    )


# §6.1 atomicity super-cluster: the 6 paper-cited concepts dominate the
# top of the relaxed-modularity ranking under the strict p=0 criterion.
ATOMICITY_SUPER_CLUSTER = {
    "ast__Assert", "ast__Break", "ast__Continue",
    "ast__Import", "ast__ImportFrom", "ast__Pass",
}


def test_top_three_modular_concepts_are_atomicity() -> None:
    """The top 3 concepts by p=0 modularity are all atomicity members.

    **Paper-vs-data note**: paper §6.1 implies all 6 atomicity concepts
    are the most modular. The data only places 3 of 6 (Break, ImportFrom,
    Assert) at the top — the other 3 (Continue, Import, Pass) tie with
    many others at score=1. The §6.1 finding is more accurately stated as
    "the atomicity super-cluster contains the three most-modular concepts;
    all six form a single hierarchical cluster (see L3 analysis)" — the
    cluster claim is the load-bearing one (`test_atomicity_cluster_forms_at_l3`).
    """
    fname = DATA_ROOT / "relaxed_modularity_scores.csv"
    if not fname.exists():
        pytest.skip(f"modularity scores not present at {fname}")
    scores = load_modularity_scores()
    top3 = scores.sort_values("p=0.0", ascending=False).head(3).index.tolist()
    expected = {"ast__Break", "ast__ImportFrom", "ast__Assert"}
    assert set(top3) == expected, (
        f"Top 3 by p=0 modularity drifted: got {top3}, expected {expected}"
    )


def test_modularity_break_has_highest_p0_score() -> None:
    """The §6.1 finding: Break has the highest p=0 significant-layer count."""
    fname = DATA_ROOT / "relaxed_modularity_scores.csv"
    if not fname.exists():
        pytest.skip(f"modularity scores not present at {fname}")
    scores = load_modularity_scores()
    top_concept = scores.sort_values("p=0.0", ascending=False).index[0]
    top_score = scores.loc[top_concept, "p=0.0"]
    assert top_concept == "ast__Break", (
        f"§6.1: top-modularity concept drifted to {top_concept!r} (expected ast__Break)"
    )
    assert top_score == 3, (
        f"§6.1: top-modularity (Break) p=0 score drifted to {top_score} (expected 3)"
    )


# §6.1 atomicity cluster: at L3 with a 4-cluster cut, the 6 atomicity
# concepts fall into a single cluster.
def test_atomicity_cluster_forms_at_l3() -> None:
    """The atomicity 6-set forms one cluster at L3 under a k=4 cut.

    Paper §6.1 text says "layer 5"; the data places the cluster sharpest at
    L3 (with `Raise` and the `_isolated_` pipeline marker as the 7th and
    8th cluster members). v3 paper text should reflect L3.
    """
    fname = DATA_ROOT / "13_object_masks_eps0.5_cons0.8.h5"
    if not fname.exists():
        pytest.skip(f"frozen artifact not present at {fname}")
    from csp_atlas.analysis import (
        cut_dendrogram_at_k_clusters, pairwise_jaccard_matrix,
        ward_linkage_from_jaccard,
    )
    um = load_universal_masks(eps=0.5, cons=0.8)
    masks_at_l3 = {n: pl[3] for n, pl in um.items() if 3 in pl and pl[3]}
    names, j = pairwise_jaccard_matrix(masks_at_l3)
    linkage = ward_linkage_from_jaccard(j)
    clusters = cut_dendrogram_at_k_clusters(linkage, names, k=4)
    atomicity_cluster_ids = {clusters[c] for c in ATOMICITY_SUPER_CLUSTER if c in clusters}
    assert len(atomicity_cluster_ids) == 1, (
        f"§6.1: atomicity 6-set should form one cluster at L3 (k=4), "
        f"got {len(atomicity_cluster_ids)} cluster ids: {atomicity_cluster_ids}"
    )
