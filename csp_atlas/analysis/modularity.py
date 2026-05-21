"""Relaxed-modularity score — paper §6.1.

A concept's relaxed-modularity at a given layer asks: under various
permissiveness levels `p ∈ {0.0, 0.05, 0.10, 0.25}` (drop-the-top-p
nearest neighbours), does its mean Jaccard to the rest of the population
sit in the bottom-SIGNIFICANCE fraction of the population?

Significant layers per concept aggregate into a 4-column score table
(one column per p). Top concepts by this score are the atomicity
super-cluster: Break, Assert, ImportFrom, Import, Continue, Pass.

Defaults match `scripts/regenerate_modularity_scores.py` and the
notebook `4B_relaxed_modularity.ipynb`. Don't change without
updating the locked CSVs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_P_VALUES: tuple[float, ...] = (0.0, 0.05, 0.10, 0.25)
DEFAULT_SIGNIFICANCE = 0.05


def trimmed_mean_jaccard(jaccard_row: np.ndarray, self_idx: int, p: float) -> float:
    """Mean Jaccard after dropping the top-`p` fraction of nearest neighbours.

    `p = 0` is the strict criterion (no trimming). Increasing `p` relaxes
    the threshold — required because some atomicity concepts share heavily
    with one or two close partners.
    """
    others = np.delete(jaccard_row, self_idx)
    n_others = len(others)
    k = int(np.floor(p * n_others))
    trimmed = np.sort(others)[::-1][k:] if k > 0 else others
    if len(trimmed) == 0:
        return 0.0
    return float(trimmed.mean())


def compute_modularity_scores(
    jaccard_matrices: dict[int, np.ndarray],
    concept_names: list[str],
    concept_types: dict[str, str],
    *,
    p_values: tuple[float, ...] = DEFAULT_P_VALUES,
    significance: float = DEFAULT_SIGNIFICANCE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce §6.1 relaxed-modularity scoring.

    Parameters
    ----------
    jaccard_matrices  {layer_id: (n, n) Jaccard matrix}
    concept_names     length-n list, same order as the matrix axes
    concept_types     {concept_name: "ast" | "builtin"}
    p_values          relaxation levels (default = paper's set)
    significance      bottom-fraction threshold (default 0.05)

    Returns
    -------
    `(scores, detail)` — scores has one row per concept with the
    significant-layer count under each `p`; detail has one row per
    `(object, layer, p_trim)` with the observed trimmed mean, percentile,
    and significance flag.
    """
    assert all(m.shape == (len(concept_names), len(concept_names))
               for m in jaccard_matrices.values()), (
        "every Jaccard matrix must be n × n where n = len(concept_names)"
    )

    n_objects = len(concept_names)
    n_p = len(p_values)
    significant_counts = np.zeros((n_objects, n_p), dtype=int)
    detail_records: list[dict] = []

    for layer_id, jmat in jaccard_matrices.items():
        # Trimmed means for every object at every p_value at this layer.
        all_trimmed = {
            p: np.array(
                [trimmed_mean_jaccard(jmat[i], i, p) for i in range(n_objects)],
                dtype=np.float32,
            )
            for p in p_values
        }
        for obj_idx in range(n_objects):
            obj_name = concept_names[obj_idx]
            for p_col, p_val in enumerate(p_values):
                observed = all_trimmed[p_val][obj_idx]
                percentile = float(
                    (all_trimmed[p_val] <= observed).sum() / n_objects
                )
                is_sig = percentile < significance
                if is_sig:
                    significant_counts[obj_idx, p_col] += 1
                detail_records.append({
                    "object": obj_name,
                    "type": concept_types[obj_name],
                    "layer": layer_id,
                    "p_trim": p_val,
                    "observed_trimmed_jaccard": round(observed, 4),
                    "percentile": round(percentile, 4),
                    "significant": is_sig,
                })

    scores_df = pd.DataFrame(
        significant_counts,
        index=concept_names,
        columns=[f"p={p}" for p in p_values],
    )
    scores_df.insert(0, "type", [concept_types[n] for n in concept_names])
    scores_df = scores_df.sort_values(
        [f"p={p_values[0]}", f"p={p_values[-1]}"],
        ascending=[False, False],
    )
    return scores_df, pd.DataFrame(detail_records)
