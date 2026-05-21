"""Regenerate `relaxed_modularity_scores.csv` + `_detail.csv` outside the notebook.

The original `notebooks/2_analysis/4B_relaxed_modularity.ipynb` expects a
`universal_106x50.h5` file that was never saved at the published 106×50 scale.
The universal masks DO exist — embedded inside
`13_object_masks_eps0.5_cons0.8.h5` under the `universal_masks/` HDF5 group.

This script:
  1. Loads the 106 universal masks (43 ast + 63 builtin) from that file
  2. Reproduces the §6 relaxed-modularity scoring (Cells 6–10 of the notebook)
  3. Writes both CSVs to DATA_ROOT

Identical compute to the notebook — same defaults (P_VALUES, significance),
same tie-breaking. Output bytes should match a notebook re-run.

Usage:
    python scripts/regenerate_modularity_scores.py [--data-root PATH]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SIGNIFICANCE = 0.05
P_VALUES = [0.0, 0.05, 0.10, 0.25]
N_LAYERS = 8
SOURCE_FILE = "13_object_masks_eps0.5_cons0.8.h5"


def jaccard_similarity(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def trimmed_mean_jaccard(jaccard_row: np.ndarray, self_idx: int, p: float) -> float:
    """Mean Jaccard after dropping the top-p fraction of nearest neighbours."""
    others = np.delete(jaccard_row, self_idx)
    n_others = len(others)
    k = int(np.floor(p * n_others))
    trimmed = np.sort(others)[::-1][k:] if k > 0 else others
    if len(trimmed) == 0:
        return 0.0
    return float(trimmed.mean())


def main(data_root: Path) -> None:
    h5_path = data_root / SOURCE_FILE
    assert h5_path.exists(), f"source file not found: {h5_path}"
    print(f"Loading: {h5_path}")

    # Step 1 — extract universal masks.
    masks: dict[str, dict[int, np.ndarray]] = {}
    obj_types: dict[str, str] = {}
    with h5py.File(h5_path, "r") as f:
        um = f["universal_masks"]
        for layer_id in range(N_LAYERS):
            layer_key = f"layer_{layer_id}"
            if layer_key not in um:
                continue
            for ds_name in um[layer_key]:
                obj_type = ds_name.split("__", 1)[0] if "__" in ds_name else ds_name
                masks.setdefault(ds_name, {})[layer_id] = np.array(
                    um[layer_key][ds_name], dtype=bool
                )
                obj_types[ds_name] = obj_type

    all_objects = sorted(masks.keys())
    n_objects = len(all_objects)
    n_ast = sum(1 for v in obj_types.values() if v == "ast")
    n_builtin = sum(1 for v in obj_types.values() if v == "builtin")
    print(f"Loaded {n_objects} universal circuits ({n_ast} ast + {n_builtin} builtin)")

    for obj in all_objects:
        assert len(masks[obj]) == N_LAYERS, (
            f"{obj} has {len(masks[obj])} layers, expected {N_LAYERS}"
        )

    # Step 2 — precompute Jaccard matrices per layer.
    jaccard_matrices: dict[int, np.ndarray] = {}
    for layer_id in range(N_LAYERS):
        mat = np.zeros((n_objects, n_objects), dtype=np.float32)
        for i in range(n_objects):
            for j in range(i + 1, n_objects):
                sim = jaccard_similarity(masks[all_objects[i]][layer_id],
                                         masks[all_objects[j]][layer_id])
                mat[i, j] = sim
                mat[j, i] = sim
            mat[i, i] = 1.0
        jaccard_matrices[layer_id] = mat
        print(f"  layer {layer_id}: Jaccard matrix done")

    # Step 3 — score each (object, layer) under each P value; aggregate.
    n_p = len(P_VALUES)
    significant_counts = np.zeros((n_objects, n_p), dtype=int)
    detail_records: list[dict] = []

    for layer_id in range(N_LAYERS):
        # Precompute trimmed means for ALL objects at this layer, for each p.
        all_trimmed_means: dict[float, np.ndarray] = {}
        for p_val in P_VALUES:
            means = np.array(
                [trimmed_mean_jaccard(jaccard_matrices[layer_id][i], i, p_val)
                 for i in range(n_objects)],
                dtype=np.float32,
            )
            all_trimmed_means[p_val] = means

        for obj_idx in range(n_objects):
            obj_name = all_objects[obj_idx]
            for p_col, p_val in enumerate(P_VALUES):
                observed = all_trimmed_means[p_val][obj_idx]
                percentile = float(
                    (all_trimmed_means[p_val] <= observed).sum() / n_objects
                )
                is_sig = percentile < SIGNIFICANCE
                if is_sig:
                    significant_counts[obj_idx, p_col] += 1
                detail_records.append({
                    "object": obj_name,
                    "type": obj_types[obj_name],
                    "layer": layer_id,
                    "p_trim": p_val,
                    "observed_trimmed_jaccard": round(observed, 4),
                    "percentile": round(percentile, 4),
                    "significant": is_sig,
                })

    # Step 4 — build results table.
    results_df = pd.DataFrame(
        significant_counts,
        index=all_objects,
        columns=[f"p={p}" for p in P_VALUES],
    )
    results_df.insert(0, "type", [obj_types[obj] for obj in all_objects])
    results_df = results_df.sort_values(
        [f"p={P_VALUES[0]}", f"p={P_VALUES[-1]}"],
        ascending=[False, False],
    )

    out_main = data_root / "relaxed_modularity_scores.csv"
    out_detail = data_root / "relaxed_modularity_detail.csv"
    results_df.to_csv(out_main)
    pd.DataFrame(detail_records).to_csv(out_detail, index=False)
    print(f"\nSaved: {out_main}")
    print(f"Saved: {out_detail}")
    print(f"\nTop 10 most-modular objects by p=0 score:")
    print(results_df.head(10).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=os.environ.get("CSP_ATLAS_DATA_ROOT",
                               Path.home() / "Data" / "CSP-Atlas"),
        type=Path,
        help="Directory containing 13_object_masks_eps0.5_cons0.8.h5",
    )
    args = parser.parse_args()
    main(args.data_root)
