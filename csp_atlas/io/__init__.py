"""I/O for CSP-Atlas frozen artifacts.

Submodules:
    h5       — HDF5 mask loaders (universal + pair) at any (ε, cons) cell
    prompts  — Parquet prompt loaders (object, checker, token_checker)
    results  — pre-computed analysis-result CSVs (summary, target_checker,
               modularity, token_independence)
    hf       — Hugging Face Hub fallback (optional, needs huggingface_hub)

Defaults read from `CSP_ATLAS_DATA_ROOT` env var via `csp_atlas.paths.DATA_ROOT`.
"""

from csp_atlas.io.h5 import (
    N_LAYERS,
    MLP_DIM,
    load_concept_inventory,
    load_pair_masks,
    load_universal_masks,
    load_universal_masks_as_arrays,
)
from csp_atlas.io.prompts import load_prompts
from csp_atlas.io.results import (
    load_modularity_scores,
    load_summary,
    load_target_checker,
    load_token_independence,
)

__all__ = [
    "N_LAYERS",
    "MLP_DIM",
    "load_concept_inventory",
    "load_modularity_scores",
    "load_pair_masks",
    "load_prompts",
    "load_summary",
    "load_target_checker",
    "load_token_independence",
    "load_universal_masks",
    "load_universal_masks_as_arrays",
]
