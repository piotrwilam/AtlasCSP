"""Read mask HDF5 artifacts from the decomposition stage.

The 13_*_masks_eps{ε}_cons{c}.h5 files have four top-level groups:

    metadata/         ast_nodes (43,), builtin_objs (63,)
    metrics/          jaccard_{ast,builtin}_matrix at the reference layer
    pair_masks/       layer_{0..7}/{ast_node}__{builtin}/  dataset (2048,) bool
    universal_masks/  layer_{0..7}/{ast|builtin}__{name}/  dataset (2048,) bool

`universal_106x50.h5` is the standalone universal-masks export — same
content as the embedded `universal_masks/` group but extracted from the
canonical ε=0.5, cons=0.8 cell.

All loaders return Python-native types (`set[int]`, `dict[int, set[int]]`)
so the analysis primitives in `csp_atlas.analysis.*` stay numpy-agnostic
in their interfaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from csp_atlas.paths import DATA_ROOT

PartitionStage = Literal["object", "checker"]
N_LAYERS = 8
MLP_DIM = 2048


def _mask_filename(stage: PartitionStage, eps: float, cons: float) -> str:
    return f"13_{stage}_masks_eps{eps}_cons{cons}.h5"


def load_universal_masks(
    *,
    eps: float = 0.5,
    cons: float = 0.8,
    stage: PartitionStage = "object",
    data_root: Path | None = None,
) -> dict[str, dict[int, set[int]]]:
    """Universal masks per concept per layer.

    Reads `universal_masks/layer_L/{ast|builtin}__{name}` from the
    `13_{stage}_masks_eps{eps}_cons{cons}.h5` artifact.

    Returns
    -------
    `concept_name → layer_id → set of active-neuron indices`.
    Concept name keeps the `{ast|builtin}__` prefix so callers can
    classify (e.g. `"ast__For"` vs `"builtin__len"`).
    """
    root = Path(data_root) if data_root is not None else DATA_ROOT
    path = root / _mask_filename(stage, eps, cons)
    if not path.exists():
        raise FileNotFoundError(f"Expected mask file not found: {path}")

    out: dict[str, dict[int, set[int]]] = {}
    with h5py.File(path, "r") as f:
        um = f["universal_masks"]
        for layer_id in range(N_LAYERS):
            layer_key = f"layer_{layer_id}"
            if layer_key not in um:
                continue
            for concept_name, dataset in um[layer_key].items():
                arr = np.asarray(dataset, dtype=bool)
                ids = set(np.nonzero(arr)[0].tolist())
                out.setdefault(concept_name, {})[layer_id] = ids
    return out


def load_universal_masks_as_arrays(
    *,
    eps: float = 0.5,
    cons: float = 0.8,
    stage: PartitionStage = "object",
    data_root: Path | None = None,
) -> dict[str, dict[int, np.ndarray]]:
    """Same as `load_universal_masks` but returns boolean ndarrays of
    length `MLP_DIM` per (concept, layer) instead of `set[int]`.

    Used when the downstream computation is vectorised (Jaccard
    matrices, dendrograms) and the boolean array is faster than the
    set form. For set-arithmetic use cases (decomposition, group
    membership) prefer `load_universal_masks`.
    """
    root = Path(data_root) if data_root is not None else DATA_ROOT
    path = root / _mask_filename(stage, eps, cons)
    if not path.exists():
        raise FileNotFoundError(f"Expected mask file not found: {path}")

    out: dict[str, dict[int, np.ndarray]] = {}
    with h5py.File(path, "r") as f:
        um = f["universal_masks"]
        for layer_id in range(N_LAYERS):
            layer_key = f"layer_{layer_id}"
            if layer_key not in um:
                continue
            for concept_name, dataset in um[layer_key].items():
                out.setdefault(concept_name, {})[layer_id] = np.asarray(
                    dataset, dtype=bool
                )
    return out


def load_pair_masks(
    *,
    eps: float = 0.5,
    cons: float = 0.8,
    stage: PartitionStage = "object",
    layer: int | None = None,
    data_root: Path | None = None,
) -> dict[tuple[str, str], dict[int, set[int]]]:
    """Per-pair masks: `(ast_node, builtin_obj) → layer_id → neuron-id set`.

    Keys in the HDF5 are formatted `{ast_node}__{builtin}`; this loader
    splits them back into the (ast, builtin) tuple.

    If `layer` is given, only that layer is loaded (faster for single-layer
    analyses); otherwise all 8 layers are returned.
    """
    root = Path(data_root) if data_root is not None else DATA_ROOT
    path = root / _mask_filename(stage, eps, cons)
    if not path.exists():
        raise FileNotFoundError(f"Expected mask file not found: {path}")

    layers = [layer] if layer is not None else list(range(N_LAYERS))
    out: dict[tuple[str, str], dict[int, set[int]]] = {}
    with h5py.File(path, "r") as f:
        pm = f["pair_masks"]
        for L in layers:
            layer_key = f"layer_{L}"
            if layer_key not in pm:
                continue
            for key, dataset in pm[layer_key].items():
                ast_node, builtin = key.split("__", 1)
                arr = np.asarray(dataset, dtype=bool)
                ids = set(np.nonzero(arr)[0].tolist())
                out.setdefault((ast_node, builtin), {})[L] = ids
    return out


def load_concept_inventory(
    *,
    eps: float = 0.5,
    cons: float = 0.8,
    stage: PartitionStage = "object",
    data_root: Path | None = None,
) -> tuple[list[str], list[str]]:
    """Read the (ast_nodes, builtin_objs) lists from the file metadata.

    Returns the canonical 43-element AST list + 63-element builtin list
    that defines the 1,276-pair concept matrix.
    """
    root = Path(data_root) if data_root is not None else DATA_ROOT
    path = root / _mask_filename(stage, eps, cons)
    if not path.exists():
        raise FileNotFoundError(f"Expected mask file not found: {path}")
    with h5py.File(path, "r") as f:
        md = f["metadata"]
        ast = [s.decode() if isinstance(s, bytes) else str(s)
               for s in md["ast_nodes"][:]]
        builtin = [s.decode() if isinstance(s, bytes) else str(s)
                   for s in md["builtin_objs"][:]]
    return list(ast), list(builtin)
