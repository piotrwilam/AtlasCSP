import logging
import pickle

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def save_atlas_hdf5(path: str, pair_masks: dict, universal_masks: dict,
                    metrics: dict, metadata: dict):
    """Save the full atlas to HDF5 with the schema from the spec."""
    with h5py.File(path, "w") as f:
        # Pair masks: /pair_masks/layer_{lid}/{ast_n}__{blt_o}
        pg = f.create_group("pair_masks")
        for (ast_n, blt_o), layers in pair_masks.items():
            for lid, mask in layers.items():
                pg.create_dataset(
                    f"layer_{lid}/{ast_n}__{blt_o}",
                    data=mask.astype(np.bool_),
                    compression="gzip"
                )

        # Universal masks: /universal_masks/layer_{lid}/ast__{name} or builtin__{name}
        ug = f.create_group("universal_masks")
        for name, layers in universal_masks.get("ast", {}).items():
            for lid, mask in layers.items():
                ug.create_dataset(
                    f"layer_{lid}/ast__{name}",
                    data=mask.astype(np.bool_),
                    compression="gzip"
                )
        for name, layers in universal_masks.get("builtin", {}).items():
            for lid, mask in layers.items():
                ug.create_dataset(
                    f"layer_{lid}/builtin__{name}",
                    data=mask.astype(np.bool_),
                    compression="gzip"
                )

        # Metrics
        mg = f.create_group("metrics")
        if "jaccard_ast_matrix" in metrics:
            mg.create_dataset("jaccard_ast_matrix", data=metrics["jaccard_ast_matrix"])
        if "jaccard_builtin_matrix" in metrics:
            mg.create_dataset("jaccard_builtin_matrix", data=metrics["jaccard_builtin_matrix"])

        # Metadata
        md = f.create_group("metadata")
        for key, val in metadata.items():
            if isinstance(val, list):
                md.create_dataset(key, data=np.array(val, dtype=h5py.special_dtype(vlen=str)))
            else:
                md.attrs[key] = val

    logger.info(f"Atlas saved: {path}")


def load_atlas_hdf5(path: str) -> dict:
    """Load atlas from HDF5, return all components."""
    atlas = h5py.File(path, "r")
    meta = dict(atlas["metadata"].attrs)

    pair_masks = {}
    if "pair_masks" in atlas:
        for layer_key in atlas["pair_masks"]:
            lid = int(layer_key.split("_")[1])
            for name in atlas[f"pair_masks/{layer_key}"]:
                parts = name.split("__", 1)
                if len(parts) == 2:
                    ast_n, blt_o = parts
                    key = (ast_n, blt_o)
                    if key not in pair_masks:
                        pair_masks[key] = {}
                    pair_masks[key][lid] = atlas[f"pair_masks/{layer_key}/{name}"][:]

    universal_masks = {"ast": {}, "builtin": {}}
    if "universal_masks" in atlas:
        for layer_key in atlas["universal_masks"]:
            lid = int(layer_key.split("_")[1])
            for name in atlas[f"universal_masks/{layer_key}"]:
                mask = atlas[f"universal_masks/{layer_key}/{name}"][:]
                if name.startswith("ast__"):
                    concept = name[5:]
                    if concept not in universal_masks["ast"]:
                        universal_masks["ast"][concept] = {}
                    universal_masks["ast"][concept][lid] = mask
                elif name.startswith("builtin__"):
                    concept = name[9:]
                    if concept not in universal_masks["builtin"]:
                        universal_masks["builtin"][concept] = {}
                    universal_masks["builtin"][concept][lid] = mask

    metrics = {}
    if "metrics" in atlas:
        for key in atlas["metrics"]:
            metrics[key] = atlas["metrics"][key][:]

    return {
        "pair_masks": pair_masks,
        "universal_masks": universal_masks,
        "metrics": metrics,
        "metadata": meta,
        "handle": atlas,  # caller must close atlas["handle"] when done
    }


def save_checkpoint(path: str, pair_masks: dict, pairs_processed: int):
    """Incremental checkpoint for long runs using pickle."""
    with open(path, "wb") as f:
        pickle.dump({"pair_masks": pair_masks, "pairs_processed": pairs_processed}, f)
    logger.debug(f"Checkpoint saved: {path}")


def load_checkpoint(path: str) -> tuple:
    """Resume from checkpoint. Returns (pair_masks, pairs_processed)."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info(f"Loaded checkpoint: {data['pairs_processed']} pairs")
    return data["pair_masks"], data["pairs_processed"]
