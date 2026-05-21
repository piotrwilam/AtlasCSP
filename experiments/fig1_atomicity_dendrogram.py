"""Paper Figure 1: atomicity super-cluster via Ward linkage at L5.

Loads the 106 universal masks at the chosen (ε, cons) cell, picks one
layer, computes pairwise Jaccard, applies Ward linkage on (1 − Jaccard),
and renders the coloured dendrogram.

Run:
    python experiments/fig1_atomicity_dendrogram.py --config-name paper/figure1_atomicity_dendrogram
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from csp_atlas.analysis import (
    cut_dendrogram_at_k_clusters,
    pairwise_jaccard_matrix,
    ward_linkage_from_jaccard,
)
from csp_atlas.io import load_universal_masks
from csp_atlas.plotting import apply_style, plot_dendrogram


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    apply_style(cfg.style)

    # Load universal masks at (ε, cons), extract the chosen layer's slice.
    full = load_universal_masks(
        eps=cfg.eps, cons=cfg.cons, data_root=Path(cfg.data_root),
    )
    masks_at_layer = {
        name: per_layer[cfg.layer]
        for name, per_layer in full.items()
        if cfg.layer in per_layer and per_layer[cfg.layer]
    }
    assert masks_at_layer, (
        f"no non-empty masks at layer {cfg.layer} in {cfg.eps=}, {cfg.cons=}"
    )

    names, j_matrix = pairwise_jaccard_matrix(masks_at_layer)
    linkage = ward_linkage_from_jaccard(j_matrix)

    # Identify the atomicity super-cluster by cutting at k=6 and reporting
    # which cluster contains the canonical 6-concept set.
    ATOMICITY = {"ast__Assert", "ast__Break", "ast__Continue",
                 "ast__Import", "ast__ImportFrom", "ast__Pass"}
    clusters_at_k4 = cut_dendrogram_at_k_clusters(linkage, names, k=4)
    atomicity_clusters = {clusters_at_k4[c] for c in ATOMICITY if c in clusters_at_k4}

    fig = plot_dendrogram(
        linkage,
        labels=names,
        title=f"Hierarchical clustering — universal concept-only masks at L{cfg.layer}",
        color_threshold=cfg.color_threshold,
        figsize=tuple(cfg.figsize),
    )

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_png = out_dir / f"{cfg.figure.name}.png"
    fig.savefig(out_png)
    fig.savefig(out_png.with_suffix(".pdf"))
    print(f"Saved figure: {out_png}")
    print(f"Atomicity super-cluster {sorted(ATOMICITY)} occupies "
          f"{len(atomicity_clusters)} cluster(s) at k=4 cut")

    with open(out_dir / "resolved_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    with open(out_dir / "run_info.json", "w") as f:
        json.dump(
            {
                "n_concepts": len(names),
                "concept_names": names,
                "layer": cfg.layer,
                "eps": cfg.eps,
                "cons": cfg.cons,
                "atomicity_cluster_ids_at_k4": sorted(atomicity_clusters),
                "figure_path": str(out_png),
            },
            f, indent=2,
        )


if __name__ == "__main__":
    main()
