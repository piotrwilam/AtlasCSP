"""Ward-linkage dendrogram rendering — paper Figure 1.

Used for the §6 atomicity-super-cluster figure at L5 (Qwen × Python).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram as _dendrogram


def plot_dendrogram(
    linkage: np.ndarray,
    labels: list[str],
    *,
    title: str = "",
    color_threshold: float | None = None,
    figsize: tuple[float, float] = (14, 5),
    leaf_font_size: int = 8,
    ax: Axes | None = None,
) -> Figure:
    """Render a hierarchical-clustering dendrogram.

    Parameters
    ----------
    linkage           `(n-1, 4)` scipy linkage matrix.
    labels            length-n list of leaf labels.
    title             axes title; "" to omit.
    color_threshold   branch-colour distance threshold; None → scipy
                      default (0.7 × max distance).
    figsize           used only when `ax` is None.
    ax                pre-existing axes to plot into. If None a new
                      Figure + Axes is created.

    Returns the Figure.
    """
    assert linkage.ndim == 2 and linkage.shape[1] == 4, (
        f"linkage must be (n-1, 4), got shape {linkage.shape}"
    )
    assert linkage.shape[0] == len(labels) - 1, (
        f"linkage has {linkage.shape[0]} rows but labels has {len(labels)} entries"
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        owns_fig = True
    else:
        fig = ax.figure
        owns_fig = False

    _dendrogram(
        linkage,
        labels=labels,
        color_threshold=color_threshold,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=leaf_font_size,
    )
    if title:
        ax.set_title(title)
    ax.set_xlabel("")
    ax.grid(False)
    if owns_fig:
        fig.tight_layout()
    return fig
