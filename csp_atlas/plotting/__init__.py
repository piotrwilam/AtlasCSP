"""Plotting: style + per-figure-type rendering functions.

Plotting functions take data and return a `matplotlib.figure.Figure`.
They never read or write files; that's the experiment script's job.
"""

from csp_atlas.plotting.dendrogram import plot_dendrogram
from csp_atlas.plotting.style import apply_style

__all__ = ["apply_style", "plot_dendrogram"]
