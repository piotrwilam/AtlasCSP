# Layer 2 — Analysis notebooks

These are the **exploratory analysis** notebooks used while writing the v2
paper. They read the frozen artifacts produced by Layer 1 and compute
statistics, run validations, generate intermediate plots.

For the **canonical, locked-in versions of the paper figures**, use the
scripts in [`../../experiments/`](../../experiments/) once they exist (this
is the next phase of the refactor — see the project plan). The notebooks here
are kept for context (they're what the original analysis looked like) and for
any re-analysis that doesn't fit the paper's published figures.

## Notebook map

| Notebook | Paper section | What it explores |
|---|---|---|
| `3A_evaluation.ipynb` | §6 / §7.2 | Atlas-level evaluation: per-pair circuit sizes, Jaccard heatmaps, modularity metrics. |
| `4_modularity_scores.ipynb` | §6.1 (Finding 2) | Per-concept modularity score — concept-only vs token-driven decomposition strength. |
| `4B_relaxed_modularity.ipynb` | §6.1 | Modularity under relaxed thresholds. Sensitivity probe. |
| `5B_token_independence.ipynb` | §6 (Finding 2) | Token-independence check: do AST circuits survive ablating the bare keyword token? |
| `14_comparison.ipynb` | §7.1 (Finding 1) | Parameter-stability across the 3×3 (ε, consistency) sweep. |

## A note on outputs

Most notebooks have outputs stripped (nbstripout configured in `.gitattributes`).
A few may carry embedded outputs from runs prior to the strip hook — these are
historical records, not authoritative.
