#!/usr/bin/env bash
# Copy the latest run's PDF (and PNG) for every paper figure into the
# paper repo's figures/ directory.
#
# Usage:
#   bash scripts/promote_figures.sh [PAPER_FIGURES_DIR]
#
# Default destination: ../Papers/AtlasCSP/figures

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

PAPER_FIGURES="${1:-../Papers/AtlasCSP/figures}"
mkdir -p "$PAPER_FIGURES"

declare -a MAPPING=(
    "figure1_atomicity_dendrogram:F1_atomicity_dendrogram"
)

manifest="$PAPER_FIGURES/MANIFEST.txt"
{
    echo "# Promoted from $REPO_ROOT to $PAPER_FIGURES"
    echo "# Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
} > "$manifest"

n_promoted=0
for entry in "${MAPPING[@]}"; do
    src_name="${entry%%:*}"
    dst_name="${entry##*:}"
    latest=$(ls -td "results/csp_atlas_${src_name}_"* 2>/dev/null | head -1 || true)
    if [[ -z "$latest" ]]; then
        echo "missing: $src_name (no results/ run found)"
        continue
    fi
    for ext in pdf png; do
        if [[ -f "$latest/${src_name}.${ext}" ]]; then
            cp "$latest/${src_name}.${ext}" "$PAPER_FIGURES/${dst_name}.${ext}"
        fi
    done
    echo "promoted: $src_name → $dst_name"
    echo "$dst_name ← $latest" >> "$manifest"
    n_promoted=$((n_promoted+1))
done

echo
echo "Promoted $n_promoted figures to $PAPER_FIGURES"
echo "Manifest: $manifest"
