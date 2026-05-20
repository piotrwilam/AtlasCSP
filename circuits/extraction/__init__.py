"""Module 2 — Activation extraction, binarisation, marginalisation.

Pipeline: forward hooks → (ε, consistency) thresholds → intersect across
complementary objects → universal mask per concept.

Submodules:
    extraction        — `ActivationExtractor` (needs torch)
    binarization      — `PairRepresentationBuilder`, `RawActivationCollector`
    marginalization   — `UniversalModuleComputer`
    metrics           — array-based Jaccard / entanglement-index utilities
    pipeline          — `Module2Pipeline` end-to-end orchestrator (needs torch)
    io_utils          — HDF5 readers / writers

Module-level re-exports are omitted because `extraction` and `pipeline` pull
in torch. Import submodules directly:

    from circuits.extraction.extraction import ActivationExtractor   # needs torch
    from circuits.extraction.marginalization import UniversalModuleComputer
    from circuits.extraction.io_utils import save_atlas_hdf5
"""
