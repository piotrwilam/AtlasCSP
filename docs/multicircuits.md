# Multicircuit Analysis

## Definitions

- **Object**: an AST node (e.g., `For`, `ClassDef`) or a builtin (e.g., `len`, `ValueError`)
- **Universal object**: the universal representation of an object — the set of neurons that consistently activate for that object across all prompt variations and pairing contexts
- **Multicircuit**: a union of universal representations of several objects that share neural substrate
- **Strong multicircuit**: the constituent circuits clearly coincide — high pairwise similarity across multiple layers
- **Weak multicircuit**: the constituent circuits hardly coincide — low pairwise similarity, or similarity limited to one or two layers

Multicircuits may be weak on some layers and strong on others. A multicircuit that is strong at layers 5–7 but weak at layers 0–2 suggests that the shared processing emerges in the later stages of the network.

## Objective

Prove or disprove that the following four groups, identified from hierarchical clustering at Layer 5 (EPSILON=0.5, CONSISTENCY=0.80), are strong multicircuits:

1. **Generic AST** (19 circuits): AsyncWith, Attribute, ClassDef, Delete, Dict, GeneratorExp, Lambda, Return, SetComp, Slice, Subscript, YieldFrom, classmethod, isinstance, property, repr, staticmethod, super, zip

2. **Exceptions** (20 circuits): Continue, DictComp, ListComp, With, Yield + AttributeError, Exception, FileNotFoundError, IndexError, KeyError, MemoryError, NameError, OSError, OverflowError, RecursionError, RuntimeError, StopIteration, TypeError, ValueError, ZeroDivisionError

3. **Raise + Error hierarchy** (6 circuits): While, Raise, ArithmeticError, ImportError, LookupError, NotImplementedError

4. **Conditionals + Loops** (13 circuits): For, AsyncFor, Global, ImportFrom, Nonlocal, Starred, ExceptHandler, Try, If, IfExp, FunctionDef, AsyncFunctionDef, Break

## General Setup

**Model**: openai/circuit-sparsity — 8-layer sparse transformer (circuitgpt), 2048 MLP neurons per layer.

**Data**: 1370 (AST node, builtin) pairs, 100 prompt variations each. Raw activations extracted via Module 2A. Universal objects constructed via Module 2B with configurable thresholds.

**Similarity measure**: Jaccard distance on binary masks of selective neurons (neurons that are neither dark nor always-on). Hierarchical clustering with average linkage.

**What "strong" means** (refined by experiments):
- The group's mean intra-cluster Jaccard is significantly higher than random (permutation test, p < 0.05)
- This holds across multiple layers, not just one
- This holds across a range of EPSILON values, not just one threshold
- The group is more cohesive internally than it is similar to outsiders

## Experiment Design

### Stage 1 — Layer stability (notebook 4A)

For each of the four candidate groups, compute intra-group mean Jaccard at every layer (0–7).

**Pass criterion**: The group shows significantly elevated intra-group similarity at 3 or more layers.

### Stage 2 — Genuineness (notebook 4B)

For each group that survives Stage 1, run a permutation test (1000 iterations) at every layer: randomly sample groups of the same size, compare intra-group Jaccard.

**Pass criterion**: Permutation p-value < 0.05 at the layers identified in Stage 1.

### Stage 3 — Epsilon robustness

Stages 1 and 2 were run at three epsilon values (0.05, 0.1, 0.5) to test robustness. A strong multicircuit should pass across thresholds.

**Pass criterion**: The group passes the permutation test at 2+ epsilon values.

## Experiment Results

### Stage 1 — Layer Stability

All four groups pass Stage 1 at all three epsilon values — elevated intra-group Jaccard (> 0.1) at all 8 layers. This is a necessary but not sufficient condition.

**Mean intra-group Jaccard (ε=0.1, medium resolution):**

| Group | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | Best |
|-------|------|------|------|------|------|------|------|------|------|
| A: Generic AST | 0.59 | 0.33 | 0.55 | 0.54 | **0.66** | 0.51 | 0.49 | 0.53 | L4 |
| B: Exceptions | 0.58 | 0.54 | 0.72 | 0.69 | **0.76** | 0.59 | 0.55 | 0.62 | L4 |
| C: Raise+Errors | 0.62 | **0.78** | **0.78** | 0.59 | 0.62 | 0.48 | 0.57 | 0.66 | L2 |
| D: Control flow | 0.41 | 0.41 | 0.48 | 0.47 | 0.56 | 0.38 | 0.42 | 0.49 | L4 |

**Selective neurons per layer (ε=0.1):** L0:19, L1:13, L2:88, L3:50, L4:30, L5:65, L6:236, L7:332

### Stage 2 — Genuineness (Permutation Tests)

**Significant layers (p < 0.05) per group across epsilon values:**

| Group | ε=0.5 (strong) | ε=0.1 (medium) | ε=0.05 |
|-------|---------------|----------------|--------|
| **B: Exceptions** | 3/8 (L5,6,7) | **8/8 (all)** | **8/8 (all)** |
| **C: Raise+Errors** | 3/8 (L0,5,6) | 3/8 (L1,2,7) | 5/8 (L2,4,5,6,7) |
| **A: Generic AST** | 1/8 (L5) | 2/8 (L0,5) | 6/8 (L0,1,2,4,5,6) |
| **D: Control flow** | 2/8 (L5,6) | **0/8 FAIL** | **0/8 FAIL** |

### Stage 3 — Cross-Epsilon Summary

| Group | ε=0.5 | ε=0.1 | ε=0.05 | Verdict |
|-------|-------|-------|--------|---------|
| **B: Exceptions** | PASS (3 layers) | PASS (8 layers) | PASS (8 layers) | **STRONG MULTICIRCUIT** |
| **C: Raise+Errors** | PASS (3 layers) | PASS (3 layers) | PASS (5 layers) | **GENUINE MULTICIRCUIT** |
| **A: Generic AST** | PASS (1 layer) | PASS (2 layers) | PASS (6 layers) | **WEAK MULTICIRCUIT** |
| **D: Control flow** | PASS (2 layers) | FAIL | FAIL | **NOT A MULTICIRCUIT** |

## Findings

### B: Exceptions — Strong Multicircuit

The strongest result. All 15 exception builtins (ValueError, TypeError, KeyError, etc.) plus AST nodes Continue, DictComp, ListComp, With, and Yield share neural substrate **at every layer of the network** at ε=0.1 and ε=0.05. The model processes all exception types through a common circuit.

This is functionally coherent: exception handling is syntactically uniform (`try/except`) regardless of exception type, so the model can reuse the same neural pathway. The inclusion of AST nodes like Continue and ListComp suggests these constructs share some processing infrastructure with exception handling — possibly related to flow control or scope management.

Best Jaccard: 0.76 at Layer 4 (ε=0.1), 1.0 at Layer 5 (ε=0.5).

### C: Raise+Errors — Genuine Multicircuit

While, Raise, and four abstract exception types (ArithmeticError, ImportError, LookupError, NotImplementedError) form a genuine multicircuit. Significant at 3–5 layers across all epsilon values, but at **different layers** depending on threshold:
- ε=0.5: layers 0, 5, 6 (strong signal)
- ε=0.1: layers 1, 2, 7 (medium signal)
- ε=0.05: layers 2, 4, 5, 6, 7 (weak but widespread signal)

This suggests the group has both strong-firing shared neurons (visible at high epsilon) and weaker consistent neurons (visible at low epsilon), distributed across different layers.

### A: Generic AST — Weak Multicircuit

19 circuits that appeared identical at Layer 5 (ε=0.5). At lower epsilon values, the group does share neurons significantly, but weakly — the shared signal only emerges when the activation threshold is low enough to include subtle activations.

At ε=0.05: significant at 6/8 layers. At ε=0.1: only 2/8. At ε=0.5: only 1/8.

Interpretation: these circuits share a common low-amplitude "scaffolding" signal — neurons that fire weakly for any structured Python construct. This is a real shared substrate but not a strong functional module. The group is semantically incoherent (ClassDef, Lambda, zip, isinstance have nothing in common functionally).

### D: Control flow — Not a Multicircuit

13 AST nodes (For, If, Try, FunctionDef, Break, etc.) that appeared to cluster at ε=0.5. At ε=0.1 and ε=0.05, the group is **not more similar than random** at any layer (0 significant layers at both thresholds).

The apparent clustering at ε=0.5 was an artifact of sparse signal — with only 5 selective neurons at Layer 5, many circuits collapsed into identical masks by chance. When more neurons are available at lower epsilon, these circuits differentiate and the group dissolves.

This group should be investigated for genuine subgroups via leave-one-out analysis (notebook 4B, cells 9–10).

## Verdict Criteria

| Result | Interpretation |
|--------|---------------|
| Passes all 3 stages | **Strong multicircuit** — stable, genuine, robust |
| Passes stages 1+2, fails 3 | **Threshold-dependent multicircuit** — real but only at specific resolution |
| Passes stage 1, fails 2 | **Apparent multicircuit** — cohesive but not more than random, likely artifact |
| Fails stage 1 | **Not a multicircuit** — grouping is a single-layer artifact |

## Notebooks

- `4A_layer_stability.ipynb` — Stage 1: intra-group Jaccard across layers
- `4B_genuineness.ipynb` — Stage 2: permutation tests + leave-one-out refinement
- Reports saved to `Data/CSP-Atlas/report_4{A,B}_{dataset}_{date}.txt`
