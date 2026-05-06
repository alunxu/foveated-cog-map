# Persistent homology of the cognitive-map manifold

## Reference paper

Carlsson, G. (2009). *Topology and data.* **Bull. Amer. Math. Soc.** 46, 255-308.

Cogneuro replications:
- Curto & Itskov (2008) *PLoS Comput Biol* — place-cell topology recovers spatial layout.
- Singh et al. (2008) *J. Vis.* — persistent homology of natural images.
- Rieck et al. (2019) *ICLR* — topology in deep nets.
- Stolz et al. (2017) *J Comput Neurosci* — TDA on hippocampal data.
- Gardner, Hermansen, Pachitariu, Burak, Baas, Dunn, Moser, Moser (2022) **Nature** — toroidal topology of grid-cell module via persistent homology. **This is the cleanest cogneuro precedent.**

## Original cogneuro use

Sample population activity at many timepoints, compute pairwise distances, build a Vietoris–Rips filtration, and read off **persistent Betti numbers**. Betti-0 = connected components. Betti-1 = independent loops. Betti-2 = voids. The Gardner et al. 2022 result: grid-cell activity in mouse MEC lives on a **torus** (Betti-1 = 2, Betti-2 = 1 persistent), even when the animal explores a 2D box — a striking topological invariant.

The promise: **the topology of the cognitive-map manifold is a near-invariant of the spatial code**, robust to scaling/rotation, that distinguishes "pure place code" from "trajectory code" from "context-mixed code".

## DL analogue (our setting)

For each condition, sample N=2000 hidden states `h_t` (subsampled to roughly cover the agent_pos extent). Compute a Vietoris–Rips persistence diagram on cosine distance with `ripser` or `gudhi`. Compare:

1. **Persistent Betti-0** (number of well-separated clusters): does the manifold have multiple components per condition?
2. **Persistent Betti-1** (loops): are there persistent 1D loops — e.g. a circular "head-direction" or "trajectory" loop?
3. **Bottleneck / Wasserstein distance** between persistence diagrams across conditions: are blind and foveated topologically distinguishable?

## Hypothesis

1. **Blind and coarse** agents will show a **lower-dimensional, more connected** manifold (small Betti-0 after burn-in, low Betti-1) — consistent with a near-1D path-integration code along the integrated-position direction.
2. **Foveated and uniform** agents will show **higher Betti-1**: more persistent loops, reflecting the encoder injecting room-context structure into `h_t` and creating context-conditioned cycles.
3. **Wasserstein distance** between the persistence diagrams of blind vs uniform should be **greater** than the within-condition seed-to-seed distance (sanity: across-condition gap > within-condition noise). This is the comparative-cogneuro test.

## Pseudocode

```python
import numpy as np
from ripser import ripser
from persim import wasserstein, plot_diagrams

def persistence(H, n_subsample=2000, max_dim=2):
    idx = np.random.choice(len(H), n_subsample, replace=False)
    X = H[idx]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)  # cosine via Euclidean
    pd = ripser(X, maxdim=max_dim)['dgms']
    return pd

# Per condition:
diagrams = {}
for cond in conditions:
    H = load_cache(cond)['h_t']
    diagrams[cond] = [persistence(H) for _ in range(5)]  # 5 subsample seeds

# Cross-condition Wasserstein
for c1, c2 in combinations(conditions, 2):
    w = np.mean([wasserstein(diagrams[c1][i][1], diagrams[c2][i][1]) for i in range(5)])

# Persistent Betti counts (per condition, averaged over seeds)
def persistent_count(pd, dim, persistence_thresh=0.1):
    return np.sum((pd[dim][:, 1] - pd[dim][:, 0]) > persistence_thresh)
```

## Runtime estimate

- Ripser on N=2000 in 512-d cosine: **~5 min/sample × 5 seeds × 5 conditions ≈ 2 hours**.
- Wasserstein distances + plotting: 30 min.
- Total: **~3 hours** (CPU).

If 2000 is too slow, downsample to 1000 with PCA → 32-d projection (validate: PCA-32 preserves the persistence structure on a held-out condition before scaling).

## Success criteria

**Strong signal**: across-condition Wasserstein > 2× within-condition seed-noise; Betti-1 monotonically increasing in encoder bandwidth; topology survives 5 random-seed subsamples and a PCA-32 robustness check.

**Modest signal**: detectable but small Wasserstein gap; Betti structure dominated by Betti-0 (clustering rather than loops). → supplementary figure with honest framing "topological signature is dominated by component-count differences, not higher-order structure."

**Failure mode**: persistence diagrams indistinguishable across conditions. The Gardner et al. precedent suggests this is unlikely if a real geometric structure exists, but TDA is famously HP-sensitive (subsample density, max-dim, threshold). Mitigation: lock all hyperparameters in pre-reg; report bottleneck distance instead of relying on Betti numbers, since bottleneck is more robust.

## Fit to backbone

The format axis already shows that linear-readability of position differs across conditions. PH adds: **the manifold's topology** also differs — and topology is invariant to linear transformations, so it's a parameter-free check that format differences are not artefactual to the choice of decoder.

PH is also the cleanest "comparative cognitive-neuroscience" test in the battery: it ports a *literal cell-Nature precedent* (Gardner 2022, "the toroidal grid-cell manifold") to artificial agents.

## Pre-registered prediction

> Persistence diagrams computed via Vietoris–Rips on N=2000 subsampled `h_t` per condition (cosine distance, max-dim=2, 5 seeds per condition, persistence threshold 0.1) will satisfy: (a) the median Wasserstein-2 distance between {blind, foveated_uniform} pairs exceeds 2× the median within-condition seed-to-seed Wasserstein; (b) persistent Betti-1 will be lower for {blind, coarse} than for {foveated, uniform, foveated_logpolar}; (c) bottleneck distance (the most robust scalar) will rank-order conditions matching the encoder-bandwidth axis.

If (a) + (c) hold, this is a **topology-level confirmation of capacity allocation** — and arguably the single strongest "comparative cogneuro" claim the paper can make, because it directly invokes Gardner et al. 2022 as cogneuro precedent.
