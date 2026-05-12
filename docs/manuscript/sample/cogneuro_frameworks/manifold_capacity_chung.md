# Manifold Capacity (MFTMA)

**References.**
- Chung, S., Lee, D.D., Sompolinsky, H. (2018). *Classification and Geometry of General Perceptual Manifolds.* **Phys. Rev. X** 8, 031003. https://doi.org/10.1103/PhysRevX.8.031003
- Cohen, U., Chung, S., Lee, D.D., Sompolinsky, H. (2020). *Separability and geometry of object manifolds in deep neural networks.* **Nature Communications** 11, 746. https://doi.org/10.1038/s41467-020-14578-5
- Code: https://github.com/schung039/neural_manifolds_replicaMFT

**One-line idea.** Reduce a population code to three numbers: **manifold capacity** alpha_M (how many object manifolds can be linearly classified per neuron), **manifold radius** R_M (size in the relevant subspace), and **manifold dimension** D_M (effective dimensionality of each manifold). Together they give a geometric account of why a representation does or doesn't support linear readout.

## Original cogneuro / DL use

Chung et al. extended the Gardner / Cover capacity from points to manifolds: given P object manifolds of arbitrary geometry in N-dim space, what is the largest P/N at which they remain linearly separable with random labels? They derived a replica-mean-field (RMF) estimator that decomposes capacity into geometric quantities (R_M, D_M).

Cohen et al. 2020 demonstrated the toolkit on AlexNet, ResNet, and primate IT cortex: across a CNN's layers, alpha_M *increases* monotonically while R_M and D_M *decrease* — formalising "untangling" (DiCarlo & Cox 2007). This is now a standard NeuroAI geometric probe.

## DL analogue on h_t

Define manifolds via a discrete labelling that we want to test for separability. For our setup, the most informative choice is **per-episode manifolds**:

- Manifold m = the cloud of h_t hidden states from episode m (sampled at P=50 timesteps).
- Capacity alpha_M = max P/N s.t. random binary labels of these manifolds are linearly separable.
- Radius R_M / dimension D_M = average geometry of the per-episode point cloud.

Alternative manifold definitions worth running:
- **Goal-bin manifolds** (group by `goal_quadrant` x `goal_distance_bin`): tests how cleanly goal-relative state is encoded.
- **Scene manifolds** (group by `scene_id`): tests scene factorisation.

## Hypothesis for our 5 conditions

Pre-registered predictions on the **magnitude / format** split:

| Quantity | Predicted ordering | Magnitude or format? |
|----------|---------------------|----------------------|
| alpha_M (per-episode) | foveated > uniform > coarse > blind | mainly magnitude (more sensor info -> easier to disentangle episodes) |
| R_M | blind > coarse > foveated (smaller is more compact) | format (geometric compactness of useful subspace) |
| D_M | uniform > foveated > coarse > blind (lower is more focused) | format (dimensions devoted to per-episode features) |

The interesting case: if alpha_M and R_M *anti-correlate* across conditions (foveated has high alpha_M but *also* small R_M), we get a clean magnitude-vs-format dissociation. If they correlate trivially (everything just scales together), the analysis adds little beyond PR.

## Implementation cost

- Use the schung039 repo verbatim. ~80 LOC of glue code on top.
- Add: `scripts/probing/extra/manifold_capacity.py`.
- Runtime: ~30 min per condition for the basic config (M=100 manifolds, P=50 points/manifold). Full HP sweep ~3 h total.
- Total: 3-4 h end-to-end including HP sweep.

## Pseudocode

```python
# pip install git+https://github.com/schung039/neural_manifolds_replicaMFT
from manifold_analysis import manifold_analysis_corr

def per_episode_manifolds(h, ep_id, P=50):
    """Returns list of (D, P) arrays — one per episode, P points each."""
    mfds = []
    for e in np.unique(ep_id):
        H_e = h[ep_id == e]
        if len(H_e) < P: continue
        idx = np.random.choice(len(H_e), P, replace=False)
        mfds.append(H_e[idx].T)  # (D, P)
    return mfds

mfds = per_episode_manifolds(h, ep_id, P=50)
alpha, R, D, _, _ = manifold_analysis_corr(mfds, kappa=0, n_t=200, t_vecs=None)
# Returns: capacity, mean radius, mean dim, plus per-manifold quantities.
```

## What success / failure tells us

- **Success — alpha_M monotone in sensor info, R_M anti-monotone:** beautiful magnitude/format dissociation. Headline figure material.
- **Success — alpha_M flat but R_M ordered:** even cleaner — *capacity is the same, but the geometry of how it's spent differs*. Strongest possible support for the format-axis story.
- **Null — everything trivially scales with PR:** drop it; report PR-only.
- **HP-unstable across (M, P, projection-d):** **drop and walk away**, like DSA. Pre-register the HP sweep.

## Risk

**MEDIUM.** Known sensitivities:

1. **Number of manifolds M.** Capacity estimates can drift when M is too small relative to the subspace dimension. Run M in {50, 100, 200} and only keep results that are within 10% across M.
2. **Points per manifold P.** Geometric quantities depend weakly on P; verify with P in {25, 50, 100}.
3. **Kappa parameter (margin).** Default kappa=0 is fine; do not sweep — that's HP-shopping.
4. **Random-projection initial dim.** schung039 repo handles this internally; trust the default.

**Mitigation against the "DSA fate":** *pre-register* the HP sweep (see `scripts/probing/world_model_probe/PRE_REGISTRATION.md` for template). Decision rule: keep MFTMA only if alpha_M ordering is invariant across all 9 (M, P) combinations. Otherwise drop with a one-paragraph methodology footnote.

## Fit to capacity-allocation backbone

**Sharpens (S) magnitude axis** with potential to **dissociate magnitude from format**. The most theoretically principled tool we'd add — Sompolinsky-style RMF capacity is the textbook mathematical formalisation of "capacity allocation". The whole *language* of the paper (capacity allocation across axes) is more defensible if we measure capacity in the Chung sense, not just PR.

## Why this is risk #3 not risk #1

Even though it's the most theoretically aligned, MFTMA is the highest-variance bet because of HP sensitivity. CCGP and TGM are nearly-guaranteed informative figures. MFTMA is "potentially the headline panel" but with non-trivial probability of being a `risks.md` casualty.
