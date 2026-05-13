# Representational Drift across Episodes

**References.**
- Driscoll, L.N., Pettit, N.L., Minderer, M., Chettih, S.N., Harvey, C.D. (2017). *Dynamic reorganization of neuronal activity patterns in parietal cortex.* **Cell** 170(5), 986-999. https://doi.org/10.1016/j.cell.2017.07.021
- Mau, W., Hasselmo, M.E., Cai, D.J. (2020). *The brain in motion: How ensemble fluidity drives memory-updating and flexibility.* **eLife** 9, e63550.
- Rule, M.E., O'Leary, T., Harvey, C.D. (2019). *Causes and consequences of representational drift.* **Curr. Opin. Neurobiol.** 58, 141-147. https://doi.org/10.1016/j.conb.2019.08.005
- Rule, M.E. et al. (2020). *Stable task information from an unstable neural population.* **eLife** 9, e51121.

**One-line idea.** Even when behaviour is stable, the population code can rotate / reassign units across days — the *task*-relevant subspace is preserved while *unit-level* tuning drifts. Quantify by tracking a probe basis (or a subspace) across blocks of trials.

## Original cogneuro use

Driscoll et al. found that across days, parietal cortex single-cell tuning drifted substantially while task performance was stable; the same task variable could be decoded with high accuracy on each day, but the cells *carrying* the information changed. Rule et al. formalised this as *drift in unit assignment + stability in subspace*. Multiple subsequent papers (Rubin, Geva, Ziv 2015; Mau 2018; Rule 2020) confirmed across hippocampus, motor cortex, cortex.

## DL analogue on h_t

In our setup we don't have multi-day recordings — the agent's weights are frozen. But we *do* have **episode blocks** within a single eval rollout, and we can ask analogous questions:

1. **Across-episode drift of unit-level tuning.** Compute per-unit place-field-like maps in episode block 1 (eps 1-25) vs block 4 (eps 76-100). Per-unit cosine similarity of tuning maps across blocks. Does the median similarity differ across conditions?
2. **Stability of the decoder subspace.** Train a position decoder on block 1, test on block 4. Compare to "fresh" within-block decoder. The ratio = subspace stability index.
3. **Procrustes drift trajectory.** Procrustes-align block-i representations to block-1 across i in 1..4. Plot drift magnitude as a function of episode-block index.

## Hypothesis for our 5 conditions

Pre-registered:

- **Blind** should show the *most* drift in per-unit tuning but *most* stability in the decoder subspace (drift hidden in the null space of behaviour).
- **Sighted** should show *less* per-unit drift (sensor pinning).
- **Foveated** is the interesting case: peripheral context might drift while fovea stays pinned -> a *bimodal* drift histogram.

This would be a strong format/consumption result: drift is allowed precisely in the dimensions the policy doesn't read.

## Implementation cost

- ~60 LOC numpy/sklearn.
- Add: `scripts/probing/extra/representational_drift.py`.
- Runtime: ~5 min per condition.
- Total: 1.5 h end-to-end.

## Pseudocode

```python
def per_unit_drift(h, ep_id, pos, n_blocks=4):
    """For each unit, compute correlation of position-tuning between block 1 and block n_blocks."""
    eps = np.unique(ep_id)
    block_size = len(eps) // n_blocks
    blocks = [eps[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
    sims = np.zeros(h.shape[1])
    for d in range(h.shape[1]):
        m1 = build_tuning_map(h[np.isin(ep_id, blocks[0]), d], pos[np.isin(ep_id, blocks[0])])
        m2 = build_tuning_map(h[np.isin(ep_id, blocks[-1]), d], pos[np.isin(ep_id, blocks[-1])])
        sims[d] = pearsonr(m1.flatten(), m2.flatten())[0]
    return sims

def subspace_stability(h, ep_id, target, blocks):
    """Train decoder on block 1, test on block n; compare to within-block."""
    clf = Ridge().fit(h[blocks[0]], target[blocks[0]])
    cross = clf.score(h[blocks[-1]], target[blocks[-1]])
    within = Ridge().fit(h[blocks[-1]], target[blocks[-1]]).score(...)
    return cross / within
```

## What success / failure tells us

- **Success — bimodal drift in foveated:** strong format-axis result. Cheap and clean.
- **Null — uniform drift across conditions:** modest cost.
- **Drift exceeds typical biological levels:** weakens our biological-analogy framing — interesting in its own right.

## Risk

**Low.** No HPs to shop other than block-count. Fix at n_blocks=4 (so each block has 25 episodes, enough to estimate tuning maps with reasonable noise). Use Pearson correlation as the standard metric.

## Caveat

We have a frozen policy, so "drift" here is *intra-rollout* drift driven by changing scene / goal distribution across blocks. This is *not* the same as biological multi-day drift driven by synaptic plasticity. **Be explicit in the writeup** — frame as a "stability check" rather than as a direct biological analogue.

## Fit to capacity-allocation backbone

**Orthogonal (O) — would add a stability dimension to the format axis.** Mainly useful as a control to show our representations are not an artefact of one slice of the eval set. A null result here is a *positive* methodology footnote (representations are stable across the eval distribution).
