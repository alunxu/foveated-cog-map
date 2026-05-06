# Avalanche / criticality statistics

## Reference paper

Beggs, J. M. & Plenz, D. (2003). *Neuronal avalanches in neocortical circuits.* **J Neurosci** 23, 11167-11177.

Cogneuro replications:
- Petermann et al. (2009) *PNAS* — avalanches in awake monkey cortex.
- Friedman et al. (2012) *Phys Rev Lett* — universality class.
- Massobrio et al. (2015) *Front Sys Neurosci* — review.
- Bertschinger & Natschläger (2004) *Neural Comp* — edge-of-chaos and computation in random RNNs.
- Cocchi et al. (2017) *Prog Neurobiol* — clinical relevance / methodological pitfalls.

## Original cogneuro use

Cortical activity at rest shows scale-free avalanches: the distribution of avalanche sizes (number of co-active neurons in a contiguous time window) follows a power law \(P(s) \propto s^{-\alpha}\) with \(\alpha \approx 1.5\), consistent with a critical branching process. **Branching ratio** \(\sigma = \langle \text{descendant} / \text{ancestor} \rangle = 1\) at criticality. Critical networks are claimed to maximise dynamic range, information transmission, and memory capacity (Shew & Plenz 2013, *Trends Neurosci*).

## DL analogue (our setting)

For each condition, threshold `h_t` activity (z-score per unit, threshold at 1σ) to define "active" units per timestep. An **avalanche** = a contiguous run of timesteps with at least one active unit, separated by quiet timesteps. Avalanche size = total number of unit-activations in the run.

Compute:
1. Avalanche-size distribution \(P(s)\); fit power-law exponent \(\alpha\) (Clauset 2009 MLE).
2. Branching ratio \(\sigma\).
3. Compare to AR(1) shuffle baseline.

## Hypothesis

Mostly weak. **Marginal prediction**: blind agents — being purer dynamical integrators — may show more critical-like statistics than sighted agents which are forced toward task-relevant non-critical regimes. But this is highly speculative.

## Why this is on the bench (HIGH RISK)

- Avalanche analysis is **notoriously HP-sensitive** (threshold choice, time-bin width, network size). Cocchi et al. 2017 documents the methodological pitfalls.
- LSTM hidden states are continuous, not spike-trains; the avalanche definition is awkward without thresholding.
- The "criticality is computationally optimal" claim is contested in computational neuroscience (Wilting & Priesemann 2018).
- Even if we find a difference, the cogneuro story would be an over-stretch; reviewers will rightly push back.

**Recommendation**: include as a *supplementary sanity check* only if other Tier-S/A methods all land and we have spare time. Do **not** lead the narrative with this. If we run it, **pre-register the threshold and bin-width choice** (1σ, 1-step bin) and report exactly one statistic (the power-law exponent) — no narrative shopping.

## Pseudocode

```python
import powerlaw

def avalanche_stats(H, threshold=1.0, bin_size=1):
    Z = (H - H.mean(0)) / H.std(0)
    active = (Z > threshold).any(axis=1)
    # find avalanches: contiguous runs of active steps
    runs = find_runs(active)
    sizes = np.array([Z[r[0]:r[1]].sum() for r in runs if r[1] - r[0] > 0])
    fit = powerlaw.Fit(sizes, discrete=False)
    alpha = fit.alpha
    branching = np.mean([
        (Z[r[0]+1:r[1]] > threshold).sum() / max(1, (Z[r[0]] > threshold).sum())
        for r in runs
    ])
    return alpha, branching, fit
```

## Runtime estimate

- ~30 min/condition with proper bootstrap.
- Total: **~3 hours** including AR(1) shuffles.

## Success criteria

**Strong signal**: very unlikely. Even with a real difference, fitting power laws on RNN activity rarely produces a clean publication-grade result.

**Failure mode**: high probability. Defer.

## Fit to backbone

Marginal. Could supplement the dynamics axis with "blind agents are closer to a critical regime than sighted, suggesting the recurrent integration is operating at the edge of long-memory chaos" — but this is exactly the kind of speculative narrative the user wants to avoid.

## Pre-registered prediction

> If run, this analysis will report the power-law exponent \(\alpha\) of the avalanche-size distribution per condition (1σ threshold, 1-step bin, Clauset MLE) and the branching ratio \(\sigma\). Pre-registered hypothesis: \(\sigma\) closer to 1 in blind than in sighted, by ≥ 0.1; if not, declare null.

**Recommendation: skip unless other methods finish early. If runtime allows, run it strictly as the pre-registered single test above; do not narrative-shop.**
