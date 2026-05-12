# Linear Fisher information / population sensitivity

## Reference paper

Beck, J. M., Ma, W. J., Kiani, R., Hanks, T., Churchland, A. K., Roitman, J., Shadlen, M. N., Latham, P. E. & Pouget, A. (2008). *Probabilistic population codes for Bayesian decision making.* **Neuron** 60, 1142-1152.

Cogneuro replications:
- Brunel, Hakim, Richardson (2014) *Curr Opin Neurobiol* — single-neuron / population sensitivity.
- Kanitscheider, Coen-Cagli, Pouget (2015) *PNAS* — origin of information-limiting noise.
- Moreno-Bote et al. (2014) **Nat Neurosci** — information-limiting correlations.
- Kafashan, Jaffe, Chettih, Nogueira, Arandia-Romero, Harvey, Moreno-Bote, Drugowitsch (2021) **Nat Commun** — measuring high-dim Fisher in cortex.

## Original cogneuro use

Linear Fisher information about a stimulus parameter \(s\) given population activity \(\mathbf{r}\):

\[ I_F(s) = (\partial_s \boldsymbol{\mu})^\top \Sigma^{-1} (\partial_s \boldsymbol{\mu}) \]

where \(\boldsymbol{\mu}(s) = E[\mathbf{r} \mid s]\) and \(\Sigma\) is the noise covariance at fixed `s`. \(I_F\) bounds the variance of any unbiased estimator (Cramér–Rao). It is the cleanest **geometry-aware** measure of how much information about `s` the population carries, accounting for noise structure.

In V1 (Kafashan 2021), high-dim Fisher about orientation reaches ~5× the linear-Fisher estimate, revealing nonlinear-decodable information not captured by linear probes.

## DL analogue (our setting)

For each condition, estimate linear Fisher about `agent_pos[0]` (x-coordinate) and `agent_pos[1]` (y-coordinate) from `h_t`:

1. Bin `agent_pos` into discrete 'stimulus' levels.
2. Estimate \(\boldsymbol{\mu}(s)\) as the mean `h_t` per bin.
3. Estimate \(\Sigma(s)\) as the within-bin covariance (averaged across bins for stability).
4. Compute \(\partial_s \boldsymbol{\mu}\) via finite differences.
5. \(I_F = (\partial_s \boldsymbol{\mu})^\top \Sigma^{-1} (\partial_s \boldsymbol{\mu})\).

## Hypothesis (capacity-allocation prediction)

This test is the **most rigorous magnitude-axis check** because it explicitly accounts for noise correlations — addressing the reviewer concern that "linear R² differences could come from noise structure, not signal magnitude."

1. **Total Fisher** (sum across `x` and `y`) decreases monotonically with encoder bandwidth: blind > coarse > sighted, mirroring linear-decode R² but with proper noise correction.
2. **Bias-corrected Fisher** (Kanitscheider 2015 estimator) shows the same ordering, ruling out the "low-data-bias-inflates-blind-Fisher" objection.
3. **Decomposition into signal vs noise contribution**: the magnitude difference comes mostly from \(\partial_s \boldsymbol{\mu}\) (signal direction) rather than from \(\Sigma\) (noise) — confirming the format axis.

## Pseudocode

```python
import numpy as np

def linear_fisher(H, s, n_bins=20, regularise=1e-3):
    """Linear Fisher info about scalar s from population H."""
    bins = np.linspace(s.min(), s.max(), n_bins + 1)
    mu = np.zeros((n_bins, H.shape[1]))
    Sigma = np.zeros((H.shape[1], H.shape[1]))
    for i in range(n_bins):
        m = (s >= bins[i]) & (s < bins[i + 1])
        if m.sum() < 50: continue
        mu[i] = H[m].mean(0)
        Sigma += np.cov(H[m].T) * m.sum()
    Sigma /= len(H)
    Sigma += regularise * np.eye(H.shape[1])  # regularised inverse
    Sigma_inv = np.linalg.inv(Sigma)
    # finite-diff signal
    dmu_ds = np.gradient(mu, axis=0).mean(0) / np.gradient(bins).mean()
    return dmu_ds @ Sigma_inv @ dmu_ds

def bias_corrected_fisher(H, s, n_bootstrap=20):
    """Kanitscheider 2015 bias correction via train/test split."""
    fishers = []
    for _ in range(n_bootstrap):
        train, test = episode_split(H, s, 0.5)
        I_train = linear_fisher(*train)
        I_test = linear_fisher(*test)
        fishers.append(2 * I_train * I_test / (I_train + I_test))  # harmonic-mean estimator
    return np.mean(fishers), np.std(fishers)

# Per condition × axis:
for cond in conditions:
    H = load_cache(cond)['h_t']
    pos = load_cache(cond)['agent_pos']
    I_x = bias_corrected_fisher(H, pos[:, 0])
    I_y = bias_corrected_fisher(H, pos[:, 1])
    results[cond] = {'I_x': I_x, 'I_y': I_y}
```

## Runtime estimate

- Per (condition, axis, bootstrap): ~30 sec.
- 5 conditions × 2 axes × 20 bootstrap = 200 estimates = **~1 hour**.
- Plus regularisation tuning + signal/noise decomposition: **~2 hours total**.

## Success criteria

**Strong signal**: bias-corrected Fisher decreases monotonically across the bandwidth axis, with > 2× ratio between blind and uniform. Decomposition shows the change is signal-dominated.

**Modest signal**: ordering matches but bootstrap CIs overlap; relative effect < 1.5× → supplement.

**Failure mode**: regularisation bias dominates (regularise too low → covariance-singular; too high → identity-collapse). Mitigation: report Fisher across regularisation grid; pre-register the median value.

## Fit to backbone

Fisher info is a **noise-corrected magnitude axis**. The R² result in §3.4 already shows the linear position decode varies; Fisher info shows that variation is real (noise-corrected) and quantifies the ratio with the right Cramér–Rao constant. It is a cleaner measure than R² for cross-condition comparisons because R² depends on label variance.

## Pre-registered prediction

> Bias-corrected linear Fisher information \(I_F(\text{pos})\) about `agent_pos` from cached `h_t` (computed via the Kanitscheider 2015 train-test harmonic-mean estimator with regularisation 1e-3 and 20-bin discretisation) will be monotone decreasing across {blind, coarse, foveated_logpolar, foveated, uniform}, with the relative ratio (blind / uniform) ≥ 2. The decomposition into signal and noise contributions will show the magnitude difference is dominated by the signal direction \(\partial_s \boldsymbol{\mu}\), with noise covariance \(\Sigma\) contributing < 30 % of the variance.

This is the safest, lowest-risk magnitude refinement in round 2 — recommend running it as the **fallback** if slow-point search hits a checkpoint-loading blocker.
