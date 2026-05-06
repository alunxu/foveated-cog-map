# Intrinsic Timescale Hierarchy

**Reference.** Murray, J.D., Bernacchia, A., Freedman, D.J., Romo, R., Wallis, J.D., Cai, X., Padoa-Schioppa, C., Pasternak, T., Seo, H., Lee, D., Wang, X.-J. (2014). *A hierarchy of intrinsic timescales across primate cortex.* **Nature Neuroscience** 17, 1661-1663. https://doi.org/10.1038/nn.3862

**One-line idea.** The autocorrelation timescale of a neuron's spontaneous spike count grows monotonically along the cortical hierarchy (V1 ~50 ms, parietal ~100 ms, ACC/dlPFC ~300+ ms). Sensors are fast, integrators are slow.

## Original cogneuro use

Murray et al. fit the spike-count autocorrelation of single neurons during inter-trial intervals (or fixation) with a single-exponential A * exp(-Delta/tau) + offset, and showed tau increases reliably from sensory cortex through parietal to prefrontal. This is now a standard quick-look measure of where in the cortical hierarchy a region sits, and has been replicated cross-species (mice, macaques, humans).

## DL analogue on h_t

For each unit i in {1..512} of the LSTM hidden state, compute the autocorrelation function (ACF) of h_t[i] across timesteps, separately within each episode. Average ACFs across episodes. Fit single-exponential decay:

ACF(Delta) ~= A * exp(-Delta / tau_i) + offset

Report:
- per-unit tau_i
- distribution of tau_i across the 512 units (histogram)
- median tau and IQR per condition

## Hypothesis for our 5 conditions

The information-bottleneck framing in our discussion *predicts* a relationship between sensor bandwidth and integration timescale. Pre-registered:

| Condition | Predicted median tau | Why |
|-----------|----------------------|-----|
| blind | longest | must integrate proprioception across many steps |
| coarse 1x1 | second longest | weak per-step signal, integration helps |
| foveated | shortest? or bimodal | rich per-step signal; might have a *bimodal* distribution: fovea-locked fast units + peripheral-context slow units |
| uniform | short | richest per-step signal |
| logpolar | bimodal predicted | log-polar prior should produce a clean fast/slow split |

The bimodality prediction for foveated and logpolar is the most distinctive — if true, this is a strong format-axis result (not just "magnitude differs"; the *shape of the timescale distribution* differs).

## Implementation cost

- ~40 LOC numpy.
- Add: `scripts/probing/extra/intrinsic_timescales.py`.
- Runtime: ~3 min per condition.
- Total: 1 h end-to-end.

## Pseudocode

```python
import numpy as np
from scipy.optimize import curve_fit

def autocorr(x, max_lag=30):
    """1D autocorrelation up to max_lag steps."""
    x = x - x.mean()
    n = len(x)
    out = np.zeros(max_lag+1)
    out[0] = 1.0
    var = (x*x).mean()
    for k in range(1, max_lag+1):
        out[k] = (x[:-k]*x[k:]).mean() / var
    return out

def per_unit_tau(h, ep_id, max_lag=30):
    """Returns (D,) array of fitted tau values."""
    eps = np.unique(ep_id)
    D = h.shape[1]
    taus = np.full(D, np.nan)
    for d in range(D):
        acfs = []
        for e in eps:
            x = h[ep_id==e, d]
            if len(x) < max_lag+5: continue
            acfs.append(autocorr(x, max_lag))
        acf_avg = np.mean(acfs, axis=0)
        try:
            (A, tau, c), _ = curve_fit(
                lambda d, A, tau, c: A*np.exp(-d/tau)+c,
                np.arange(max_lag+1), acf_avg,
                p0=[1.0, 5.0, 0.0], bounds=([0,0.1,-1],[2,200,1]))
            taus[d] = tau
        except Exception:
            pass
    return taus
```

## What success / failure tells us

- **Success — distinct distributions per condition (esp. bimodality in foveated/logpolar):** fast-cheap figure showing that *temporal architecture* differs across conditions. Fits as a supplementary panel or as a brief inset.
- **Null — all conditions identical:** modest cost, just don't include it.

## Risk

**Low.** Single-exponential fit is standard. Two safeties: (1) drop units with R^2 of fit < 0.5, (2) use ACF averaged within-episode then across episodes (don't pool across episodes — that mixes within- and across-episode statistics).

## Fit to capacity-allocation backbone

**Sharpens (S) magnitude axis** with a temporal flavour, **and** could sharpen format if bimodality emerges. Provides a fast / cheap supplementary panel that grounds our LSTM-as-integrator claim in an established neuroscience metric.

## Bonus extension

Compute per-unit tau separately for c_t (cell state) and h_t (hidden state). The classical LSTM intuition says c is "longer-memory" than h. Verify and report — would be an interesting figure on its own but is orthogonal to the paper's main story.
