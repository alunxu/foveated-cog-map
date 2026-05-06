# Transfer entropy: sensor → h_t → policy

## Reference paper

Schreiber, T. (2000). *Measuring information transfer.* **Phys. Rev. Lett.** 85, 461-464.

Cogneuro replications:
- Vicente, Wibral, Lindner, Pipa (2011) *J Comput Neurosci* — TE for neural directed-information flow.
- Wibral, Vicente, Lizier (2014) *Direct. Inf. Meas. in Neuroscience* — comprehensive review.
- Pearl-causal foundations: Pearl (2009).
- ML implementation: Kraskov, Stögbauer, Grassberger (2004) *Phys Rev E* — KSG estimator (de-facto standard).

## Original cogneuro use

Transfer entropy from `X` to `Y`:

\[ TE_{X \to Y} = I(Y_{t+1} ; X_t \mid Y_t) \]

i.e. the conditional mutual information between past `X` and future `Y` given past `Y`. Operationally: how much does knowing `X_t` help predict `Y_{t+1}` beyond what `Y_t` already tells us? It is a directed, model-free measure of information flow used to infer functional connectivity in neural recordings (Wibral 2014, Bossomaier 2016 *Introduction to Transfer Entropy*).

## DL analogue (our setting)

We have three time series at each step:
- `u_t` = encoder output (visual features). For blind, this is the GPS+compass sensor only.
- `h_t` = top-layer LSTM hidden state.
- `a_t` = policy action (or policy logits).

Transfer entropies of interest (per condition):

1. \(TE_{u \to h}\) — how much new information flows from sensors into recurrent state per step?
2. \(TE_{h \to a}\) — how much policy is driven by recurrent state vs by direct sensor pathways?
3. \(TE_{u \to a}\) — direct sensor → policy flow (skip-recurrent).
4. \(TE_{\text{GPS} \to h}\) — specifically the GPS / self-motion channel into `h`.

## Hypothesis (capacity-allocation prediction)

1. \(TE_{u \to h}\) increases with encoder bandwidth: blind ≈ coarse < foveated ≈ uniform. (Sighted agents ingest more visual info per step.)
2. \(TE_{\text{GPS} \to h}\) decreases with encoder bandwidth: blind > coarse > sighted. (Blind is forced to integrate GPS more.)
3. \(TE_{h \to a}\) is roughly constant (the policy depends similarly on `h` regardless of condition — consumption is similar in magnitude but draws different content).
4. \(TE_{u \to a}\) is small for all conditions (policy reads from `h`, not directly from `u`) — unless the encoder has a learned shortcut, in which case sighted may show a non-zero direct path. **This is a sharp test for "policy bypasses memory in sighted agents."**

## Pseudocode

```python
from sklearn.feature_selection import mutual_info_regression
from npeet import entropy_estimators as ee  # Kraskov KSG estimator

def transfer_entropy(X, Y, k=4):
    """TE_{X -> Y} = I(Y_{t+1}; X_t | Y_t) via KSG."""
    Y_next, Y_past, X_past = Y[1:], Y[:-1], X[:-1]
    # CMI(Y_next; X_past | Y_past)
    return ee.cmi(Y_next, X_past, Y_past, k=k)

# Per condition:
for cond in conditions:
    cache = load_cache(cond)
    H = pca(cache['h_t'], 32)  # reduce dim — KSG breaks in high-d
    A = cache['action_logits']  # or one-hot action
    U = cache['encoder_out']    # or sensor proxy if not cached
    GPS = cache['agent_pos_delta']  # self-motion sensor
    results[cond] = {
        'TE_u_to_h': transfer_entropy(U, H),
        'TE_GPS_to_h': transfer_entropy(GPS, H),
        'TE_h_to_a': transfer_entropy(H, A),
        'TE_u_to_a': transfer_entropy(U, A),
    }
```

## Runtime estimate

- KSG on N=10000 samples, 32-d: ~10 min per TE call.
- 4 TEs × 5 conditions = 20 calls = **~3 hours** on CPU.
- Risk: KSG is sensitive to `k` (number of nearest neighbours). Pre-reg `k=4`, validate stability across `k ∈ {3, 4, 5}` on one condition before scaling.

## Success criteria

**Strong signal**: predictions 1 + 2 land directionally; effect size > 1.5× across blind/uniform on both. Prediction 3 satisfied (no significant ordering of \(TE_{h \to a}\)).

**Modest signal**: only 1 directional trend lands → supplementary report.

**Failure mode**: KSG estimates are highly variable with sample size; bias depends on dimensionality. Mitigation: use **MINE** (Mutual Information Neural Estimation, Belghazi 2018) on a subset to validate ordering; bootstrap the TE differences across episodes. If ordering is unstable, declare "TE-based test inconclusive" and rely on the other axes.

## Fit to backbone

Transfer entropy is the **information-flow axis** of capacity allocation. We have already shown:
- magnitude (probes): position info changes magnitude in `h_t` across conditions.
- format (Procrustes, transplant): the *direction* changes.
- consumption (transplant asymmetry): the policy uses different routes.

TE adds: **directed, per-step information flow** consistent with the route picture. This is the strongest formalisation of "two routes carrying spatial information."

## Pre-registered prediction

> Transfer entropy estimated via KSG (`k=4`) on N=10000 sub-sampled aligned timesteps from each condition will satisfy: (a) \(TE_{\text{visual} \to h}\) is monotone increasing across {blind, coarse, foveated_logpolar, foveated, uniform}; (b) \(TE_{\text{GPS} \to h}\) is monotone decreasing along the same axis; (c) the ratio \(TE_{u \to h} / TE_{\text{GPS} \to h}\) varies by ≥ 3× across the extremes.

If (c) lands, that is the cleanest possible "two-route allocation" claim — a single scalar that summarises the capacity-allocation story.
