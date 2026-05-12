# Communication subspace

## Reference paper

Semedo, J. D., Zandvakili, A., Machens, C. K., Yu, B. M. & Kohn, A. (2019). *Cortical areas interact through a communication subspace.* **Neuron** 102, 249-259.

Replications / extensions:
- Kaufman et al. (2014) *Nat Neurosci* — null-space framework (already-published-precursor).
- Pinto, Tang, Goutte, Brody (2019) *eLife* — area-to-area communication.
- Stringer et al. (2019) **Nature** — high-dim cortical activity (related framing).
- Veuthey, Derosier, Kondapavulur, Ganguly (2020) *Nat Commun* — cortico-cortical communication.

## Original cogneuro use

When two brain areas communicate, the activity that is *transmitted* often lies in a low-dimensional **communication subspace** — only a subset of the source area's activity dimensions is read out by the target. Reduced-rank regression on simultaneous recordings:

\[ R_{\text{target}} = R_{\text{source}} \cdot B + \epsilon, \quad \text{rank}(B) = k \]

The optimal rank `k` (selected by cross-validation) and the directions of `B` characterise the *channel* between the two areas. Semedo 2019 showed that V1 → V2 communication uses only ~3 dimensions out of hundreds.

## DL analogue (our setting)

We have:
- `h_t` (LSTM top-layer hidden state, R^512).
- `policy_logits[t]` (action distribution, R^4 typically).
- (If cached) `lstm_layer1_h[t]` and `lstm_layer2_h[t]` for layer-to-layer communication.

Reduced-rank regression from `h_t` → `policy_logits[t]` gives us the **policy-readable subspace** of `h_t`. Its dimensionality `k` and directions characterise what the policy *consumes* from the recurrent state.

## Hypothesis (consumption axis)

1. **Effective rank** `k` of the `h_t → policy` channel is **smaller** for **blind** (the policy reads a more compact, integrator-like code) and **larger** for **uniform** (the policy reads a richer code that includes encoder-derived features).
2. The policy-readable subspace **aligns more strongly with the GPS-decoder direction** in blind/coarse than in foveated/uniform — confirming that "consumption" of the recurrent code in blind is GPS-aligned, while sighted policies consume non-position content too.
3. **Variance explained per dimension** decays faster in blind (rapid saturation at low `k`) than in sighted (gradual decay across many dims).

## Pseudocode

```python
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.linear_model import RidgeCV

def reduced_rank_regression(X, Y, rank):
    """RRR via CCA: project X → Y through rank-k bottleneck."""
    cca = CCA(n_components=rank).fit(X, Y)
    Y_hat = cca.predict(X)
    r2 = 1 - np.var(Y - Y_hat) / np.var(Y)
    return r2, cca

# Per condition:
for cond in conditions:
    cache = load_cache(cond)
    H, P = cache['h_t'], cache['policy_logits']
    # train/test split by episode
    train, test = episode_split([H, P], 0.8)
    r2_curve = [reduced_rank_regression(*train, rank=k)[0] for k in range(1, 32)]
    # effective rank: smallest k achieving 95 % of full-rank r2
    full = max(r2_curve)
    eff_rank = next(k for k, r in enumerate(r2_curve, 1) if r > 0.95 * full)
    # alignment with GPS direction
    rrr = reduced_rank_regression(*train, rank=eff_rank)[1]
    gps_dir = load_gps_decoder_direction(cond)  # from §3.4
    align = subspace_angle(rrr.x_loadings_, gps_dir)
    results[cond] = {'eff_rank': eff_rank, 'r2_curve': r2_curve, 'gps_align': align}
```

## Runtime estimate

- 32 ranks × 5 conditions × CCA fit: ~30 min CPU.
- Subspace angles + plotting: 30 min.
- Total: **~1.5–2 hours**.

## Success criteria

**Strong signal**: effective rank is monotonically increasing with sensor bandwidth, ratio > 2×; GPS-alignment is monotonically *decreasing*.

**Modest signal**: only effective-rank trend lands; alignment trend is noisy → supplementary.

**Failure mode**: policy logits are too low-dim (4 actions) for RRR to be informative — in that case, switch to communication subspace from `h_t` to `lstm_layer1_h` (intra-network rather than network-to-policy). Or use the **action-value head** if cached, which has higher dimension.

## Fit to backbone

Communication subspace is the **formal consumption axis**. The transplant experiment (§3.6) shows asymmetric consumption *behaviourally*; the communication-subspace test shows it *representationally*: the policy reads a different subspace of `h_t` per condition. The two together are a complete consumption-axis story.

## Pre-registered prediction

> The effective rank of the `h_t → policy_logits` reduced-rank regression (smallest `k` reaching 95 % of full-rank R²) computed via CCA on a held-out 20 % episode split will be monotone increasing across {blind, coarse, foveated_logpolar, foveated, uniform}, with the ratio (uniform/blind) ≥ 2. The principal subspace angle to the GPS-position decoder direction will be monotone increasing along the same axis (smaller alignment in sighted agents).

If predictions land, the consumption axis acquires a clean representational fingerprint complementing the behavioural transplant evidence.
