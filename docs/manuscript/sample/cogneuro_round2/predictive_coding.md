# Predictive-coding residual analysis

## Reference paper

Rao, R. P. N. & Ballard, D. H. (1999). *Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects.* **Nature Neuroscience** 2, 79-87.

Replications / extensions:
- Keller & Mrsic-Flogel (2018) *Neuron* — predictive coding as a generic cortical computation.
- Bastos et al. (2012) *Neuron* — canonical microcircuits for predictive coding.
- Heilbron & Chait (2018) *Neuroscience* — empirical tests of prediction-error magnitude.
- Recent ML implementations: Boutin et al. (2021), Millidge et al. (2022) *Annu. Rev.*

## Original cogneuro use

A predictive-coding system maintains an internal forward model of its sensory input and propagates **only the prediction error** to higher layers. Mechanistic prediction: error neurons have firing rates proportional to \(|x_t - \hat{x}_t|\) where \(\hat{x}_t\) is the model's expectation. In an LSTM, the analogue is the **innovation** at the recurrent state: \(\epsilon_t = h_t - \hat{h}_t\), where \(\hat{h}_t = f(h_{t-1}, a_{t-1}, u_{t-1})\) is the one-step forward prediction from a learned dynamics model.

## DL analogue (our setting)

Fit a **post-hoc** one-step dynamics regressor `f̂: (h_{t-1}, a_{t-1}, u_{t-1}) → h_t` per condition (5-layer MLP, no retraining of the LSTM). The residual \(\epsilon_t = h_t - \hat{h}_t\) measures how much the actual recurrent update deviated from a smooth dynamics model — i.e. how much *new information* (sensory innovation) flowed into `h_t` at step `t`.

Capacity allocation makes a sharp prediction: the **size and structure of the prediction-error signal** should differ across conditions because the encoder delivers different amounts of new information per step.

## Hypothesis

1. **Mean prediction-error magnitude** \(E[\|\epsilon_t\|]\): larger for **foveated/uniform** (more novel sensory input lands at each step) than for **blind/coarse** (fewer novel inputs; more deterministic dynamics).
2. **Prediction-error structure** (spectrum of \(\Sigma_\epsilon = \text{Cov}(\epsilon_t)\)): blind agents should have a **lower-rank** error covariance — error lives in a 1-D innovation subspace tied to GPS sensor delta. Sighted agents should have a higher-rank, more isotropic error covariance.
3. **Error correlates with policy entropy** at the same step: at high-error steps the policy should be less confident (sensory innovation triggers re-evaluation). The slope of this correlation should be steeper for sighted conditions where the encoder genuinely brings new information.
4. **Error correlates with surprise / collision events**: if we have a "near-collision" or "scene-transition" indicator, predictive error should spike at those events more in sighted than in blind agents.

## Pseudocode

```python
from sklearn.neural_network import MLPRegressor

def predictive_coding_residuals(cond):
    H, A, U = load_cache(cond)  # h_t, action_t-1, encoder_out_t-1 (if cached) or proxy
    # train one-step forward model
    X = np.concatenate([H[:-1], one_hot(A[:-1]), U[:-1]], axis=1)
    Y = H[1:]
    # 80/20 episode split
    train, test = episode_split(X, Y, 0.8)
    f_hat = MLPRegressor(hidden_layer_sizes=(512, 512), max_iter=200).fit(*train)
    eps = test[1] - f_hat.predict(test[0])
    return {
        'mean_norm': np.linalg.norm(eps, axis=1).mean(),
        'cov_eigs': np.linalg.eigvalsh(np.cov(eps.T))[::-1],
        'entropy_corr': np.corrcoef(np.linalg.norm(eps, axis=1), policy_entropy[test_idx])[0, 1],
    }
```

## Runtime estimate

- MLP fit per condition: ~10 min on CPU (5 conditions × 10 min = 50 min).
- Spectrum + correlation analysis: ~5 min/condition.
- Total: **~1.5 hours**.

## Success criteria

**Strong signal**: predictions 1 + 2 land directionally. Effect on 1 should be > 30 % between blind and uniform.

**Modest signal**: only 1 of 4 lands → supplementary figure, framed as "predictive-error magnitude scales with encoder bandwidth, consistent with the encoder route carrying more information per step."

**Failure mode**: residual is dominated by MLP under-fitting noise. Mitigation: report \(R^2\) of `f̂` per condition; if `R^2` < 0.5, sample more data or use a larger MLP.

## Fit to backbone

This is the **innovation / surprise axis** of capacity allocation. Magnitude says how much position-info lives in `h_t`; predictive-coding residual says how much fresh sensor-info gets injected per step. If sighted agents have larger innovations *and* larger encoder-derived position information, this confirms the encoder is the relevant route. If blind agents have small innovations *and* larger linear-position decode, this confirms the recurrent integration is the relevant route.

This is one of the cleanest tests of the "two routes carrying information" claim because it operates per-step, not per-episode.

## Pre-registered prediction

> The mean per-step prediction error \(E[\|h_t - \hat{h}_t\|]\) computed from a held-out 5-layer MLP one-step forward model will be monotonically increasing across {blind, coarse, foveated, uniform}, with foveated_logpolar between coarse and foveated. The relative effect (uniform vs blind) will exceed 30 %. Furthermore, the rank-90 % of \(\Sigma_\epsilon\) will be lower for blind/coarse than for foveated/uniform.

If prediction holds, the paper's two-routes claim acquires a **per-step** validation (the consumption axis works frame-by-frame, not just episode-aggregate). This directly addresses the "is this just statistical agreement?" reviewer concern.
