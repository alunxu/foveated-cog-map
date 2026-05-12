# Forget / input / output gate statistics

## Reference paper

Hochreiter, S. & Schmidhuber, J. (1997). *Long short-term memory.* **Neural Computation** 9, 1735-1780.

Cogneuro / interpretation precedents:
- Karpathy, Johnson, Fei-Fei (2015) *ICLR-w* — "Visualizing and understanding RNNs"; gate-firing analyses.
- Levy, Hoffmann, Andreas (2018) *ICLR* — "Long Short-Term Memory as a Dynamically Computed Gate" (gating as adaptive timescale).
- Tallec & Ollivier (2018) *ICLR* — "Can Recurrent Neural Networks Warp Time?" (gate openness ↔ effective integration timescale).
- Gao & Goldman (2011) *Annu Rev Neurosci* — biological short-term memory and adaptive timescales.

## Original cogneuro use

The forget gate \(f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)\) controls how much of the cell state is retained:

\[ c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \]

When \(f_t \to 1\), the cell holds onto its past indefinitely (long integration timescale); when \(f_t \to 0\), it dumps and writes fresh content. Tallec & Ollivier 2018 show gate openness directly determines the network's effective time constant, an analogue of the **intrinsic timescale** measured in cortex (Murray 2014). Mean forget-gate openness per unit ≈ a measurable correlate of biological adaptive-time-constant tuning.

## DL analogue (our setting)

We have cached `h_t` and `c_t`. To recover gate values we need a forward pass through the LSTM with cached inputs. **Two paths**:

(a) **Direct**: load LSTM checkpoint from RCP, forward `(h_{t-1}, c_{t-1}, encoder_out_t)` to recover `f_t, i_t, o_t`. Cleanest.

(b) **Inference-only**: from cached `h_t, c_t` we can derive
\[ f_t \odot c_{t-1} = c_t - i_t \odot \tilde{c}_t \]
which requires `i_t` and `\tilde{c}_t`. Without LSTM weights, this is **under-determined**. So path (a) is required.

Compute per-condition statistics:
1. Mean and variance of \(f_t\), \(i_t\), \(o_t\) per unit.
2. Distribution of "always-open" units (\(\bar{f}_i > 0.95\)) and "always-closed" units.
3. Effective time constant per unit: \(\tau_i = -1 / \log(\bar{f}_i)\).
4. Distribution of \(\tau_i\) per condition — this is the **gating analogue** of the intrinsic-timescale spectrum.

## Hypothesis

1. **Mean forget-gate openness** is **higher** for blind/coarse than foveated/uniform: blind agents need long-range integration, so their gates favour retention. Sighted agents update more on each step.
2. **Distribution of \(\tau_i\)**: blind has a heavier tail at long \(\tau\) (some units acting as quasi-permanent integrators); sighted has more mass at short \(\tau\).
3. **Cross-correlation with intrinsic-timescale result** (round 1 already done): the gate-derived \(\tau_i\) should correlate with the autocorrelation-derived intrinsic timescale at the unit level. This **confirms the round-1 timescale result has a mechanistic substrate** (gates), not just a population-statistics artifact.

## Pseudocode

```python
def gate_stats(lstm, encoder_out, h0, c0):
    f_seq, i_seq, o_seq = [], [], []
    h, c = h0, c0
    for x_t in encoder_out:
        # custom LSTM step that returns gates
        f, i, g, o, h, c = lstm_step_with_gates(lstm, h, c, x_t)
        f_seq.append(f); i_seq.append(i); o_seq.append(o)
    return np.stack(f_seq), np.stack(i_seq), np.stack(o_seq)

# Per condition:
for cond in conditions:
    lstm = load_lstm(cond)
    F, I, O = gate_stats(lstm, encoder_out_cache[cond], h0_cache[cond], c0_cache[cond])
    f_mean[cond] = F.mean(axis=(0, 1))
    tau[cond] = -1.0 / np.log(np.clip(f_mean[cond], 1e-6, 1 - 1e-6))
    # cross-correlate with round-1 intrinsic-timescale per unit
    intrinsic_tau = load_round1_timescales(cond)
    corr = np.corrcoef(tau[cond], intrinsic_tau)[0, 1]
```

## Runtime estimate

- Forward pass per condition: ~10 min on GPU (5000 episodes × 500 steps × 512 units).
- 5 conditions = **~1 hour** on GPU.
- Statistics + cross-correlation: 30 min.
- Total: **~1.5 hours**.

## Success criteria

**Strong signal**: f-gate openness ordering matches sensor axis with > 10 % gap; \(\tau_i\) distribution heavier-tailed for blind; cross-correlation with round-1 timescales > 0.6.

**Modest signal**: only the cross-correlation lands (validating round 1) — supplement.

**Failure mode**: gate distributions are roughly identical because all conditions trained on the same task (the gate equilibrium is set by the task loss, not the sensor). If true, this is a *negative result that supports the format-not-magnitude story*: same gating dynamics, different content. We should report this honestly.

## Fit to backbone

Gate analysis is the **mechanistic substrate** of the dynamics axis. It tells us *why* blind has long timescales (round 1) — because forget gates are more open. It tells us *why* sighted updates faster — because input gates are more open. It is a unit-level confirmation of population-level dynamics.

## Pre-registered prediction

> Mean forget-gate openness \(\bar{f}_i\) per unit, computed by a forward pass of cached encoder outputs through the frozen LSTM, will be monotone decreasing across {blind, coarse, foveated_logpolar, foveated, uniform}, with the population mean differing by ≥ 5 % between extremes. The unit-level effective time constant \(\tau_i = -1/\log(\bar{f}_i)\) will be Spearman-correlated (ρ > 0.5) with the intrinsic-timescale value derived from autocorrelation analysis (round 1).

If both directional predictions land, the round-1 timescale result acquires a mechanistic explanation; if only the correlation lands, the round-1 result is validated even if the gating equilibrium is task-set rather than sensor-set.
