# Sequenceness / Replay Detection (TDLM)

**References.**
- Liu, Y., Dolan, R.J., Kurth-Nelson, Z., Behrens, T.E.J. (2019). *Human Replay Spontaneously Reorganizes Experience.* **Cell** 178(3), 640-652. https://doi.org/10.1016/j.cell.2019.06.012
- Liu, Y., Dolan, R.J., Higgins, C., et al. (2021). *Temporally delayed linear modelling (TDLM) measures replay in both animals and humans.* **eLife** 10, e66917. https://doi.org/10.7554/eLife.66917
- Pfeiffer, B.E., Foster, D.J. (2013). *Hippocampal place-cell sequences depict future paths to remembered goals.* **Nature** 497, 74-79. https://doi.org/10.1038/nature12112

**One-line idea.** Train state decoders on hidden activity, then test whether decoded state sequences during rest / planning periods systematically follow a transition matrix of interest (forward, reverse, or shuffle baseline). This quantifies "sequenceness" as a function of time lag.

## Original cogneuro use

Pfeiffer & Foster (2013) showed in rodents that hippocampal place-cell sequences during pauses depict *future paths* to remembered goals (preplay) — a key piece of cognitive-map evidence. Liu et al. (2019) brought this to human MEG: they trained item-level decoders, then during 5-min rest looked at how the decoded sequence reflected a *learned* transition structure (with state-to-state lags ~40-60 ms). The eLife 2021 paper formalised this as TDLM (Temporally Delayed Linear Modelling): regress decoded probabilities at time t+lag on time t, get a matrix B(lag); contrast against a transition-matrix-of-interest T to get a single scalar sequenceness(lag).

## DL analogue on h_t

For our setup, **"replay" requires a quiescent / non-action period** which we don't have during rollouts (the agent is always acting). We can instead test:

1. **Within-trajectory preplay:** at the *first* k steps of an episode, before the agent has moved much, do decoded position estimates already trace out the eventual trajectory? (This is the Pfeiffer/Foster preplay analogue.)
2. **Cross-trajectory replay:** at episode end (after success), is there a replay-like signature where decoded position briefly traces back over the just-traversed path? (Probably weak in feedforward LSTM rollouts.)
3. **Counterfactual mental travel:** force-feed the LSTM with a held-out (pseudo-rollout) sequence of actions and decode positions; compare to actual rollouts.

The TDLM regression is well-defined for any decoded-state sequence, so the framework applies as long as we have a state-decoder and a state-transition-matrix-of-interest.

## Hypothesis for our 5 conditions

This is the most speculative of the shortlist. Tentative pre-registered predictions:

- **Blind** should show *more* preplay (it must internally simulate possible trajectories given its weak sensors).
- **Sighted** should show *less* preplay (each step's sensor reading drives a fresh estimate; less need to simulate).

A clean preplay-ordering result would be a *consumption-axis* finding: blind agents *use* their internal simulator more.

## Implementation cost

- ~150 LOC TDLM-style regression on top of the existing position decoder.
- Add: `scripts/probing/extra/preplay_tdlm.py`.
- Runtime: ~30 min per condition.
- Total: 3 h end-to-end.

## Pseudocode

```python
def tdlm_sequenceness(decoded_probs, T, max_lag=20):
    """decoded_probs: (T_steps, K_states); T: (K, K) transition-of-interest.
    Returns sequenceness(lag) for lag in 1..max_lag."""
    out = np.zeros(max_lag+1)
    for L in range(1, max_lag+1):
        X = decoded_probs[:-L]
        Y = decoded_probs[L:]
        B, *_ = np.linalg.lstsq(X, Y, rcond=None)  # (K, K) regressor
        # sequenceness = trace(T @ B) - trace(T.T @ B)  (forward minus reverse)
        out[L] = np.trace(T @ B) - np.trace(T.T @ B)
    return out
```

## What success / failure tells us

- **Success — clear preplay signature, ordered blind > coarse > sighted:** the headline consumption-axis figure we don't currently have.
- **Null — no signature:** likely outcome given LSTM PointNav doesn't have explicit rest periods. Would simply not be reported.

## Risk

**MEDIUM-HIGH.** Two reasons:

1. **Conceptual fit risk.** TDLM was designed for *resting-state* MEG. Our agent is always acting; the analogue is forced and might not produce a signal.
2. **HP-sensitivity.** The choice of state discretisation (number of K position bins), max lag, and transition-matrix-of-interest can all be HP-shopped. *Pre-register K=16 spatial bins, lag=1..20, transition=true graph adjacency from the navmesh.*

This method is **highest in upside, highest in risk**. Do not start it before CCGP/TGM are done. Skip entirely if MFTMA also needs to run.

## Fit to capacity-allocation backbone

**Orthogonal (O) — would test consumption axis** in a way nothing else on the shortlist does. But the conceptual fit to a continuous-acting LSTM is forced. Save for a "future work" position unless we have spare budget after the safer picks land.
