# Tangling Q Metric

**Reference.** Russo, A.A., Bittner, S.R., Perkins, S.M., Seely, J.S., London, B.M., Lara, A.H., Miri, A., Marshall, N.J., Kohn, A., Jessell, T.M., Abbott, L.F., Cunningham, J.P., Churchland, M.M. (2018). *Motor Cortex Embeds Muscle-like Commands in an Untangled Population Response.* **Neuron** 97(4), 953-966. https://doi.org/10.1016/j.neuron.2018.01.004

**One-line idea.** A trajectory in state space is "untangled" if nearby points have similar derivatives (a smooth flow field). Russo et al. defined a tangling metric Q(t) = max_t' ||x'(t) - x'(t')||^2 / (||x(t) - x(t')||^2 + eps). High Q at any point means the derivative is multivalued there — the trajectory cannot come from an autonomous dynamical system.

## Original cogneuro use

Russo et al. asked: is motor cortex activity better described as encoding muscle commands (which are highly tangled) or as a smooth dynamical system (untangled)? They found that motor cortex *untangles* the muscle commands — population dynamics live on a smoother manifold than the muscle output, consistent with a generative/dynamical-system view of M1.

The metric is now used as a quick diagnostic: "is this population activity well-described as autonomous dynamics?"

## DL analogue on h_t

Compute Q for the LSTM trajectory in each condition:
- For all (t, t') pairs of states across all episodes, compute Q(t) = max_t' ||delta_h(t) - delta_h(t')||^2 / (||h(t) - h(t')||^2 + eps).
- Report distribution of Q across all timepoints, and median / 95th percentile per condition.

Since LSTM dynamics ARE autonomous given the input stream, the relevant question is: how *much* is the input doing the work vs the recurrent dynamics? Two extremes:
- Pure feedforward (input = everything): Q very high (nearby h, very different next-step delta because input differs).
- Pure autonomous (input = nothing): Q very low.

## Hypothesis for our 5 conditions

Pre-registered:

| Condition | Predicted Q | Why |
|-----------|-------------|-----|
| blind | low | dynamics dominate; sparse input |
| coarse | low-medium | weak input, mostly recurrent |
| foveated | medium-high | rich input drives dynamics fresh each step |
| uniform | high | richest input, most input-driven |
| logpolar | medium | intermediate |

This gives a *consumption-axis* claim: blind agents *use* their recurrent dynamics more (lower tangling = more autonomous).

## Implementation cost

- ~50 LOC numpy.
- Add: `scripts/probing/extra/tangling.py`.
- Runtime: ~3-5 min per condition (O(N^2) but only ~30k step pairs).
- Total: 1.5 h end-to-end.

## Pseudocode

```python
def tangling(h, ep_id, max_pairs=50000):
    """h: (N, D); compute Q distribution from random pairs of consecutive deltas."""
    deltas = np.zeros_like(h)
    for e in np.unique(ep_id):
        m = ep_id == e
        idx = np.where(m)[0]
        if len(idx) < 2: continue
        deltas[idx[:-1]] = h[idx[1:]] - h[idx[:-1]]
    # exclude episode-boundary deltas
    valid = ~np.all(deltas == 0, axis=1)
    h_v, d_v = h[valid], deltas[valid]
    n = len(h_v)
    # subsample pairs
    rng = np.random.default_rng(0)
    i = rng.integers(0, n, max_pairs)
    j = rng.integers(0, n, max_pairs)
    h_dist = np.linalg.norm(h_v[i] - h_v[j], axis=1)**2
    d_dist = np.linalg.norm(d_v[i] - d_v[j], axis=1)**2
    Q = d_dist / (h_dist + 1e-6)
    return Q
```

## What success / failure tells us

- **Success — Q ordering matches sensor info:** clean consumption-axis evidence (sensors do the work, recurrent dynamics do less).
- **Success — Q ordering inverted (blind = high Q):** would tell us blind has *messy* dynamics (which would be surprising and interesting).
- **Null — flat Q:** modest cost.

## Risk

**Low.** Single number, single HP (max_pairs / subsampling). Use percentile-based summaries (median + 95th) rather than means to be robust to long tails.

## Fit to capacity-allocation backbone

**Orthogonal (O) on consumption axis with a dynamics flavour.** Pairs naturally with intrinsic timescales (Pick #6) — together they triangulate "how autonomous is the recurrent code per condition". Cheap supplementary panel; not headline material but solid.
