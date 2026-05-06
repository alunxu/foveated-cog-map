# Time Cells / Ramping / Persistent-Activity Catalogue

**References.**
- MacDonald, C.J., Lepage, K.Q., Eden, U.T., Eichenbaum, H. (2011). *Hippocampal "time cells" bridge the gap in memory for discontiguous events.* **Neuron** 71(4), 737-749. https://doi.org/10.1016/j.neuron.2011.07.012
- Eichenbaum, H. (2014). *Time cells in the hippocampus: a new dimension for mapping memories.* **Nature Reviews Neuroscience** 15, 732-744. https://doi.org/10.1038/nrn3827

**One-line idea.** Hippocampal neurons fire at successive moments within structured experiences — orthogonal to place coding. The same population thus encodes *when* as well as *where*. Use a small fixed catalogue of motifs (time cell, ramping cell, persistent cell) and count units in each per condition.

## Original cogneuro use

MacDonald et al. found rodent CA1 cells that fired at specific time points within an empty delay between two events, irrespective of place — *time cells*. Eichenbaum (2014) reviewed and unified place, time, ramp, and persistent-activity motifs into a single coding catalogue. This catalogue is now standard for characterising hippocampal-style temporal memory codes.

The diagnostic motifs:
- **Time cell:** unimodal firing peak at a specific step-in-episode, robust across episodes.
- **Ramp cell:** monotonically increasing or decreasing across step-in-episode.
- **Persistent cell:** elevated activity at first event, sustained until second event (working-memory).
- **Phasic cell:** transient at event onset only.

## DL analogue on h_t

For each unit i in {1..512}:
1. Compute trial-averaged within-episode firing curve: f_i(t) = mean across episodes of h_t[i] at step t.
2. Classify each unit by a simple template-matching score:
   - time-cell-score: peakiness of f_i (1 - entropy of normalised f_i, or kurtosis).
   - ramp-score: |Pearson(f_i, t)| (linear trend).
   - persistent-score: variance(f_i) at high values vs return-to-baseline.
3. Each unit gets a (time, ramp, persistent) score triple. Plot population breakdown per condition.

## Hypothesis for our 5 conditions

Pre-registered:

- **Blind** should have the most time/ramp cells (it must internally track time-since-start to dead-reckon distance from origin).
- **Sighted** should have *fewer* time/ramp cells (sensor pinning provides position directly).
- **Foveated** is interesting: peripheral context might drive ramps while fovea does not.

A clean count-by-motif breakdown across the 5 conditions tests the **magnitude** axis with bio-aligned categories.

## Implementation cost

- ~80 LOC numpy.
- Add: `scripts/probing/extra/time_cell_catalogue.py`.
- Runtime: ~3 min per condition.
- Total: 2 h end-to-end including figure design.

## Pseudocode

```python
def per_unit_motif_scores(h, ep_id, step_id, T=100):
    """Returns (D, 3) array of (time, ramp, persistent) scores."""
    H = build_step_tensor(h, ep_id, step_id, T)  # (E, T, D)
    f = np.nanmean(H, axis=0)                    # (T, D)  trial-averaged curves
    D = h.shape[1]
    out = np.zeros((D, 3))
    t = np.arange(T) - T/2
    for d in range(D):
        curve = f[:, d]
        # time-cell: peakiness
        p = curve - curve.min()
        p = p / (p.sum() + 1e-9)
        ent = -(p * np.log(p + 1e-9)).sum()
        out[d, 0] = 1 - ent / np.log(T)  # in [0, 1]
        # ramp
        out[d, 1] = abs(np.corrcoef(curve, t)[0, 1])
        # persistent: ratio of high-value variance to overall
        hi = curve > np.percentile(curve, 75)
        out[d, 2] = curve[hi].std() / (curve.std() + 1e-9)
    return out
```

## What success / failure tells us

- **Success — clear shift in motif distribution from blind (time/ramp-rich) to sighted (mixed):** strong magnitude-axis figure with bio-aligned categories. Perfect for a reviewer asking "what kinds of cells are these like?"
- **Null — all conditions have similar motif counts:** modest cost, drop.

## Risk

**Low.** Three hand-crafted motif scores; report as a population breakdown so one or two outliers can't drive the result. No HP shopping (use fixed thresholds at the 50th/75th percentile).

## Caveat

The motif catalogue is by construction simplified — these aren't unitary "cell types" but axes of a 3D motif space. Frame as "we score every unit on three motif axes" rather than "we found N time cells".

## Fit to capacity-allocation backbone

**Orthogonal (O) on magnitude axis** with strong bio-language. Useful as a bridge to the cogneuro reviewer: "we don't just compute capacity; we recover MacDonald-style time cells in the blind condition." A modest but reviewer-friendly add.
