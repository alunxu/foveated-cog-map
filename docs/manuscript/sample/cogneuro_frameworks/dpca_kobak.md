# Demixed PCA (dPCA)

**Reference.** Kobak, D., Brendel, W., Constantinidis, C., Feierstein, C.E., Kepecs, A., Mainen, Z.F., Qi, X.-L., Romo, R., Uchida, N., Machens, C.K. (2016). *Demixed principal component analysis of neural population data.* **eLife** 5, e10989. https://doi.org/10.7554/eLife.10989  
Code: https://github.com/machenslab/dPCA (PyPI: `dPCA`).

**One-line idea.** Linear, supervised dimensionality reduction that decomposes population activity into orthogonal components each capturing variance attributable to a *named* task variable (stimulus, decision, time, condition).

## Original cogneuro use

dPCA was developed because PCA mixes time/condition/stimulus components together, and ANOVA-based tuning indices ignore population structure. Given a tensor of trial-averaged neural data X[neuron, time, stim, choice, ...], dPCA finds a separate low-rank decoder per *marginalisation* such that:
- the reconstruction sums to the original tensor
- each component captures variance only of its assigned variable, with cross-talk minimised.

Used widely in PFC, hippocampus, motor cortex to answer: how many population dimensions encode stimulus, vs decision, vs time, vs their interactions?

## DL analogue on h_t

Build a 4-way tensor:
H[neuron, time-in-episode, goal-bin, scene-bin]
- neuron: 512 (top-layer LSTM h)
- time-in-episode: bin step into 10 quantiles
- goal-bin: 4 quadrants of `goal_vec` direction
- scene-bin: scene id reduced to {bin0, bin1} by clustering scene embedding

Then run dPCA with marginalisations {time, goal, scene, time*goal, time*scene, goal*scene}.

For each condition (5 sensors), report % variance per marginalisation. The **format axis** prediction: foveated has more variance in `goal` and `time*goal` components; blind has more variance in `time` (because it relies on dead-reckoning over time).

## Hypothesis for our 5 conditions

Pre-registered: variance allocation across marginalisations differs per condition. Specifically:

| Marginalisation | blind | coarse | foveated | uniform | logpolar |
|------------------|-------|--------|----------|---------|----------|
| time | high | low | low | low | low |
| goal | low | high | high | high | high |
| time*goal | low | low | medium | low | high |

A single bar chart of "variance allocation per marginalisation" across 5 conditions is a self-contained format-axis figure.

## Implementation cost

- ~100 LOC, `pip install dPCA` then call `dPCA.dPCA().fit(H)`.
- Add: `scripts/probing/extra/dpca_marginalisation.py`.
- Runtime: ~5 min per condition (depends on tensor size).
- Total: 2 h end-to-end.

## Pseudocode

```python
from dPCA import dPCA

# H: trial-averaged tensor of shape (N=512, T=10, G=4, S=2)
dpca = dPCA.dPCA(labels='tgs', n_components=5)
dpca.protect = ['t']  # preserve time-axis variance allocation
Z = dpca.fit_transform(H)

# Z is dict; e.g. Z['t'] is time component, Z['g'] goal, Z['tg'] interaction.
var_explained = {k: dpca.explained_variance_ratio_[k] for k in Z}
```

## What success / failure tells us

- **Success — variance allocation differs systematically across conditions:** clean format-axis evidence. Reads as "the population spends its variance budget differently".
- **Null — same allocation everywhere:** weak result; no real cost.

## Risk

**Low.** dPCA is a standard linear method, no HPs to shop. The only judgement call is the binning of `goal` and `scene` — fix those upfront. Trial-averaging is required: ensure each (goal-bin, scene-bin, time-bin) cell has >= 5 trials.

## Fit to capacity-allocation backbone

**Sharpens (S) format axis** with the most readable *budget-allocation* figure type. Maps directly onto the paper's "capacity allocation" metaphor: dPCA literally shows how variance is allocated across task variables.

## Caveat

dPCA requires *trial-averaged* data, so it's a complement to (not replacement for) per-trial probes. With 100 episodes per condition and 4 goal bins x 2 scene bins x 10 time bins, we have ~80 cells with avg ~12 trials each — borderline. If too noisy: collapse scene-bin to 1 (i.e. drop scene marginalisation) and report a cleaner 3-marginalisation analysis.
