# Tensor Component Analysis (TCA)

**Reference.** Williams, A.H., Kim, T.H., Wang, F., Vyas, S., Ryu, S.I., Shenoy, K.V., Schnitzer, M., Kolda, T.G., Ganguli, S. (2018). *Unsupervised Discovery of Demixed, Low-Dimensional Neural Dynamics across Multiple Timescales through Tensor Component Analysis.* **Neuron** 98(6), 1099-1115. https://doi.org/10.1016/j.neuron.2018.05.015  
Code: https://github.com/neurostatslab/tensortools

**One-line idea.** Approximate the (neuron x time x trial) tensor as a sum of R rank-1 outer products, each contributing one neuron factor, one temporal factor, and one trial factor. Unsupervised analogue of dPCA — discovers structure rather than tests labelled marginalisations.

## Original cogneuro use

TCA (also called CP / PARAFAC decomposition) was popularised in neuroscience by the Ganguli lab. Given X[neuron, time, trial], a rank-R decomposition gives:

X[i,t,k] ~= sum_r a_r[i] * b_r[t] * c_r[k]

with three factor matrices A (neurons), B (time), C (trials). Williams et al. used this on prefrontal cortex during a maze task and motor cortex during reach: trial factors revealed slow learning trends, temporal factors revealed within-trial dynamics, neuron factors revealed cell assemblies — all *jointly fitted*, not labelled.

## DL analogue on h_t

Build X[neuron=512, time-in-episode (truncated to T=100), episode_id (E=100)] for each condition.

Run TCA with R in {3, 5, 8} components. For each component r, examine:
- a_r (neuron factor): which units belong to the assembly
- b_r (temporal factor): the within-episode dynamics it captures
- c_r (episode factor): which episodes it activates in (and: does this correlate with goal direction? scene? distance?)

## Hypothesis for our 5 conditions

Pre-registered:

1. **Number of meaningful components** (where reconstruction R^2 plateaus): foveated should need *more* components than blind (richer representation -> more independent assemblies).
2. **Episode-factor structure:** in sighted conditions, c_r should correlate with goal-related variables; in blind, c_r should correlate with episode length / scene complexity.
3. **Temporal-factor structure:** blind should show more sustained / ramping b_r; sighted should show more periodic / step-locked b_r.

## Implementation cost

- ~80 LOC using `tensortools` package.
- Add: `scripts/probing/extra/tca_decomposition.py`.
- Runtime: ~10 min per condition x rank.
- Total: 2 h end-to-end.

## Pseudocode

```python
import tensortools as tt
# X: (neurons=512, time=100, trials=100)
X = build_tensor(h, ep_id, step_id, T=100)
ensembles = tt.Ensemble(fit_method='ncp_hals')  # non-negative CP
ensembles.fit(X, ranks=[3,5,8,10], replicates=4)
ensembles.plot()  # shows rank vs error elbow
factors = ensembles.results[5][0].factors  # rank-5 best fit
```

## What success / failure tells us

- **Success — distinct factor structures per condition:** orthogonal evidence on the format axis (assembly composition differs).
- **Success — rank elbows differ:** quantitative claim on representational complexity per condition (magnitude axis).
- **Null — all conditions give similar factors:** report as supplementary; no cost.

## Risk

**Low-medium.** TCA has an HP: the rank R. Mitigate by reporting an *elbow plot* (R vs reconstruction error) per condition rather than fixing R. Use non-negative CP (NCP) — better identifiability than vanilla CP. The `tensortools` package handles random restarts and reports stability across restarts; require stability >= 0.7 (cosine similarity of factors across replicates) before reporting.

## Caveat / overlap with dPCA

TCA is unsupervised, dPCA is supervised. They answer related but different questions:
- **dPCA**: "How is variance allocated to my labelled task variables?"
- **TCA**: "What latent assemblies and dynamics emerge without labels — and do they then match my labels?"

Run *one* of these in the budget. Recommendation: **dPCA first** because it produces a cleaner figure aligned with the capacity-allocation framing. Use TCA as supplementary if dPCA is null.

## Fit to capacity-allocation backbone

**Orthogonal (O) — adds dynamics-aware geometric evidence.** The neuron-factor view also gives us a *unit-level* story (which units cluster into which assemblies) that pairs nicely with the population-level Procrustes / PR analyses.
