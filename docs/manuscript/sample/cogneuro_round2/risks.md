# Risk classification per method (DSA-style)

Risk = the probability the method produces a result that *appears* meaningful but is actually a hyperparameter / estimator artifact, leading to narrative-shoppy claims.

| Method | Risk | Drivers | Mitigation |
|---|---|---|---|
| Splitter cells | **LOW** | ANOVA is well-understood; FDR-correction handles multiple comparisons. Trajectory feature definition is the only HP. | Pre-reg the trajectory feature (`prev_action`, optionally augmented with rolling-mean direction). Report results separately for each definition; reject narrative if direction-flip across definitions. |
| Predictive coding residual | **LOW** | One-step forward MLP fit is stable; residual norm and covariance spectrum are robust. | Report `R²` of the forward model per condition first; if `R² < 0.5`, declare under-fit and stop. Pre-reg MLP architecture and train/test split. |
| Fisher info | **LOW** | Bias-correction via Kanitscheider 2015 estimator is established; only HP is regularisation. | Report Fisher across regularisation grid (λ ∈ {1e-4, 1e-3, 1e-2}); pre-reg the median value; require ordering to be stable across the grid. |
| Communication subspace | **LOW–MED** | RRR via CCA is stable; effective-rank threshold (95 %) is the main HP. | Pre-reg 95 % threshold; verify ordering at 90 % and 99 % thresholds as robustness check. Risk if policy is too low-dim (4 actions) — switch target to layer-1 LSTM activity if so. |
| Slow-point search | **LOW–MED** | Sussillo–Barak is well-replicated, but slow-point clustering depends on DBSCAN ε. Init-distribution affects coverage. | Pre-reg init = sample from cached `h_t`. Tune ε via silhouette per-condition. Run 3 random seeds; report mean ± std of slow-point count. Skip narrative if seed-variance > between-condition variance. |
| Persistent homology | **LOW–MED** | Subsample density and persistence threshold are HPs. PCA-32 projection adds risk but stabilises ripser runtime. | Pre-reg N=2000 subsample, 5 random seeds. Pre-reg persistence threshold 0.1. Use bottleneck distance (more robust than Betti counts). Run on PCA-32 *and* full 512-d on one condition to verify projection doesn't change ordering. |
| Gate analysis | **LOW** | Forward pass is deterministic; statistics are population-level. | Pre-reg input cache used for forward pass (must match what was used during training). Verify gate distributions are stable across episode subsets. |
| Time geometry | **MED** | Definition of "time on trajectory" is loose; angle estimate sensitive to PCA dimensionality. | Pre-reg `time_on_trajectory = step_in_episode - last_turn_step`. Report angle in 3-D, 5-D, 10-D PCA; ordering must match across at least 2/3. |
| Transfer entropy | **MED** | KSG is variance-prone in high-d; bias depends on `k`. Sample-size effects can flip sign. | Pre-reg `k=4`, N=10000 sub-samples, PCA-32 of `h_t`. Validate stability across `k ∈ {3,4,5}` on one condition before scaling. Cross-validate via MINE on a subset. |
| Avalanche / criticality | **HIGH** | Threshold, bin size, fitting method are all HPs known to flip results. Continuous LSTM activity is a poor match for the spike-train methodology. | Defer to supplement at most. If run, pre-reg one threshold (1σ), one bin (1-step), and report exactly one statistic (branching ratio). Do not run unless other methods finish early. |

---

## Decision matrix for the 4-hour subset

Pick from LOW risk first:

1. **Splitter cells** — pre-reg-safe, clear cogneuro precedent, 2h.
2. **Predictive coding** — pre-reg-safe, low HP surface, 2h.
3. **Fisher info** — pre-reg-safe, refines magnitude axis, 2h.

OR (more ambitious, if checkpoint loading works):

1. **Splitter cells** — 2h.
2. **Persistent homology** — 3h, but reads as the most "wow" cogneuro test.
3. **Slow-point search** — 3h, but requires checkpoint forward access.

Either combination = LOW–MED risk + strong narrative.

**Avoid** running transfer entropy or time geometry as Tier-S; they are MED risk and would benefit from being run after the LOW-risk core lands so we know whether the story is consistent.

**Skip** criticality unless slack time appears.

---

## What "honest report" looks like per method

For every method, the paper text must state:

1. The pre-registered prediction (verbatim from the per-method `.md`).
2. Whether the prediction landed (directional yes/no) and effect size (with bootstrap CI).
3. A 1-2 sentence "robustness" note covering HP variations checked.
4. Either "this analysis is in the main text" (predictions landed) or "this analysis is in the supplement; result is null/HP-sensitive".

This is the discipline that prevents the DSA / CCGP / Tangling-Q failure mode the user has flagged.

---

## Decision: which methods make the main text

Default policy:

- **Main text**: any method whose pre-reg landed directionally with effect size ≥ 1.5× and HP-stability across pre-registered grid.
- **Supplement**: any method that ran but failed pre-reg, with the explicit honest report.
- **Don't run**: any method whose risk is HIGH and whose contribution is a "nice-to-have" — including criticality.

Per principle: **Don't speculate. Don't be over-optimistic. Verify, dive deeper, only commit strong results to the paper.**
