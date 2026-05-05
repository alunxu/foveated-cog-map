# Flagged for user review (loop 2026-05-05)

Each row in this file failed or partially failed the B.1 sanity gate. None of these were auto-applied to main.tex; the user should inspect, decide whether the paper claim needs revision, and update accordingly.

## P1-F: Subspace divergence — angles narrower than paper claim

**Paper claim** (line ~358): "mean principal angles between top-K subspaces lie in $[86°, 89°]$"

**Post-loop measurement**: angles in $[69°, 87°]$ — wider distribution, still near-orthogonal but the tight 86-89° claim is too narrow.

| | blind | coarse | foveated | fov_logpolar | uniform |
|---|---|---|---|---|---|
| blind  | 0 | 81 | 77 | 86 | 87 |
| coarse | 81 | 0 | 78 | 77 | 81 |
| foveated | 77 | 78 | 0 | 69 | 78 |
| fov_logpolar | 86 | 77 | 69 | 0 | 85 |
| uniform | 87 | 81 | 78 | 85 | 0 |

**Smallest pair**: foveated ↔ foveated_logpolar = 69°. These two are the most-similar conditions by design (both foveation; one with log-polar resampling). The 69° lower bound is consistent with that.

**Recommended paper edit**: `[86°, 89°]` → `[69°, 87°]` (or `at least 69° across all pairs; rich-encoder same-family pairs cluster at the lower end`). Direction (near-orthogonal subspaces) survives qualitatively; the specific tight interval was overstated.

**Source**: `/tmp/rcp_analysis_v3/subspace_divergence_5cond.json` (re-run with mean-center-only, K_per_cond = blind 7 / coarse 21 / foveated 33 / fov_logpolar 4 / uniform 9 — paper claim K∈{8,9,10} also broader than reality)

## P1-H: Predictive horizon blind k=20 weaker than paper

**Paper claim** (line ~287): "Blind sustains a long predictive horizon: $R^2 \geq 0.94$ from $k=0$ to $k=20$"

**Post-loop measurement**: blind k=20 R² = 0.733 (well above sighted but well below paper's 0.94)

**Source**: `/tmp/rcp_analysis_v3/predictive_horizon_5cond.json`

**Recommended action**: pull the actual k-profile from the JSON, update paper text. Possible rewrite: "Blind sustains a substantial predictive horizon: linear R² $\geq 0.73$ from $k=0$ to $k=20$, decaying gradually". Direction (blind predicts longer than sighted) survives.

## P1-G: Subspace evolution NOT computed

**Status**: blind cross-training NPZs (c10/15/20) still in flight at end of loop; c5 was killed for being too slow.

**Recommended action**: when c10/15/20 land later (~30-60 min after loop end), re-run `run_subspace_evolution.py` with full 5-cond × 4 ckpt grid. If results land before deadline, regenerate `fig_subspace_evolution.pdf`. Otherwise the existing pre-retrain figure stands; paper claim that subspaces are "established early" was not re-verified.

## Transplant 6 cells missing (cross-spatial-size limitation)

The 5×5 transplant matrix has 14 of 20 valid cross-pair cells. The 6 missing cells are:
- coarse → foveated, foveated_logpolar, uniform
- foveated, foveated_logpolar, uniform → coarse

**Cause**: transplant.py builds a single shared environment from one config; coarse uses 48×48 RGB while sighted use 256×256 RGB, producing incompatible recipient model architectures. Fix would require building two separate envs (donor + recipient), running each for half the trajectory.

**Mitigation**: the asymmetry test relies on blind axis only (3 bn→rich + 3 rich→bn cells). Direction confirmed (-0.23 vs ~0). Coarse axis would have doubled the cells; with current 14, the test is qualitatively decisive but not as tight as a full 5×5.

**Recommended paper note**: add a footnote to §H2 transplant paragraph: "*The 5×5 matrix excludes coarse↔{foveated, foveated-logpolar, uniform} pairs because their differing input spatial size (48×48 vs 256×256) is incompatible with our shared-env transplant pipeline; the asymmetry test relies on the blind axis.*"

## P0-C cross-cond probe transfer: mild not catastrophic

**Paper claim** (line ~334): "probe-transfer R² is catastrophically negative across conditions"

**Post-loop measurement**: max off-diagonal R² = -0.124 (off-diagonals range -0.124 to -5.97).

**Note**: blind ROW off-diagonals are MILD (-0.15 to -0.28) while OTHER rows are strongly negative (-1 to -6). This is a NEW asymmetry finding: blind's linear position direction works "partly" on other conds; other conds' don't generalize to blind.

**Recommended paper edit**: add 1 sentence to §H2 "Convergent evidence" para: "*The asymmetry shows up at the probe level too: probes trained on blind hidden states transfer with mild loss to rich-encoder states ($R^2 \approx -0.15$ to $-0.28$), while the reverse direction is catastrophically negative ($R^2$ from $-1$ to $-6$). This mirrors the transplant-asymmetry direction.*"

**Source**: `/tmp/rcp_analysis_v3/cross_transfer_5cond.json`

## Excursion-forgetting: blind value pending

Auto_fire_v2 ran on the existing wjf_v2_summary.json which had only 4 sighted (matched/foveated/uniform/foveated_logpolar). Blind exc-5 did Complete during loop — the data exists at `/scratch/wxu/.../excursion_results/blind_excursion.npz` but the wjf_v2_summary.json aggregator was not re-run with blind included.

**Recommended action**: rerun `excursion_analyze_v2.py` with all 5 conds. The currently-in-paper claim "blind +0.17 (smallest)" was from prior data. New sighted values are 0.17-0.21 (similar across) per existing wjf_v2_summary.json; whether blind is genuinely smaller needs the merge.
