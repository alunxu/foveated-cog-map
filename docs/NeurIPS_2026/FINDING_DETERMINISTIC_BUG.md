# Finding: Probe data was collected with stochastic action sampling

## TL;DR

Every eval script in the codebase uses `deterministic=True` with an explicit
"deterministic for eval" comment — except `scripts/probing/collect.py`, which
hardcoded `deterministic=False`. This caused a systematic trajectory-
distribution mismatch across conditions in the probing data:

| Condition       | Probe episode length | Probe success rate | Training-eval SPL |
|-----------------|---------------------:|-------------------:|------------------:|
| foveated-fixed  | **4 steps (mean)**   | **0 %**            | 0.83              |
| uniform         | **4 steps**          | **0 %**            | 0.85              |
| foveated-learned| 171 steps            | 96.6 %             | 0.80              |

The mechanism is straightforward: under stochastic sampling, expected
time-to-STOP is `1 / p(STOP)`. Fov-fix and uniform policies have higher
action-entropy (p(STOP) ≈ 0.25 ⇒ 4-step average), while fov-learned has
low action-entropy (p(STOP) ≈ 0.006 ⇒ 171-step average). The
`deterministic=True` eval protocol collapses this away.

## What this affects

**Affected** (uses probe data collected via collect.py):
- H1: per-step GPS/compass R² curves
- H2: probe transfer matrix, lag-k R² curves, CKA
- H3: compass R² under sensor-mask, all probe-based comparisons
- Summary table: GPS R², compass R², distance-to-goal R²

**Not affected** (uses its own deterministic eval loop):
- Training SPL/success from habitat-baselines
- Shortcut test (`scripts/eval/shortcut.py` — `deterministic=True`)
- Transplant test (`scripts/eval/transplant.py` — `deterministic=True`)
- MP3D eval (uses habitat-baselines)
- Debug eval (`scripts/eval/debug_eval.py` — `deterministic=True`)

## What changed in the codebase

1. `scripts/probing/collect.py`: added `--deterministic` flag (default: `True`).
   Old hardcoded `deterministic=False` removed.

2. `scripts/cluster/submit_probe_deterministic.sh`: new submit script that
   re-collects with `--deterministic=True`, writes to `*_det.npz` so the
   stochastic collection is preserved for diffing.

3. `scripts/cluster/resubmit_probes_deterministic.sh`: batch submitter that
   fires off re-collection for all five canonical conditions.

## Action items

### Step 1 — Confirm the diagnosis on existing data (fast)

Copy the diagnostic to the cluster and run on the login node (no GPU
needed, just reads npz files):

```bash
scp /tmp/fov_probe_diagnosis.py izar:/tmp/
ssh izar 'cd ~/cs503-project && conda activate habitat && python /tmp/fov_probe_diagnosis.py'
```

Expected output confirming the diagnosis:
- fov-fix/uniform: >80% of episodes have length ≤ 5, <20% have any FWD action
- fov-learned: broad length distribution with substantial FWD fraction

### Step 2 — Re-collect probe data with deterministic=True

```bash
# Push the fixed collect.py + new submit scripts:
bash scripts/cluster/sync_to_cluster.sh

# Then on Izar:
ssh izar 'cd ~/cs503-project && bash scripts/cluster/resubmit_probes_deterministic.sh'

# This kicks off 5 jobs (blind, uniform, fov, fov-learned, matched) each
# producing <run_name>_det.npz. Monitor with: squeue -u $USER
```

Each job takes ~1–3h depending on episode length.

Optional, run later: `resubmit_probes_deterministic.sh all` for MP3D +
masked variants.

### Step 3 — Re-run all downstream analyses on det data

Once all 5 `*_det.npz` files exist:

```bash
ssh izar 'cd ~/cs503-project && sbatch scripts/cluster/rerun_analyses_on_det.sh'
# Or run inline if it's fast; that script produces:
#   probing_results/*_det_analysis.json
#   probing_results/extended_lag_det.json
#   probing_results/cross_transfer_det.json
#   probing_results/h3_det_analysis.json
#   probing_results/goal_vector_det.json
#   probing_results/cka_det.json
```

### Step 4 — Sync results back and update paper

```bash
bash scripts/cluster/sync_from_cluster.sh
```

Then update:

| Section                            | What to refresh                                        |
|------------------------------------|--------------------------------------------------------|
| §4.2 & Table 1 (`tab:summary`)     | GPS R², compass R², DtG R² columns — all 5 rows        |
| §4.3 H1 (per-step probe curves)    | Per-step R² at k=0…5, lag-k curves                     |
| §4.4 H2 (CKA + transfer + lag-5)   | CKA matrix, transfer matrix, lag-5 R²                  |
| §4.5 H3 (compass R² comparison)    | Fov-learned vs fov-fix compass R², goal-vector R²      |
| Methods §3.3                       | Add: "probing uses deterministic action selection      |
|                                    | (`collect.py --deterministic=True`), matching the      |
|                                    | eval protocol in `shortcut.py`, `transplant.py`, and   |
|                                    | habitat-baselines eval."                               |
| §H2 fov-learned caveat (line 213)  | Reframe: "All conditions produce comparable 100–200    |
|                                    | step trajectories under deterministic evaluation. The  |
|                                    | first-4-steps-per-episode matched subset we adopt is   |
|                                    | a conservative length-matched control."                |

### Step 5 — Update figures

Regenerate via `scripts/paper_figures/make_h1h2_figures.py` and
`scripts/paper_figures/make_h3_content_figure.py` after pointing them at
the new analysis JSONs. The `h1_order` exclusion of fov-learned from
`fig_path_history` can likely be dropped (that was a work-around for the
sample-count artefact caused by this bug).

## Why the first-4-steps truncation was only a partial fix

The paper currently matches trajectory length by taking the first 4 steps
of every episode. This does make lengths comparable — but the nature of
those 4 steps still differs across conditions:
- For fov-fix/uniform (under the bug), the "first 4 steps" = the entire
  episode, leading up to a stochastic STOP. Agent is deciding to stop.
- For fov-learned (under the bug), the "first 4 steps" = the opening of
  a long, directed trajectory. Agent is committing to a direction.

Both sets of 4 steps happen under the same policy at comparable early-
episode state, so the compensatory-memory claim (H1) is not invalidated,
but the effect size is confounded by the stop-vs-go distinction. The
deterministic re-collection removes this confound.

## Confidence

**High** that this is the mechanism:
- Every other eval script in the repo explicitly uses `deterministic=True`
- `collect.py` is the only inconsistency
- Observed probe episode lengths (4 vs 171) exactly match the predicted
  `1/p(STOP)` ratio for high- vs low-entropy policies
- Training-eval SPL for fov-fix (0.83) is incompatible with 0% probe
  success unless the two evals use different sampling protocols
- **Paper internal contradiction (Methods §3.3)**:
  - Line 105: "Four of the five conditions produce short evaluation
    episodes (mean ≤ 5 steps)" — from the probing data (stochastic bug)
  - Line 129: "Gibson PointNav episodes average ∼100–120 steps" — from
    transplant/behavioral eval (deterministic)
  - Same underlying policies, same test split, same dataset. The only
    difference is the sampling protocol, which reproduces the two
    regimes exactly.

## Draft replacement text for §3.3 Methods (apply after re-collection)

Old (line 105):
> Four of the five conditions (blind, uniform, foveated-fixed, matched-
> compute) produce short evaluation episodes (mean ≤ 5 steps in the
> Gibson val split) because the agents navigate efficiently to goal.
> Foveated-learned produces much longer episodes (mean 171, median 77
> steps) because its collapsed gaze yields inefficient but ultimately
> successful trajectories. This causes an ∼100× difference in the
> effective spatial range covered by the probe-training set per episode
> between foveated-learned and the other four conditions.

New (if re-collection confirms ~uniform episode lengths):
> Probing rollouts use deterministic (argmax) action selection, matching
> the eval protocol of our shortcut-discovery and memory-transplant
> interventions and of standard habitat-baselines evaluation. Under this
> protocol, all five conditions produce comparable episode lengths
> (mean ~100–150 steps, consistent with the 100–120-step transplant
> baseline in §3.4). We collect 500 episodes per condition on the held-
> out Gibson val split, yielding ~60–80k labelled per-step hidden states
> per condition — enough for stable episode-split Ridge probes.

(Replaces the "mean ≤ 5 steps" framing + the "natural 4-step horizon"
truncation policy; the lag-$k$ probe description simplifies because
every condition now has sufficient consecutive-step samples.)

## Draft replacement text for H2 fov-learned caveat (line 213)

Old:
> Foveated-learned's transfer values are one-to-two orders of magnitude
> worse than other pairs because its average episode is ∼40× longer,
> spreading its probe data over a much larger spatial range. On a
> first-4-steps-per-episode subset matched to the others, its cross-
> transfer falls into the $[-2, -15]$ band of every other pair.

New:
> (Revisit once we have deterministic-collection numbers; under matched
> episode lengths the ~40× imbalance should vanish and the cross-
> transfer interpretation is straightforward. Expected: all pairs in
> the same $[-2, -15]$ band, no special caveat needed.)

**Remaining uncertainty**: quantitative effect on probe R² values can only
be measured by actually re-running. Our qualitative predictions:
- Re-collection will give all conditions 100–300 step episodes.
- GPS R² comparison (foveated 0.84 vs uniform 0.61) likely *survives*
  because the effect is driven by representation, not trajectory length.
- Compass R² gap (fov-learned 0.94 vs fov-fix 0.72) likely *survives*
  because fov-learned genuinely sees a different foveation location.
- Rank ordering across conditions should be preserved.
- H3 gaze-location mechanism story is unaffected (foveated-shifted
  training in progress provides clean causal control regardless).
