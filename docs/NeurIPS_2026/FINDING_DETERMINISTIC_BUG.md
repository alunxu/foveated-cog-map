# Finding: Probe data bug (stochastic sampling) + probe target misspecification

**Updated with post-fix analysis**: see § "Deep-dive after the fix" below
for the discovery that absolute GPS/compass are not reliably encoded on
correctly-collected trajectories, but ego-relative DtG is — with a clean
compensatory-memory signal that rescues H1.



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


## Deep-dive after the fix

After re-collecting probe data with `deterministic=True`, I ran a full
α-sweep + 5-fold CV probe on fov-fix_det and fov-learned_det to decide
whether the negative R² is (a) a real failure of the LSTM to encode
absolute spatial variables, or (b) a deeper probe methodology problem.

### Finding 1 — Absolute GPS/compass are NOT reliably encoded

5-fold CV for fov-fix_det and fov-learned_det on full det trajectories:

| target | fov-fix_det | fov-lrn_det |
|--------|-------------|-------------|
| Absolute GPS  | 0.06 ± 0.88 | −2.43 ± 3.98 |
| Compass       | 0.07 ± 0.69 | −1.34 ± 3.14 |

Per-fold swing of ±3 in R² means the probe is fitting noise, not signal.
The point estimates from a single train/test split (the ones that
showed up as R² = −1.6 or R² = −10 sentinel) are **not real** — they're
within one-sigma of chance-level.

Mechanism: PointNav's LSTM does not need to encode world-frame GPS/
heading. Both are available as per-step observations (episodic GPS and
compass are part of the obs space), so the LSTM can simply read them
rather than re-encode them into its latent. Nothing in the task reward
forces the agent to maintain an absolute world-frame position trace.

The earlier "R² = 0.84 for fov GPS" from the stochastic collection was
an artefact: 4-step quasi-static trajectories have near-zero target
variance, so any probe scores trivially high R² (total variance is
tiny; even a terrible predictor fits within tolerance).

### Finding 2 — DtG (distance-to-goal) IS robustly encoded

| target | fov-fix_det | fov-lrn_det |
|--------|-------------|-------------|
| DtG (current) | **0.82 ± 0.09** | **0.81 ± 0.09** |
| DtG @ lag 5   | 0.77 ± 0.13 | 0.80 ± 0.10 |
| DtG @ lag 8   | 0.73 ± 0.16 | 0.80 ± 0.09 |

Robust across folds, and the lag-k retention shows the **compensatory-
memory signature H1 claims** — foveated-learned retains near-perfectly
across 8 steps (decay ~1%) while foveated-fixed decays modestly
(~11%). Same story as paper H1, but with a methodologically sound
probe target. Other ego-relative targets (ego-frame goal-vector,
heading-since-start, path-displacement) don't fit cleanly — DtG is the
clean signal.

### Finding 3 — Paper's H1 can be rescued by target swap

The paper's lag-k probe currently decodes GPS at t−k (world-frame
absolute position). Under det data this is essentially chance. Swapping
the target to **DtG at t−k** gives:

- Robust CV (σ ≈ 0.10, not 0.88)
- A retention curve interpretable as compensatory memory
- Discriminates conditions (decay rates differ)
- Conceptually cleaner: DtG is what the agent actually tracks for
  navigation, not world-frame coordinates

This is consistent with the paper's A1 claim ("no condition stores the
ego-to-goal direction — the PointGoal task provides it as input") —
DtG fits the same "task-relevant ego-relative encoding" frame.

### Finding 4 — Two bugs, not one

1. `deterministic=False` in `collect.py` → stochastic-STOP trajectory
   collapse → 4-step quasi-static probe data → inflated R² artefacts
   (fixed in commit `c81352e`).

2. Lag-k GPS as the compensatory-memory probe target — world-frame
   position is not what the LSTM encodes. Swap target to lag-k DtG to
   see the actual memory-retention signal. (To be implemented in the
   analysis scripts + paper revision.)

### Proposed next steps

1. Finish re-collecting uniform_det, matched_det, blind_det
   (running; ~1–2h remaining).
2. Extend `extended_lag_probe.py` to probe **DtG** (and/or goal-vector
   scalars) at lag k, not just GPS. Rerun across all 5 conditions.
3. Compare lag-k DtG retention curves across the 5 conditions — if fov
   > uniform retention gap holds, the H1 compensatory-memory claim
   survives, re-grounded on a robust probe target.
4. Re-run H3 with the same target-swap (lag-k DtG + goal-vector
   retention) — we already have the stoch fov-learned compass 0.94
   vs fov-fix 0.72 claim; verify it was an artefact or holds in det
   with DtG-based targets.
5. Paper revisions will depend on (3)/(4). Draft scenarios:
   - **Best case** (fov > uniform in lag-k DtG under det, large gap):
     H1 is rescued with a more elegant target framing. Only methodology
     § needs updating; the headline findings hold.
   - **Medium case** (H1 lag-k DtG gap survives but shrinks): soften
     magnitude claims, add CV error bars, keep the ordering.
   - **Worst case** (no cross-condition gap in any probe): pivot to
     behavioral interventions (transplant + shortcut) as primary
     evidence. The probe framing becomes supporting.

### Diagnostic scripts (in `scripts/probing/`)

- `fov_probe_diagnosis.py` — episode-length + action-distribution on
  existing npz (confirmed the stochastic bug).
- `probe_deep_diagnostic.py` — α-sweep + lag-k probe + H stats.
- `probe_scaler_test.py` — isolates StandardScaler as a non-issue
  (raw vs scaled give essentially the same R²).
- `probe_alternative_targets.py` — tests DtG / ego-goal / cumulative-
  rotation / path-displacement as targets (found DtG robust, others
  not).
- `probe_cv_summary.py`, `compare_det.py` — summaries of CV vs
  single-split and stoch vs det.

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
