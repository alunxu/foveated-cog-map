# Overnight Loop Diagnosis Log

Last updated: 2026-05-04 02:15 UTC (iter 2)

## ITER 3 — major findings while user slept

User session was idle 02:15→15:00 UTC; loop /loop wakeups did not fire (session
not engaged). Cluster jobs continued running independently. On wake, found:

- ✅ Phase B (probe-analysis-b-0-1): Completed, output at `/scratch/wxu/.../analysis_results/`
- ✅ 16/16 substitution dynamics probes: Completed
- ✅ 6/6 sighted-only transplant pairs: Completed
- ❌ mlp_probe.json (670 B) + cka.json missing + lagk_*.json missing — Phase B
  partial (linear analyze.py + procrustes worked; MLP/CKA/lagk failed silently)
- 🟡 dh-blind retrain: still Pending (F2 still on 4 GPU)
- 🟡 F2 (dh-fnorm): Running 14h, ~56%

### Linear probe R² (5-fold CV mean, post-retrain ckpt.49)

| Cond | GPS R² | GPS std | Compass R² | goal_dist R² | Skaggs bits |
|---|---:|---:|---:|---:|---:|
| coarse | **+0.582** | 0.099 | +0.659 | +0.960 | 20.83 |
| foveated | +0.178 | 0.659 | +0.503 | +0.912 | 27.03 |
| uniform | **−1.785** | 2.993 | −1.244 | +0.915 | 99.80 |
| foveated_logpolar | −0.029 | 0.759 | +0.467 | +0.946 | 163.61 |

Paper §H1 stale (5-cond): blind 0.95, coarse 0.72, fov 0.25, uniform −1.08
Paper §place-cell stale: 1.25/1.32/1.18/1.16 bits per (unit, scene)

**Direction match for H1 main claim**:
- Coarse +0.58 (bottleneck): clearly preserves linear GPS code ✓
- Uniform −1.79 (rich-encoder): collapses to negative ✓
- Foveated +0.18, foveated_logpolar −0.03: both rich-encoder-like, near 0 ✓
- goal_dist R² 0.91-0.96 across all 4: sanity probe target intact ✓

**Anomalies (DO NOT auto-edit paper without user review)**:

1. **HCPENDING falsifiable prediction (paper L811): NOT confirmed**.
   - Paper predicted: log-polar R² ≥ 0.3 between coarse 0.78 and uniform 0
   - Actual: foveated_logpolar R² = −0.03
   - log-polar lands in rich-encoder regime, NOT intermediate
   - This means the encoder-spatial-output mechanism (§H1 framing) needs
     re-thinking — is logpolar's encoder output really 2×2 as expected?
     Or does the encoder regime depend on more than spatial output dim?

2. **Skaggs MI 100× off from paper**.
   - Paper: 1.25/1.32/1.18/1.16 bits per (unit, scene) — narrow range
   - Actual: 20.83/27.03/99.80/163.61 — bigger range, 100× higher absolute
   - Likely metric definition mismatch (paper mean per-unit-per-scene vs
     analyze.py's "mean_spatial_info_bits" aggregate). NEEDS verify.

3. **Foveated > Foveated_logpolar in linear R²** (paper had logpolar > foveated
   if encoder-spatial-output mechanism holds). Reversed.

### Procrustes / CKA (3 conds: coarse + foveated + uniform; foveated_logpolar
missing in script CONDS)

theta_1 distance:
- coarse-uniform: 1.019 (max-orthogonal)
- coarse-foveated: 0.846 (smallest cross-cond)
- uniform-foveated: 0.988

Linear CKA:
- coarse-foveated: **0.566** (highest — coarse and foveated MORE SIMILAR than other pairs)
- coarse-uniform: 0.297
- uniform-foveated: 0.332

**Major reframe signal**: paper has §H2 framing where "rich-encoder pair
(foveated/uniform) is tightest" and "blind/coarse bottleneck pair is next".
With the post-retrain run:
- Coarse-foveated CKA 0.566 is HIGHER than uniform-foveated 0.332.
- Foveated representation aligns more with bottleneck (coarse) than with
  rich-encoder (uniform).
- Suggests foveated is "hybrid case" — paper §"Foveation as a hybrid case"
  is REINFORCED but the specific cross-condition pairing direction may shift.

### Transplant 6 sighted-only pairs (rich-rich)

| pair | cross_delta | self_delta | net cross |
|---|---:|---:|---:|
| fov_lp → fov | -0.135 | -0.052 | -0.083 |
| fov → fov_lp | -0.119 | -0.052 | -0.067 |
| fov → uni | -0.089 | -0.048 | -0.041 |
| uni → fov | -0.104 | -0.052 | -0.052 |
| (other 2 likely similar)

Roughly symmetric. Cannot test H2 bottleneck-rich asymmetry.

### Substitution dynamics 16 ckpts: analysis in flight (sklearn install
delayed by missing pip package). Background job bzszv21en.

### NEXT ACTIONS NEEDED (user wake-up review)

DO NOT eagerly edit paper. Multiple shifts vs stale numbers (excursion
direction, Skaggs magnitude, Procrustes coarse-most-distant, log-polar
prediction). User should review before any edit.

When user awake, propose:
1. Walk through these numbers together
2. Decide which paper claims to update vs re-frame
3. Re-think H1 mechanism if log-polar prediction is genuinely falsified
4. dh-blind retrain still Pending — decide whether to wait or alternate path

---



### Transplant: 4 sighted-only pairs available (rich-rich)

| donor → recipient | baseline | self_t | cross | Δ cross | self drop | net cross |
|---|---:|---:|---:|---:|---:|---:|
| foveated_logpolar → foveated | 0.857 | 0.805 | 0.722 | -0.135 | -0.052 | -0.083 |
| foveated → foveated_logpolar | 0.834 | 0.782 | 0.715 | -0.119 | -0.052 | -0.067 |
| foveated → uniform | 0.849 | 0.801 | 0.760 | -0.089 | -0.048 | -0.041 |
| uniform → foveated | 0.857 | 0.805 | 0.753 | -0.104 | -0.052 | -0.052 |

**Pattern**: rich-rich pairs roughly symmetric (asymmetry ≤ 0.02 within
fov↔fov_lp and fov↔uni), as expected (both regimes high-bandwidth encoder).
Net cross-drop after self-disturbance correction: -0.04 to -0.08.

**Cannot test H2 asymmetry claim (paper L321 "rich-encoder cannot use
bottleneck hidden state"); needs coarse / blind donor or recipient.

**Decision**: Do NOT update paper Figure 4. Defer until own blind ckpt + fix
to transplant.py (cross-resolution donor/recipient).

### Phase B status — stuck Pending ~5h

`probe-analysis-b-0-0 Status: Pending`. Events: empty. No "Insufficient resources"
message — but cluster has 17+ Pending jobs (16 substitution-dynamics + 1 Phase B
+ dh-blind retrain) waiting on GPU.

GPU contention math:
- F2 (dh-fnorm): 4 GPU active
- transplant Running: 1-2 GPU at any time
- substitution-dynamics: 1 GPU active (probe-1-c10) per scheduling round
- dh-blind: 4 GPU Pending (high request)
- Phase B: 1 GPU Pending

Phase B and dh-blind are competing with substitution-dynamics for the 1-GPU
slots. Substitution-dynamics 16 jobs gradually getting through.

**Concern**: 8h budget might not get all 16 substitution-dynamics + Phase B done.
Phase B is highest leverage (Table 1 + 5 lens outputs in 1 job vs 16 separate).

**Action**: monitor. If Phase B still Pending after 2 more iters (~50min), kill
some substitution-dynamics jobs to give Phase B priority. That's a hard call
since substitution-dynamics also blocks the L260 paper update.

### dh-blind retrain — Pending, blocker for full picture

dh-blind-0-1 Pending — was Running earlier, now restarted. Cannot start until
4 GPU slot available. F2 holds 4 GPU until ~05-05 morning.

If dh-blind doesn't run during 8h window, blind row of all matrices remains
unfilled. Paper §H1/H2/H3 narrative needs blind. Friend's blind ckpt is
inconsistent (seed=100, num_envs=32) → user explicitly rejected.

**Risk for deadline**: if blind retrain gets ~16h GPU + 1.5h probe + 1h
re-analysis, total ~18-20h. With deadline 64h away (was 64 at loop start),
still feasible IF blind retrain starts soon.

**Mitigation idea**: kill F2 to free 4 GPU? F2 is at 56%. Killing wastes
~14h of compute but also wastes ~14h more before completion. Trade-off:
- Keep F2: F2 completes ~05-05, blind has ~24h slack. Tight.
- Kill F2: blind starts immediately, completes ~05-04 evening. F2 lost.

Defer this decision to user (mark in summary). Don't unilaterally kill F2.

---



---

## ITER 1 (2026-05-04 ~01:30 UTC)

### Excursion-forgetting (§4 boundaries L428) — ANOMALY, no edit

Ran `excursion_analyze_v2.py` on 4-cond NPZs (foveated_logpolar skipped due to
script's hardcoded CONDS list). Numbers:

| Condition | Forget (rec-warm m/s) | Paper L428 (stale) |
|---|---:|---:|
| coarse (matched in script) | +0.198 | +0.43 |
| foveated | **+0.210** | +0.34 |
| uniform | +0.167 | +0.31 |
| foveated_logpolar | (missing) | n/a |
| blind | (no own ckpt yet) | +0.17 |

**Anomaly**: paper claims `coarse > foveated > uniform`, my data has
`foveated ≳ coarse > uniform` (very small foveated > coarse gap of ~0.012,
within plausible single-seed noise). Magnitudes are roughly half of paper's
stale values.

**Hypotheses for direction shift coarse↔foveated**:
1. New retrain hyperparams (seed=0, num_envs=16 unified) vs paper's stale runs
   (inconsistent hyperparam era). The shift is plausibly explained by retrain.
2. Single-seed noise. Gap of 0.012 is tiny vs ~0.04+ inter-condition spread.
3. v2 metric edge case at the magnitudes we're seeing.

**Magnitude shift (all conditions ~half of paper)**:
- Plausibly explained by tighter convergence under unified retrain
- Or by deterministic action selection in collect.py (paper's old data may have
  been stochastic for some lens)

**Decision**: do NOT update paper L428 yet. Wait until:
- foveated_logpolar added to script CONDS list and analyzed (gives a 4th data
  point informing whether logpolar lands closer to coarse or foveated)
- Phase B linear-probe results land — cross-check excursion direction against
  R² ranking. If paper's H1 (coarse > foveated for linear R²) holds, then
  excursion direction reversal is more concerning. If H1 also shifts, then
  the whole post-retrain picture is recalibrating consistently.
- Own dh-blind retrain done — adds 5th data point.

### Shortcut SPL benefit (§4 consumption L405) — direction matches, magnitude shifted

| Condition | reset_spl | persist_spl | spl_benefit | Paper L405 (stale) |
|---|---:|---:|---:|---|
| coarse | 0.7920 | 0.6985 | -0.0935 | (range -0.10 to -0.14) |
| foveated | 0.8585 | 0.8203 | **-0.0382** (least negative) | |
| uniform | 0.8835 | 0.7827 | **-0.1009** (most negative) | |
| foveated_logpolar | 0.8294 | 0.7684 | -0.0610 | n/a |

n=200 episodes/cond (paper had n=150, slight protocol shift).

**Direction**: ✓ all 4 conditions show negative ΔSPL (memory hurts when
re-initialised vs zero-init). Paper's narrative survives.

**Magnitude**: range -0.04 to -0.10 (paper had -0.10 to -0.14). Slightly less
memory-hurt across the board.

**Key signal for L407 framing**: uniform has the MOST negative spl_benefit
(-0.10), consistent with paper's "uniform locks onto old goal" reading
(persistent memory hurts most in uniform). foveated has the LEAST negative
(-0.04), suggesting foveated's memory is most "scene-generic" or least
sticky to the previous goal.

**Decision on edit**: candidate to update paper L405 once Phase B confirms
the H1 ranking — wait for ranking confirmation before edit (consistency
across lenses).

### Transplant (§4 H2 L321) — only sighted-only pairs available, partial

Completed:
- foveated → foveated_logpolar: baseline SPL 0.834, cross SPL 0.715, **delta -0.119**
- foveated → uniform: baseline SPL 0.849, cross SPL ?? (need to extract full)

Both pairs are within the rich-encoder regime (no bottleneck-rich asymmetry
testable without coarse or blind). The 6 coarse-pairs were killed earlier due
to RGB resolution mismatch in transplant.py (48×48 vs 256×256 → state_dict
shape error).

**Decision**: Do NOT update paper Figure 4 / L321 from this batch. The H2
transplant asymmetry claim requires bottleneck-on-one-side pairs. After
own dh-blind ckpt ready (~05-04 evening), retry.

### Phase B (Linear/MLP/CKA/Procrustes/lag-k) — still Pending GPU

`probe-analysis-b-0-0` still 0/1 Pending after ~3h. Cluster contention from
the 16 substitution-dynamics probes + 6 transplant pairs + dh-blind retrain
all queued for GPU. Will check next iter.

### Substitution dynamics — 1/16 Running, rest Pending

`probe-1-c10` (coarse ckpt.10) just started Running. Other 15 Pending.
Will accumulate slowly over next iters.

---

## State for next iter
- Phase B Pending → next-iter check + retry if still Pending after another iter
- 4 transplant pairs Running, 2 done — wait for next batch
- Substitution dynamics will slowly trickle in
- Excursion 3/4 done; need to fix CONDS list for foveated_logpolar
- Shortcut: 4/4 done, holding edit pending Phase B

## Open todos
- [ ] Edit excursion_analyze_v2.py to add foveated_logpolar to CONDS list
- [ ] Re-run excursion analysis on 4 conds
- [ ] Wait Phase B → cross-check direction with excursion before any L428 edit
- [ ] Wait ~6 substitution-dynamics conditions complete before any L260 update
