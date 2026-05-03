# Overnight Loop Diagnosis Log

Last updated: 2026-05-04 01:35 UTC (iter 1)

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
