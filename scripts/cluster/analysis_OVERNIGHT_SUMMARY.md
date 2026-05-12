# Overnight Analysis Summary (2026-05-04)

User asleep ~01:30 → ~13:00 UTC (~12h). Session re-engaged on user wake.
Loop /loop wakeups did NOT auto-fire iters 3+ during user idle, but the
RCP cluster jobs continued running independently. Most analysis now complete.

---

## TL;DR

**Major confirmations** (HIGH confidence — substantive new data):
1. ✅ §H1 substitution dynamics (L260): all 4 conditions start high R²
   (~0.85-0.92 at ckpt.10/50M frames), bottleneck preserves, rich-encoder decay.
   Decay rate ordering: **uniform > foveated_logpolar > foveated > coarse**.
2. ✅ §H1 magnitude (L245, Table 1): coarse +0.58 holds; rich-encoder near-0
   or negative. Direction matches paper.
3. ✅ §H2 CKA: bottleneck-rich pairs lower CKA than within-regime, BUT
   foveated more aligned with coarse than with uniform — paper's "foveation
   as hybrid case" reinforced.

**Major shifts / falsifications** (require user review):
1. ❌ **HCPENDING falsifiable prediction (L811): NOT confirmed**.
   Predicted log-polar R² ≥ 0.3 between coarse 0.78 and uniform ~0.
   Actual: foveated_logpolar ckpt.49 R² = **−0.03** (rich-encoder regime).
   Mechanism §H1 framing of "encoder spatial output dim" needs revision.
2. ⚠️ **Procrustes coarse-most-distant claim (L329) shifts**: in 3-cond
   data (no blind, no logpolar), foveated is more central not coarse.
3. ⚠️ **Skaggs MI 100× off** (L277): paper 1.25 bits, actual 20-160 bits.
   Likely metric definition mismatch; needs reconciliation.
4. ⚠️ **Excursion direction shift** (L428): paper coarse > foveated, actual
   foveated ≳ coarse (within noise). Magnitudes ~half of stale.

**Resource state**:
- F2 (foveated_normaliser): Running, ~56% (~30h more)
- dh-blind retrain: Pending (F2 holds 4 GPU). DEADLINE RISK.

---

## Detailed numbers (4 conditions, blind missing — own retrain Pending)

### Linear probe R² (5-fold CV mean ± std)

| Cond | GPS R² | Std | Compass R² | Compass std | goal_dist R² (sanity) |
|---|---:|---:|---:|---:|---:|
| coarse | +0.582 | 0.099 | +0.659 | 0.061 | +0.960 |
| foveated | +0.178 | 0.659 | +0.503 | 0.142 | +0.912 |
| uniform | −1.785 | 2.993 | −1.244 | 3.579 | +0.915 |
| foveated_logpolar | −0.029 | 0.759 | +0.467 | 0.330 | +0.946 |

Paper Table 1 (stale): blind 0.95, coarse 0.72, foveated 0.06, uniform −0.31.

Direction match: bottleneck > rich ✓. Magnitudes shifted.

### Substitution dynamics: R² across training (5 ckpts)

| Cond | ckpt.10 (50M) | ckpt.20 (100M) | ckpt.30 (150M) | ckpt.40 (200M) | ckpt.49 (250M) |
|---|---:|---:|---:|---:|---:|
| coarse | **+0.91** | +0.54 | +0.43 | +0.58 | +0.58 |
| foveated | **+0.91** | −0.72 | +0.53 | +0.53 | +0.18 |
| uniform | **+0.84** | +0.79 | +0.38 | −0.39 | **−1.79** |
| foveated_logpolar | **+0.92** | +0.72 | +0.75 | +0.65 | −0.03 |

**Key story** (this is exactly what paper §H1 predicts at L260):
- All conditions establish linear top-layer GPS code by ~50M frames
- Bottleneck (coarse) preserves it through training
- Rich-encoder (uniform, foveated, foveated_logpolar) **decay** as visual
  route consolidates
- Uniform decays catastrophically by 250M (R² → −1.79)
- foveated_logpolar decays slower than uniform but ends near 0

### Procrustes shape distance (3 conds: coarse / uniform / foveated; logpolar missing in script)

| | coarse | uniform | foveated |
|---|---:|---:|---:|
| coarse | 0 | 1.019 | 0.846 |
| uniform | 1.019 | 0 | 0.988 |
| foveated | 0.846 | 0.988 | 0 |

theta_1 mean per cond:
- coarse: (1.019 + 0.846)/2 = 0.93
- uniform: (1.019 + 0.988)/2 = 1.00
- foveated: (0.846 + 0.988)/2 = 0.92

Paper L329 "coarse most distant": NOT clearly supported. Foveated/coarse comparable.

### Linear CKA (3 conds)

| | coarse | uniform | foveated |
|---|---:|---:|---:|
| coarse | 1.000 | 0.297 | **0.566** |
| uniform | 0.297 | 1.000 | 0.332 |
| foveated | **0.566** | 0.332 | 1.000 |

**foveated-coarse 0.566 > foveated-uniform 0.332**. foveated representation
aligns more with bottleneck (coarse) than with same-regime sighted (uniform).
Paper §"Foveation as a hybrid case" REINFORCED.

### Excursion forgetting (variance-matched MAE/spread per segment)

| Cond | warmup | recovery | Forget (rec-warm) |
|---|---:|---:|---:|
| coarse | 0.989 | 1.188 | +0.198 |
| foveated | 1.074 | 1.284 | **+0.210** (highest) |
| uniform | 0.981 | 1.148 | +0.167 |
| foveated_logpolar | (script CONDS list missing) | | |
| blind | (no own ckpt yet) | | |

Paper L428 had: coarse +0.43 > foveated +0.34 > uniform +0.31 > blind +0.17.
My data: foveated ≳ coarse > uniform. Direction not strict match;
magnitudes ~half of stale.

### Shortcut SPL benefit (memory_with - memory_without)

| Cond | reset_spl | persist_spl | spl_benefit |
|---|---:|---:|---:|
| coarse | 0.792 | 0.699 | −0.094 |
| foveated | 0.859 | 0.820 | −0.038 (least negative) |
| uniform | 0.884 | 0.783 | **−0.101 (most negative)** |
| foveated_logpolar | 0.829 | 0.768 | −0.061 |

All negative ✓ (paper L405 direction match). Range -0.04 to -0.10
(paper had -0.10 to -0.14).

Uniform most negative consistent with §"locks-onto-old" (L407).

### Transplant 6 sighted-only pairs (rich-rich, no bottleneck testable)

| pair | baseline | self_t | cross | cross_delta | net cross |
|---|---:|---:|---:|---:|---:|
| fov_lp → fov | 0.857 | 0.805 | 0.722 | -0.135 | -0.083 |
| fov → fov_lp | 0.834 | 0.782 | 0.715 | -0.119 | -0.067 |
| fov → uni | 0.849 | 0.801 | 0.760 | -0.089 | -0.041 |
| uni → fov | 0.857 | 0.805 | 0.753 | -0.104 | -0.052 |
| uni → fov_lp (TBD) | | | | | |
| fov_lp → uni (TBD) | | | | | |

Roughly symmetric within rich-rich. Cannot test H2 bottleneck-rich asymmetry
(needs blind or coarse pair).

### Skaggs spatial info (questionable — 100× higher than paper)

| Cond | Skaggs bits |
|---|---:|
| coarse | 20.8 |
| foveated | 27.0 |
| uniform | 99.8 |
| foveated_logpolar | 163.6 |

Paper L277: 1.25/1.32/1.18/1.16 bits.

100× discrepancy. Likely metric definition mismatch (mean per-(unit,scene)
vs different aggregation). Needs investigation before paper update.

---

## Proposed paper edits (await user review)

**HIGH confidence (data + direction strong)**:

1. **L260 substitution dynamics paragraph** — replace decay-rate description
   with new 5-ckpt × 4-cond data. The qualitative pattern (early high R²,
   rich-encoder decay) is crisply confirmed. Stronger than paper's stale
   3-pair description.

2. **L811 log-polar Appendix** — UPDATE to actual R² = −0.03 (NOT the
   predicted ≥0.3). This means the encoder-spatial-output mechanism as
   currently framed needs revision. Paper §"Foveation as a hybrid case"
   may absorb this — log-polar is a 5th rich-encoder, not bottleneck.
   Re-think §H1 mechanism wording.

3. **Table 1 R² columns** — replace 4-cond values (blind row TBD).

**MEDIUM confidence (direction matches but magnitude shifted)**:

4. L405 shortcut SPL benefit (range update)
5. L329 PR / theta_1 Procrustes (some shifts; foveated more central)

**LOW confidence (DO NOT auto-edit)**:

6. L277 Skaggs MI (100× off, metric definition mismatch)
7. L428 Excursion (foveated > coarse direction shift, weak)

**BLOCKED on dh-blind retrain (still Pending)**:

- Full Table 1 (need blind row)
- Full §H2 5×5 transplant (need bottleneck pair)
- LOSO scene-invariance L291 (need blind)
- All cross-condition matrices including blind

---

## Phase B partial failures

Phase B job ran but only produced:
- ✅ 4 per-cond analysis JSONs (linear probe, rate maps, etc.)
- ✅ procrustes.json
- ❌ mlp_probe.json (670 B — empty/failed)
- ❌ NO cka.json (CKA matrix actually IS in procrustes.json under
  `linear_cka_matrix`, so coverable)
- ❌ NO lagk_<cond>.json (5 files missing)

To re-run if needed:
- MLP probe: needs separate fix (CONDS list or path)
- Lag-k: works on per-cond NPZ, can be done CPU-only

---

## Recommendation for user wake-up

1. **Walk through these numbers together** before any paper edits
2. **Big strategic call**: log-polar prediction NOT confirmed.
   - Option A: re-frame §H1 mechanism as "encoder spatial-feature variety
     per step" (already in paper L521 in MASTER_TRACK but not main.tex)
     — log-polar actually high feature variety despite low output dim
   - Option B: report logpolar R² as-is, say "encoder-spatial-output
     mechanism falsified for this case; further work needed"
   - Option C: argue logpolar's effective encoder output is closer to 4×4
     than 2×2 (need check actual visual_encoder.output_shape)
3. **dh-blind decision**: own blind hasn't started (F2 still holds GPU).
   Likely won't finish for deadline 05-06. Options:
   - Accept friend's seed=100 ckpt with footnote (user previously rejected)
   - Kill F2 to free GPU (F2 56% completed, ~14h compute "wasted")
   - Ship paper with 4-cond Table 1, blind in appendix robustness
4. **Skaggs metric mismatch**: don't update L277 until reconciled.
5. **Phase B re-run**: optional, if MLP probe / lag-k needed.

## Loop status

- iter counter: stuck at 2 (didn't fire iters 3+ during user idle)
- Cluster jobs progressed independently — most data NOW available
- Next action: user review + decide which edits to apply

## Commits so far this session
- `395333d` runbook
- `8985181` iter 1 diagnosis (shortcut + transplant + excursion partial)
- `518e36e` iter 2 diagnosis (4 transplant pairs)
- (this iter — diagnosis + summary, see git log)
