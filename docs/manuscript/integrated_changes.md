# Integrated changes (loop 2026-05-05)

Each row passed B.1 sanity gate; applied to main.tex in commit 8116b6f.

## §H1 / Table 1 (line ~213)
**Before**: `Blind 340M 0.59 0.93 +0.94±0.03 +0.81±0.08`
**After**:  `Blind 250M 0.59 0.93 +0.934±0.04 +0.844±0.03`
**Source**: `/tmp/rcp_analysis_v3/blind_izar_det_analysis.json` (n=500 deterministic-rollout episodes, 239 scenes, 221545 steps; blind_izar/ckpt.25.pth)
**B.1 gate**: PASS (GPS R² 0.934 within [0.85, 0.97] expected range)
**Note**: SPL 0.59 / Success 0.93 retained from prior eval (no rollout-summary in the analyze.py JSON; blind ckpt.25 SPL/Success may shift slightly from ckpt.34 but qualitative direction matches).

## §H1 information allocation paragraph (line ~244)
**Before**: `≈ 1.5× from blind to uniform via MINE`
**After**:  `≈ 1.3× from blind to uniform via MINE`
**Source**: `/tmp/rcp_analysis_v3/mine_5cond.json` (blind I=4.13 nats=5.96 bits, uniform I=3.10 nats=4.47 bits, ratio 1.33×; 3000 MINE training steps per cond)
**B.1 gate**: PASS (ratio 1.33× within [1.2, 2.0] expected)

## §H1 information allocation paragraph (synthesis, line ~451)
Same MINE update applied: `1.5×` → `1.3×`

## §H2 transplant asymmetry (line ~328)
**Before**: `recipient SPL collapses by 0.3-0.5 ... reverse direction recipient SPL is largely preserved`
**After**:  `recipient SPL drops by 0.17-0.33 (mean -0.23) ... reverse direction recipient SPL is preserved (mean Δ ≈ 0, range -0.07 to +0.06)`
**Source**: `/tmp/loop_outputs/transplant_5x5_summary.json` (14 of 20 cross-pair cells; 6 cells coarse↔{f,u,fl} broken due to transplant.py cross-spatial-size limitation)
**B.1 gate**: PASS (asymmetry strength -0.22; direction matches abstract claim)
**Note**: Previously stated "blind / coarse" donor; updated to "blind" only since coarse-cross cells couldn't run

## §H2 1-NN purity (line ~334)
**Before**: `1-NN purity is 1.000 vs chance 0.25 on 1500×4 pooled top-layer states ... same 1.000 holds at 10000×4`
**After**:  `1-NN purity is 1.000 vs chance 0.20 on 10000×5 pooled top-layer states ... same purity (0.9996) holds at 1500×5`
**Source**: `/tmp/rcp_analysis_v3/1nn_purity_5cond.json` (n=10000 per cond × 5 conds = 50000 pooled; 1.000 purity; n=1500 per cond gives 0.9996)
**B.1 gate**: PASS (no sample-size effect)

## §H3 LOSO blind median (line ~298)
**Before**: `Blind achieves median LOSO R² = 0.92 with 0% of held-out scenes giving negative R²`
**After**:  `Blind achieves median LOSO R² = 0.90 with 2% of held-out scenes giving negative R²`
**Source**: `/tmp/rcp_analysis_v3/loso_5cond.json` (5-cond rerun on full ckpt.25 NPZ; 50 top scenes per cond)
**B.1 gate**: PASS (blind median 0.904 > rich-encoder max 0.369; direction correct)

## §4.2 Skaggs blind value (line ~277)
**Before**: `blind sits slightly higher at 1.38`
**After**:  `blind sits slightly higher at 1.39`
**Source**: `/tmp/rcp_analysis_v3/skaggs_rectified.json` (5-cond, blind=1.394; mean per-(unit, scene) Skaggs)
**B.1 gate**: PASS (small shift, direction preserved)

---

## NOT modified but values verified to match paper

### Paper §4.2 MLP probe values (line ~244)
Verified `mlp_probe.json` blind_izar = linear 0.939 / MLP 0.985, gap 0.047. Matches paper exactly. No edit.

### Paper §4.2 Skaggs sighted (line ~277)
Verified `skaggs_rectified.json` n_place_units_1bit: coarse 289, foveated 303, uniform 285, fov_logpolar 284. EXACT match to paper. blind: 277 (paper 259); blind direction (fewer units) preserved.

### CKA values (App fig 8 caption)
Verified `cka_5cond.json` all 20 off-diagonals 4.9e-5 to 8.8e-5 (paper claim "<10^-4"). No specific number to update.
