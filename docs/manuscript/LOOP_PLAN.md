# 8-hour overnight experimental loop

**Started**: 2026-05-05 (evening)
**Deadline**: 2026-05-06 (paper submission)
**Goal**: replace stale pre-retrain numbers in main.tex with verified post-retrain values; fix any abstract claim that breaks; do NOT auto-incorporate anything that fails verification.

---

## A. Experimental design

### A.1 What's being tested

The 4 sighted conditions retrained on RCP at unified hp (dh-probe-1..4, 250M frames each); blind kept on Izar (seed=100, num_envs=8, ~9.7M frames/ckpt × 35 ckpts = 340M total). Decision: use **blind ckpt.25 ≈ 252M frames** as the 250M-equivalent for hp consistency.

Hyperparameter consistency caveat: blind seed=100 / num_envs=8 vs sighted seed=0 / num_envs=16. Disclosed in §A.1 + limitations.

### A.2 What gets re-run

Twelve experiments. Sorted by load-bearingness (P0 = touches abstract; P1 = quantitative §H1/H2/H3 claims; P2 = §4.5/§4.6 specific numbers).

| ID | Experiment | Type | Inputs | Output | ETA |
|---|---|---|---|---|---|
| **P0-A** | 5×5 memory transplant matrix | eval | 5 ckpts at 250M | `transplant_5x5.json` | 8 cells × ~30min on 1 GPU each |
| **P0-B** | MINE 5-cond on h₂ NPZs | analysis-only (GPU) | 5 NPZs | `mine_5cond.json` + `fig_mine_capacity.pdf` | ~1h |
| **P0-C** | Cross-condition probe transfer | analysis-only (CPU) | 5 NPZs | `cross_transfer_5cond.json` | ~30 min |
| **P0-D** | Procrustes 5-cond complete | analysis-only (CPU) | 5 NPZs | `procrustes_5cond_v2.json` | ~30 min |
| **P1-E** | LOSO 5-cond | analysis-only (CPU) | 5 NPZs | `loso_5cond.json` + `fig_loso_cv.pdf` | ~1h |
| **P1-F** | Subspace divergence (PCA + cosines) | analysis-only (CPU) | 5 NPZs | `subspace_5cond.json` + `fig_subspace_divergence.pdf` | ~30 min |
| **P1-G** | Subspace evolution across training ckpts | analysis-only (CPU) | 5 conds × 4-5 ckpts NPZs | `subspace_evolution.json` + `fig_subspace_evolution.pdf` | ~1h |
| **P1-H** | Predictive horizon to k=50 | analysis-only (CPU) | 5 NPZs | `predictive_horizon.json` + fig | ~30 min |
| **P1-I** | 1-NN purity at 1500 + 10000 | analysis-only (CPU) | 5 NPZs | `1nn_purity.json` | ~10 min |
| **P1-J** | Place-unit count 99-shuffle null | analysis-only (CPU) | 5 NPZs (Skaggs already computed) | `place_unit_count.json` + update `fig_place_cells.pdf` | ~30 min |
| **P2-K** | Excursion-forgetting (Wijmans WJ-F v2) | eval | 5 ckpts | `excursion_5cond.json` + fig | 5 conds × ~1h on 1 GPU each |
| **P2-L** | Memory-init transplant | eval | 5 ckpts | `memory_init_5cond.json` | 5 conds × ~30 min |
| **P2-M** | GPS-sensor mid-rollout ablation | eval | 5 ckpts | `gps_ablation_5cond.json` | 5 conds × ~30 min |
| **P2-N** | Shortcut paired-traj (canonical) | eval | 5 ckpts | `shortcut_5cond.json` + `fig5_shortcut_canonical.pdf` | 5 conds × ~1-2h |
| **P2-O** | Foveation-vs-uniform table refresh | aggregator | uses outputs of above | Table fov-vs-uni | <10 min |

### A.3 Pre-launch: blind data on RCP

Sequence:
1. rsync Izar → local `/tmp/blind_xfer/` (770 MB, 35 ckpts + tb dir) — IN PROGRESS
2. `kubectl cp` → `blind-xfer-0-0:/scratch/wxu/habitat_checkpoints_rcp/blind_izar/`
3. Verify ckpt.25.pth on RCP
4. Probe-collect blind ckpt.25 → `blind_izar_det.npz` (overwrites current ckpt.34 NPZ; old ckpt.34 NPZ saved as `blind_izar_det_ckpt34_backup.npz`)
5. Probe-collect blind ckpt.5/10/15/20 for substitution dynamics — 4 jobs in parallel
6. Once ckpt.25 NPZ lands, all P0/P1 analysis-only jobs can fire concurrently.

Cluster script edits required:
- `submit_transplant_rcp.sh`: `blind` → `/scratch/wxu/habitat_checkpoints_rcp/blind_izar/ckpt.25.pth`
- `submit_shortcut_rcp.sh`: same
- `submit_excursion_rcp.sh`: same (currently points at `blind_seed_2_friend/ckpt.49.pth`)
- `submit_probe_collect_rcp.sh`: blind_izar default ckpt.34 → ckpt.25

---

## B. Verification / evaluation loop

### B.1 Per-experiment expected ranges (sanity gate)

For each experiment, before incorporating into paper, the result must pass these checks:

| Experiment | Claim direction | Numerical sanity | Hard fail (DO NOT incorporate; flag) |
|---|---|---|---|
| Table 1 GPS R² blind 250M | positive, ≥ 0.85 | std < 0.10 | < 0.7 or NaN/Inf |
| Lag-k blind to k=20 | R² ≥ 0.85 across all k ∈ {0,2,5,10,20} | monotonic-decay or flat | sudden drop > 0.2 |
| **5×5 transplant asymmetry** | bottleneck-donor → rich-recipient SPL drop > rich-donor → bottleneck-recipient | gap ≥ 0.1 SPL | symmetric (gap < 0.05) → **abstract claim at risk** |
| MINE blind nat-scale | 4–5 bits | blind > coarse > others | NaN; blind < coarse |
| MINE total range blind/uniform | ratio 1.2× – 2.0× | finite | ratio reversed (uniform > blind) |
| Place-unit count | blind has FEWER place units than sighted | difference ≥ 10 | reversed direction |
| Subspace principal angles | 84°–90° (near-orthogonal) | std < 3° | < 75° (subspaces share structure → H2 weakens) |
| Position-direction cosines | -0.1 to +0.1 | std < 0.05 | abs > 0.3 (positions correlated → H2 weakens) |
| Procrustes blind-coarse θ₁ | 1.0–1.2 (closest pair) | finite | rich-encoder pair tighter than bottleneck pair → re-derive narrative |
| LOSO blind median | ≥ 0.80 | < 5% scenes negative | < 0.5 → §H3 narrative breaks |
| LOSO sighted median | ≤ 0.40 | > 30% scenes negative | > 0.7 → §H3 narrative breaks |
| Excursion forgetting | blind smaller magnitude than coarse/sighted | ranking ordered | reversed (decd56f warning) |
| Shortcut SPL drop ranking | uniform ≥ foveated ≥ coarse ≥ blind | margins consistent | full reversal |
| GPS ablation success drop | all collapse to <15% | ranking secondary | sustained > 50% in any cond |
| Memory-init Δ-SPL | all 5 conds negative | range -0.05 to -0.20 | positive Δ in any cond |

### B.2 Verification gate logic (per experiment)

```
result = run_experiment(exp_id)
if NaN or Inf in result.numbers:
    flag(exp_id, "numerical instability"); skip incorporation
elif not within_expected_range(exp_id, result):
    diagnose(exp_id, result)  # see B.3
    if diagnostic explains divergence convincingly:
        flag(exp_id, "expected divergence: <reason>"); skip incorporation
        # paper text may need narrative rewrite — escalate to user
    else:
        flag(exp_id, "unexplained divergence; needs human review"); skip incorporation
else:
    integrate(exp_id, result)  # update main.tex line; regen fig
    log(exp_id, "incorporated")
```

### B.3 Diagnostic ladder when result diverges from expectation

When a result fails B.1's range check, do NOT immediately rewrite paper. Run this diagnostic ladder:

1. **Convergence check**: did the probe / decoder / regression converge? Inspect loss curve / R² per fold variance. If high std across folds (>0.3), result is unreliable.
2. **Selectivity control**: for probes, check Hewitt–Liang label-permutation gap. If real-permuted gap < 0.1, probe is fitting noise.
3. **Sample size check**: are the n_episodes / n_steps comparable to old paper? If much smaller, re-collect with more.
4. **Hyperparameter mismatch hypothesis**: blind is seed=100 / num_envs=8; sighted is seed=0 / num_envs=16. If blind specifically shows divergence, hp-mismatch is a candidate explanation.
5. **Stale NPZ check**: are we loading the right ckpt's NPZ? Verify `npz["ckpt"]` field or filename.
6. **Bug check**: is the analysis script using a function that was recently changed? Check git log for the script.
7. **Compare to last-known-good**: pull the same analysis from pre-retrain pre-bug-fix state, verify the script reproduces old number.

If steps 1–7 all pass and result still diverges → **the result is real and the paper narrative may need revision**. Flag for user; do NOT auto-rewrite paper narrative.

### B.4 Three categories of post-loop output

After 8h, write three docs:

1. **`loop_summary.md`**: chronological log of what ran, what landed, what failed. One-screen overview.
2. **`integrated_changes.md`**: list of paper edits applied. For each: line range in main.tex, old value, new value, the JSON/figure that backs it. So user can verify each paper edit.
3. **`flagged_for_user.md`**: experiments that did NOT pass verification gate. For each:
   - What was expected (with citation to paper line)
   - What was observed (with JSON path)
   - Diagnostic ladder steps 1–7 results
   - Hypothesised cause
   - Recommended next-step (re-run, narrative revision, accept-and-hedge)

---

## C. Cross-experiment consistency checks (logical errors)

Pre-launch self-audit. Each row must hold; if not, plan has logical error.

| # | Check | Status |
|---|---|---|
| C1 | Sighted ckpts at 250M (ckpt.49 of dh-probe-1..4) consistent with paper Table 1 "250M" | ✓ |
| C2 | Blind ckpt.25 (~252M) is 250M-equivalent vs ckpt.34 (340M) currently in paper | ✓ (but Table 1 row needs 340M → 250M update) |
| C3 | All 5 NPZs come from deterministic-rollout collect (deterministic=True) | ✓ verified default |
| C4 | Probe analysis uses 5-fold episode-level CV (not step-level) | ✓ standard in `analyze.py` |
| C5 | Substitution dynamics blind ckpts (5/10/15/20/25) ≈ 50/100/150/200/250M, comparable to sighted (10/20/30/40/49 = 50/100/150/200/250M) | ✓ |
| C6 | 5×5 transplant uses self-transplant control (donor/recipient same cond, different episodes) | ✓ existing scripts include this |
| C7 | Cross-condition probe transfer is symmetric in design (train A→test B AND train B→test A) | ✓ |
| C8 | LOSO uses scene-level held-out (not step-level), matching paper's "no remap vs global remap" framing | ✓ |
| C9 | Procrustes uses paired episode-mean h₂ (500 pairs) — same protocol as old paper | ✓ verified script |
| C10 | MINE uses Donsker-Varadhan dual; train until convergence; report nat→bit | ✓ existing script |
| C11 | Subspace divergence uses top-K PCA covering ~90% variance per cond — K may differ across cond | ✓ paper specifies K∈{8,9,10} |
| C12 | All eval jobs use deterministic (argmax) action sampling | ✓ default since c81352e |

If any of C1-C12 reveal an inconsistency mid-loop, halt and flag.

### C.13 The asymmetry claim (P0-A) — special attention

The abstract says "memory transplants are asymmetric (rich-encoder policies fail to use bottleneck states, while bottleneck policies still drive themselves from rich-encoder ones)". 

This requires:
- bottleneck-donor (blind, coarse) → rich-encoder-recipient (foveated, uniform, fov_logpolar): SPL drops a lot (recipient struggles)
- rich-encoder-donor (foveated, uniform, fov_logpolar) → bottleneck-recipient (blind, coarse): SPL drops less (recipient OK)

The decd56f commit (May 4) noted "6 sighted-only pairs roughly symmetric (rich-rich)" — symmetric between rich-encoder pairs, but the bottleneck-rich axis was NOT tested in the partial 6-cell run. This loop's P0-A is the first proper test of the abstract claim on retrained ckpts.

If P0-A shows the asymmetry: paper survives, integrate the new 5×5 numbers.
If P0-A shows symmetry: **abstract claim is at risk**. Do NOT auto-rewrite. Flag with full data, log diagnostic, escalate.

Specifically, the cells that test the asymmetry are:
- bottleneck → rich: blind→{fov,uni,fov_lp}, coarse→{fov,uni,fov_lp} (6 cells)
- rich → bottleneck: {fov,uni,fov_lp}→blind, {fov,uni,fov_lp}→coarse (6 cells)
- Plus 5 self-transplants (control floor)

Total: 17 cells (or 5×5 = 25 minus 5 same-pairs that are self = 20 cross + 5 self). Existing script supports this.

---

## D. Phase-by-phase execution

### Phase 0: Pre-launch (in progress, ~30-60 min)

- [ ] rsync Izar→local `/tmp/blind_xfer/` (RUNNING; ~641 MB / 770 MB at last check)
- [ ] kubectl cp `/tmp/blind_xfer/` → pod `blind-xfer-0-0:/scratch/wxu/habitat_checkpoints_rcp/blind_izar/`
- [ ] Verify ckpt.25.pth + tb dir on RCP
- [ ] Cleanup `/tmp/blind_xfer/` (per memory: no local persistence)
- [ ] Edit cluster scripts: blind path → `blind_izar/ckpt.25.pth`
- [ ] Submit blind probe-collect ckpt.25 → primary blind NPZ
- [ ] Submit blind probe-collect ckpt.{5,10,15,20} → substitution dynamics 4 jobs in parallel
- [ ] Kill blind-xfer pod once transfer verified

### Phase 1: Eval-time experiments (parallel, ~3-4h wall-clock with 4-8 GPU concurrency)

Launch order (to maximise GPU utilisation):
1. P0-A 5×5 transplant 17 cells (highest priority + highest GPU need) — launch immediately
2. P2-K excursion 5 conds — launch after first transplant cells finish (free GPU slots)
3. P2-L memory-init 5 conds — interleave
4. P2-M GPS ablation 5 conds — interleave
5. P2-N shortcut 5 conds — last (longest individual eval)

### Phase 2: Probe analyses (CPU, parallel, ~2h wall-clock)

Once blind ckpt.25 NPZ lands (~Phase 0 step 7):
- P0-B MINE 5-cond
- P0-C cross-condition probe transfer
- P0-D Procrustes 5-cond
- P1-E LOSO 5-cond
- P1-F subspace divergence
- P1-G subspace evolution (needs blind cross-training NPZs from Phase 0 step 8)
- P1-H predictive horizon
- P1-I 1-NN purity
- P1-J place-unit count

### Phase 3: Verification + integration (concurrent with Phase 1/2 results landing)

For each experiment that lands:
1. Run B.1 sanity gate
2. If pass → run integration (regen fig if applicable, edit main.tex line)
3. If fail → run B.3 diagnostic ladder, write to `flagged_for_user.md`

### Phase 4: Final pass (~30 min before user wakeup)

- Compile main.tex
- Verify zero compile errors
- Verify all `\toconfirm` / `\hcpending` / `\todoblue` are still empty (the "still clean" sanity check)
- Cleanup `*.aux *.bbl *.blg *.log *.out` per memory rule
- Write final `loop_summary.md` + `integrated_changes.md` + `flagged_for_user.md`
- git commit each integrated change with descriptive message

---

## E. Self-audit (before launch)

**E.1 Are all P0-P2 experiments listed in B.1?** Verify every P0-A through P2-O has a row.
- P0-A ✓ "5×5 transplant asymmetry"
- P0-B ✓ "MINE blind / range"
- P0-C ❌ missing → add: cross-condition probe transfer R² should be catastrophic negative (< -2) for all off-diagonals
- P0-D ❌ missing → add: Procrustes blind-coarse pair tightest among bottleneck-bottleneck
- P1-E ✓ LOSO
- P1-F ✓ subspace
- P1-G ❌ missing → add: subspace evolution principal angles already at >80° at earliest probed ckpt (~50M)
- P1-H ❌ missing → add: predictive horizon blind R² ≥ 0.85 to k=20, ≥ 0.5 at k=50
- P1-I ❌ missing → add: 1-NN purity ≥ 0.95 at both 1500 and 10000 pooled samples (no sample-size effect)
- P1-J ✓ place-unit count
- P2-K ✓ excursion
- P2-L ✓ memory-init
- P2-M ✓ GPS ablation
- P2-N ✓ shortcut
- P2-O — derived from above; no separate gate needed

**Resolution**: B.1 row table needs additions for P0-C, P0-D, P1-G, P1-H, P1-I before launch. Will add inline below.

**E.2 Are there any double-spending claims?** I.e., does some result feed two paper sections that need different verification?
- §H1 magnitude axis uses Table 1 GPS R² + lag-k. Same data, two sections; no conflict.
- §H2 format axis uses CKA + Procrustes + transplant + 1-NN purity + cross-condition probe. Five views, all need to point same direction. If 4/5 align and 1 dissents → flag.
- §H3 format axis uses LOSO. Single source. Direct check.
- §4.5 dissociation uses shortcut SPL drop + GPS ablation. Two views; if disagree, flag.

**E.3 What if blind ckpt.25 NPZ has anomalies?** (e.g., fewer episodes than 500, shorter mean episode length). Resolution: run analyze.py with same `--min-steps-scene 15` as elsewhere; check `n_episodes`, `n_steps`, `n_scenes` match canonical (500 / ~200k / 25–60).

**E.4 What if Phase 0 step 1 (rsync) fails partway?** Resolution: rsync supports resume via `-P`; just rerun. Already doing.

**E.5 What if RCP queue is full and jobs sit Pending?** Resolution: most analyses are CPU-only; can run inside a single sleeper pod sequentially. Eval jobs need GPU but can interleave.

---

## F. Update B.1 with missing rows (resolved E.1)

Adding to B.1 sanity gate:

| Experiment | Claim direction | Numerical sanity | Hard fail |
|---|---|---|---|
| P0-C cross-condition probe transfer | catastrophic off-diagonal | all 20 off-diag R² < -1 | any off-diag > 0 → H2 transferable, paper false |
| P0-D Procrustes 5-cond blind-coarse pair tightest | θ₁(blind,coarse) < θ₁(blind,foveated) and < θ₁(blind,uniform) | gaps ≥ 0.05 | reverse ordering → re-derive narrative |
| P1-G subspace evolution | angles ≥ 80° at earliest ckpt (~50M); cosines near zero throughout | Δangle 50M→250M < 10° | angles < 70° at 50M → "established early" claim weakens |
| P1-H predictive horizon | blind R² to k=50 ≥ 0.5; sighted decays to ~0 by k=20 | monotonic decay | reversed (sighted predicts longer) |
| P1-I 1-NN purity scaling | both 1500 and 10000 give ≥ 0.95 | identical at 2 sizes | drops at 10000 → sample-size effect, paper claim weakens |

---

## G. What if the deadline arrives before everything finishes?

**Triage if at T-3h still incomplete**:
- P0 must land OR be flagged → highest priority
- P1 should land; if not, retain stale numbers WITH "single-seed; values from prior pre-retrain run" footnote (worst case)
- P2 nice-to-have; OK to leave existing numbers if pre-retrain values already match expected direction

The paper currently compiles cleanly with all `\toconfirm` markers cleared. Worst-case fallback is the current state of main.tex.

---

End of plan. Self-audit complete; no logical inconsistencies blocking launch. Proceeding to Phase 0.
