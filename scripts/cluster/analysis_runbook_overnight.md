# Overnight Analysis Loop Runbook (2026-05-04)

**For autonomous /loop iteration. Re-read on each wake-up.**

User instructions (verbatim):
1. Design experimental loop, ensure no inconsistency / logical error before executing
2. When results arrive, do NOT be over-optimistic — dive deep to figure out why / if results make sense / consistent with expectations
3. When VERY sure, incorporate into paper

Deadline: NeurIPS Neurosci/CogSci track, 2026-05-06 evening (~64h from start of loop). User asleep 8h.

---

## 1. Current state inventory (start of loop, 2026-05-04 ~01:30 UTC)

### Training (RCP, dhlab-wxu namespace)
- **dh-fnorm** (F2 = foveated + RunningMeanAndVar): Running, ~55%, ~30h ETA
- **dh-blind** (own blind retrain, seed=0, num_envs=16): Pending (was Running, preempted) — ~17h once it gets GPU

### Phase A — probe-collect 500 deterministic episodes per condition (h2 NPZ)
- ✅ coarse_det.npz (997 MB)
- ✅ foveated_det.npz (765 MB)
- ✅ uniform_det.npz (~520 MB)
- ✅ foveated_logpolar_det.npz (994 MB)
- ❌ blind: dropped, await own dh-blind retrain (friend's seed=100 ckpt is inconsistent hyperparam)

### Phase B — analysis lenses (single GPU pod)
- ⏳ probe-analysis-b: Pending (had bugs: sklearn missing, scripts not synced, unaligned_cka --in-dir wrong arg). Fix applied + resubmitted.

### Phase B/C parallel batch (4 conditions, no blind)
- ✅ Excursion (WJ-F): 4/4 NPZ Completed in `/scratch/wxu/habitat_checkpoints_rcp/excursion_results/`
- 🟡 Shortcut (paired-traj): 3/4 done, 1 Running. JSONs in `/scratch/wxu/habitat_checkpoints_rcp/shortcut_results/`
- 🟡 Transplant: 6 sighted-only pairs (foveated/uniform/foveated_logpolar permutations) running/queued. **6 coarse-pairs killed** (incompatible 48×256 RGB → state_dict mismatch). Output: `/scratch/wxu/habitat_checkpoints_rcp/transplant_results/<donor>_to_<recipient>_mid30.json`
- 🟡 Substitution dynamics: 16 cross-ckpt probes (4 conds × c10/c20/c30/c40) Pending GPU. Output: `<cond>_det_ckpt<N>.npz` in `probing_data_rcp/`

---

## 2. Verification rules per experiment

For each result, BEFORE any paper edit, run these checks. If ANY fails, do NOT edit paper. Log to `scripts/cluster/analysis_diagnosis.md` and continue.

### 2.1 Linear probe R² (analyze.py output JSON)
Expected (paper §H1, post-retrain expected to preserve direction).
**Without blind in this batch (own dh-blind retrain pending), 4-cond ranking:**
- coarse > foveated_logpolar > foveated, uniform (linear top-layer GPS R²)
- coarse R² > 0.5 (paper had 0.78 with old hyperparam). If < 0.3 → BUG.
- foveated/uniform R² < 0.4 (paper had ~0.06 / -1.08). If > 0.5 → BUG.
- foveated_logpolar R² is THE FALSIFIABLE PREDICTION from paper §appendix L811:
  - Predicted ≥ 0.3 (between coarse and uniform/foveated). HCPENDING in paper.
  - If R² ≥ 0.3: encoder-spatial-output mechanism CONFIRMED. De-hedge \hcpending L397/803/811.
  - If R² < 0.1: mechanism FALSIFIED. Need to re-frame H1 in paper. Flag major.
- 5-fold CV std < 1.0 for bottleneck conditions (tight spread)

Fields to extract:
```
data['gps']['r2']               # mean R²
data['gps']['r2_std']           # std across folds
data['compass']['r2']           # compass R²
data['goal_distance']['r2']     # goal-distance sanity probe (should be ~0.85 across all)
```

### 2.2 MLP probe R² (run_mlp_probe_proper.py)
Expected:
- **All MLP R² > linear R²**. If MLP < linear → BUG (probe shouldn't be worse than linear).
- Gap (MLP - linear) larger for rich-encoder than bottleneck.
- Paper had: blind +0.95, coarse +0.81, fov +0.62, uniform +0.48 (range 0.5)
- Direction: sighted MLP gap > bottleneck MLP gap (info present non-linearly)

### 2.3 CKA matrix (unaligned_cka.py)
Expected:
- **Diagonal = 1.0** (CKA(X,X)=1 by definition). If off-diagonal > 0.95 → suspicious.
- Bottleneck-bottleneck pair (blind/coarse) higher than cross-regime
- Rich-encoder pair (foveated/uniform) higher than cross-regime
- Cross-regime pairs (e.g. blind/foveated) lower CKA

### 2.4 Procrustes (procrustes_shape_analyze.py)
Expected:
- **Coarse most-distant** from others on d_1 (paper: ~8.8k vs ~4k among the rest)
- PR ranking: foveated highest, coarse lowest (paper: foveated 4.34, coarse 3.10)
- TwoNN-ID ranking: coarse lowest (paper: 1.64 vs 1.77-1.85)
- d_1 should be POSITIVE distances (no negatives)

### 2.5 Lag-k probe (lagk_all_targets.py)
Expected:
- Blind: stable high R² across k=0..20+ (≥ 0.92 throughout in paper)
- Coarse: moderate-high stable (≥ 0.50)
- Rich-encoder (fov, uniform): unstable, often negative at small k

### 2.6 Excursion-forgetting (excursion NPZ → compute MAE/spread per segment)
**Need to write small analysis script. Logic:**
1. Load NPZ, get `segments` array (0=warmup, 1=detour, 2=recovery)
2. For each cond + each segment: train Ridge probe on full-episode pooled hidden states (5-fold CV)
3. Compute MAE / position-spread per segment (variance-matched)
4. Forgetting = recovery_error - warmup_error

Expected ranking (paper L428):
- Coarse: +0.43 (most forgetting)
- Foveated: +0.34
- Uniform: +0.31
- Blind: +0.17 (least forgetting — closest to passthrough)

**Without blind, expected ranking: coarse > foveated > uniform.** All POSITIVE.

### 2.7 Shortcut paired-traj margin
- Output: `<cond>_traj.json` has per-episode margin (m to old-goal vs new-goal terminal position)
- Expected (paper L407):
  - **Uniform: positive margin (locks-onto-old, +1.83m)** — unique pattern
  - Blind: small negative margin (-0.38m)
  - Coarse, foveated: between

Without blind, key claim: **uniform has positive margin, others negative or near 0**.

### 2.8 Transplant matrix (PARTIAL — sighted-only)
**Critical scope reduction:** transplant.py current architecture builds ONE shared env
from donor config; donor's sim_sensors size must match recipient ckpt's expected
input. coarse uses 48×48 RGB while foveated/uniform/foveated_logpolar all use 256×256
→ 6 coarse-pairs killed (state_dict shape mismatch). 6 sighted-only pairs remain
(permutations of {foveated, uniform, foveated_logpolar}).

This means **H2 transplant asymmetry claim CANNOT be fully tested in this batch**:
the original paper finding "bottleneck-donor → rich-recipient SPL collapses; reverse
preserves SPL" requires a bottleneck (blind or coarse) on at least one side.

What we CAN test from sighted-only 6 pairs:
- Within-rich-encoder transplant cost (foveated ↔ uniform — both rich-regime,
  expected mild drop)
- foveated_logpolar's transplant compatibility with foveated/uniform — informs
  whether log-polar is "more like rich-encoder" or carves a new regime

VERIFY each pair JSON: `mean_recipient_spl` in [0, 1], `mean_self_spl_baseline`
similar magnitude, recipient SPL drop computed correctly.

**Paper edit caveat:** do NOT update full Figure 4 5×5 matrix from this batch.
Document partial result in `analysis_diagnosis.md`. Re-run with own blind ckpt +
fix transplant.py scaling to support cross-resolution donor/recipient.

### 2.9 Substitution dynamics (cross-ckpt probe)
- For each (cond, ckpt N ∈ {10,20,30,40}): probe-collect → NPZ
- Once NPZ exists for all 4 ckpts of a condition, run analyze.py on each → R² trajectory
- Expected (paper L260):
  - Bottleneck (blind, coarse): R² stable high across all ckpts
  - Rich-encoder (fov, uniform): early (~50M) R² high, then **decays**
  - Decay-rate ordering: uniform fastest, foveated slower

Without blind: just coarse (stable) vs foveated/uniform (decay).

---

## 3. Paper edit map

For each `\toconfirm{...}` in `docs/manuscript/main.tex`, the corresponding data source:

| L# | Stale value | Replace from |
|---|---|---|
| 201 | "Sighted 96-99% success, blind 93%, Bug 0.07" | Phase A run.log success rate (already in Phase A NPZs) + later blind |
| 209 | "GPS R² ∈ [0.6, 0.95] mid-episode" | analyze.py per-step bin output |
| 209 | "(foveated 0.06±0.88, uniform -0.31±0.86, t-test)" | analyze.py R² mean/std per cond |
| 215-218 | Table 1 (per-cond SPL/success/GPS R²/compass R²) | analyze.py + Phase A run.log |
| 245 | "Linear R²: blind 0.95, coarse 0.72, fov 0.25, uni -1.08" | analyze.py (5 cond if blind) |
| 245 | "MLP R²: blind 0.95, coarse 0.81, fov 0.62, uni 0.48" | run_mlp_probe_proper.py |
| 260 | "uniform left tight-pos by 100M, fov +0.78 at 100M, crash 150M" | substitution dynamics analyze.py per-ckpt |
| 277 | "Skaggs MI: 1.25/1.32/1.18/1.16; place-units: 174/116/237/116" | NEEDS NEW SCRIPT (place-cell signature) |
| 286 | "blind R² ≥ 0.72 k=0-20, +0.56 k=50; coarse [0.46, 0.58]" | lagk_all_targets.py |
| 291 | "blind LOSO 0.92, coarse 0.17, fov 0.28, uniform 0.15" | LOSO test (NEEDS impl or analyze_h3.py) |
| 329 | "PR: foveated 4.34, blind 4.10; coarse 3.10" | procrustes_shape_analyze.py |
| 329 | Procrustes d_1 / theta_1 | procrustes_shape_analyze.py |
| 407 | "uniform margin +1.83m; blind -0.38m" | shortcut_with_trajectories.py JSON |
| 428 | "blind +0.17, uniform +0.31, fov +0.34, coarse +0.43" | excursion analyze (needs to write script) |
| 811 | "log-polar 60M R² = 0.808" | analyze.py on foveated_logpolar_det.npz (already have!) |

**Edit protocol:**
1. Find the line in main.tex
2. Replace stale value INSIDE the `\toconfirm{...}` wrapper (KEEP wrapper for user's review pass)
3. Add a `% UPDATED <DATE>: source=<filename>` comment near the change
4. Compile sanity-check: `cd docs/manuscript && pdflatex -interaction=batchmode -draftmode main.tex 2>&1 | tail -3` (must report no errors)
5. Cleanup aux/log/etc per memory rule

**Do NOT edit unless ALL verification checks pass for that experiment.**

---

## 4. Per-iteration protocol (each ~25 min wake-up)

Step 1: **Snapshot cluster state**
```bash
kubectl get pods -n runai-dhlab-wxu --no-headers | grep -E "^probe-|^exc-|^sc-|^tp-|dh-blind|dh-fnorm|probe-analysis"
```

Step 2: **Identify newly Completed since last iteration**
- Compare current Completed list to `/tmp/runbook-state.json` (init empty if missing)
- Update state file at end of iteration

Step 3: **For each new result, pull + verify**
- Pull JSON or NPZ via `kubectl exec ... -- cat` or `kubectl cp` to a /tmp dir
- Run verification rules from §2
- If pass: stage edit
- If fail: append to `scripts/cluster/analysis_diagnosis.md`, do NOT edit paper

Step 4: **Apply staged paper edits**
- Edit main.tex with values, keeping `\toconfirm{}` wrappers
- pdflatex draftmode compile
- Commit with descriptive message: `paper: update L<N> from <experiment>`

Step 5: **Schedule next iteration**
- Use ScheduleWakeup with delay 1500s (25 min)
- pass `<<autonomous-loop-dynamic>>` sentinel as prompt
- Update `/tmp/runbook-state.json` with snapshot

**Stop conditions (any one halts the loop):**
- All Phase B/C results collected AND paper updated for verified items
- **Iteration counter ≥ 20** (hard cap — read/write `/tmp/runbook-iter-count`)
- **Wall time ≥ 8h** (loop start 2026-05-04 01:30 UTC; stop ≥ 09:30 UTC)
- Critical failure (e.g., kubectl unreachable for >30 min, > 5 consecutive iterations with no progress)
- Severe verification failure on a major finding (e.g., H1 ranking violated) — STOP and write
  detailed diagnosis to `analysis_diagnosis.md` for user review on wake-up

On stop: write summary to `scripts/cluster/analysis_OVERNIGHT_SUMMARY.md`:
- What was verified + edited into paper (with commit SHAs)
- What was collected but NOT edited (and why)
- What's still in-flight (still Pending / Running)
- Diagnosis log entries
- Next-action recommendation for user

---

## 5. Failure modes + responses

| Failure | Response |
|---|---|
| Phase B Pending > 2h | Try restart job (delete + resubmit). If still Pending after 4h, doc + skip. |
| Pod Errors with new traceback | kubectl logs, append diagnosis to `analysis_diagnosis.md`, kill job, continue |
| Result NPZ unreadable / corrupt | Skip that result, log to diagnosis |
| Verification rule fails (sign/magnitude wrong) | Do NOT edit paper. Investigate + log. |
| Cluster network unreachable | Skip iteration, continue at next wake-up |
| Cross-method inconsistency (e.g., linear says A>B but Procrustes says A=B) | Edit paper only with the result that's verified; flag inconsistency in diagnosis |

---

## 6. Things explicitly NOT in scope

- F2 ckpt analysis (will finish ~05-05, not in 8h window)
- Own dh-blind ckpt analysis (will finish ~05-04 evening, may overlap end of 8h)
- Place-cell signature (L277) — needs NEW script to write, defer until verified rest works
- LOSO scene-invariance (L291) — analyze_h3.py exists, need to verify it runs with our NPZs
- MINE I(h2;pos) — defer
- Occupancy decoder per-cond — defer
- Stale-results notice strip — DEFER until all updates done + user review

---

## 7. Tools / paths cheat-sheet

- Pod for kubectl exec: `kubectl get pods -n runai-dhlab-wxu --no-headers | awk '$3=="Running" && $1~"dh-fnorm"{print $1; exit}'`
- Activate env: `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate habitat`
- PYTHONPATH: `/scratch/wxu/dh-spatial`
- USER: must be set to `wxu` for habitat_env.py
- Results dirs: `/scratch/wxu/habitat_checkpoints_rcp/{analysis_results,excursion_results,shortcut_results,transplant_results,probing_data_rcp}/`
- Paper: `docs/manuscript/main.tex`
- Diagnosis log: `scripts/cluster/analysis_diagnosis.md`
- State file: `/tmp/runbook-state.json`
