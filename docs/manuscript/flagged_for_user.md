# Flagged for user review (loop 2026-05-05)

Updated 2026-05-05 after D1 + E1 follow-up runs. Some items resolved; one new interpretive call requires your judgement.

---

## ⚠️ ACTIVE — needs your judgement

### E1 excursion-forgetting: numbers reversed; narrative flipped

**What changed**: `excursion_analyze_v2.py` re-run with all 5 conditions (incl. blind which had been collected as exc-5 but never aggregated). Result is a clean reversal of paper §4.6's headline:

| Cond | Paper (pre-loop) | Post-loop |
|---|---|---|
| blind | +0.17 (smallest) | **+0.381 (LARGEST)** |
| coarse | +0.43 (largest) | +0.198 |
| foveated | +0.34 | +0.210 |
| uniform | +0.31 | +0.167 |
| foveated_logpolar | (not reported) | +0.135 (smallest) |

**Old interpretation** (rejected by new data): blind's hidden state is closer to GPS-passthrough → recovery error stays near warmup error.
**New interpretation** (committed dda5d8a, wrapped in `\uncertain{}` + pendnote): integrated codes are action-stream-dependent and drift under random-action detour; blind is the most heavily integrated (per §H1: highest top-layer GPS, longest horizon) so it suffers most. Sighted conditions re-anchor on visual landmarks.

**This interpretation IS consistent with the rest of the paper** (it actually unifies §H1's "blind has the most integrated GPS code" with §4.6's "blind decoheres most under random-action interruption"). But the flip is large enough that you should sanity-check the paragraph's framing against the rest of the paper.

**Source**: `wjf_v2_summary.json` (re-aggregated; blind exc-5 was Completed at end of loop, blind_det.npz symlinked to blind_excursion.npz; CONDS list updated to include foveated_logpolar). 5-fold episode-level CV, n=100 episodes/cond, MAE/spread variance-matched metric.

**Possible alternative explanations to rule out**:
1. Different statistical artefact in the v2 metric for blind specifically? Check that the warmup MAE/spread isn't anomalous (current data: blind warmup 0.960, sighted 0.98–1.07 — comparable, no anomaly).
2. ckpt.25 (250M) vs ckpt.34 (340M) — could blind at 250M still be in a different regime? The hp-consistency switch was the right move; flipped excursion result reflects the unified protocol.
3. A bug in excursion_forgetting.py — the data is the same NPZ used pre-loop for sighted, just with blind added; sighted numbers shifted slightly too (e.g. coarse +0.43→+0.20) which suggests a methodology refinement.

→ I recommend **keeping the new narrative** with the `\uncertain{}` flag. Reverting to the old reading would be inconsistent with §H1's strong-integration story for blind.

---

## ✅ RESOLVED in commit dda5d8a

### D1 predictive horizon — re-run with mean-center protocol

Was: blind k=20 R²=0.733 (paper claims ≥0.94). Diagnosis: my horizon script used `StandardScaler` while Table 1 standard probe doesn't. Plus subsampling 200 eps vs 500.

**Resolution**: re-ran `loop_horizon_v2.py` with mean-center only + ALL 500 episodes. blind k=0=0.944 (matches Table 1 0.94), k=20=0.903, k=50=0.794 (matches paper's 0.79). Updated paper to "[0.90, 0.94] from k=0 to k=20 (a 0.04-point drop)" and corrected foveated text.

---

## ✅ RESOLVED in commit 2b02b9d

### Subspace divergence — angles 69-87° (paper said 86-89°)

Resolution: paper text + figure caption updated to "[69°, 87°] (most pairs in [77°, 87°])" with note that closest pair (foveated--foveated-logpolar 69°) is the same-family rich-encoder pair. K_per_cond updated from "8-10" to "[4, 33] varies per cond". Direction (near-orthogonal) preserved qualitatively.

### Transplant 6 cells missing — footnote added

Resolution: pendnote added to §H2 transplant paragraph disclosing that coarse↔{foveated, foveated-logpolar, uniform} cells (6 of 20) were excluded due to differing visual input size (48x48 vs 256x256) being incompatible with our shared-environment transplant pipeline. Asymmetry test relies on blind axis only.

### Cross-cond probe transfer asymmetry — replaced with concrete numbers

Resolution: §H2 "Convergent evidence at the probe level" paragraph rewritten with concrete row-asymmetry: blind-row off-diagonals R² ∈ [-0.28, -0.15] (mild), other rows R² ∈ [-6.0, -1.2] (catastrophic). This is a **new finding** from the loop that mirrors the transplant asymmetry direction and strengthens H2.

---

## NOT-yet-applied / deferred

### Place-unit count 99-shuffle null
Still running on RCP (loop-analysis-2 pod). Paper §4.2 already uses simpler `>1bit` threshold from skaggs_rectified.json which IS verified. The 99-shuffle JSON would replace these with more rigorous numbers but isn't on the critical path.

### Subspace evolution across training
Blind cross-training NPZs (c10/15/20) were still in flight at end of loop. Existing `fig_subspace_evolution.pdf` uses pre-retrain data. If user wants to refresh: re-run `run_subspace_evolution.py` once those NPZs land + regenerate the figure.

### GPS-sensor ablation
Paper §4.5 already heavily hedged ("consistent-but-not-conclusive support"); old data retained. Defer.
