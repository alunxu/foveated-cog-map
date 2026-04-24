# Project status — high-level snapshot

Updated 2026-04-24.

## What we found wrong

1. **Stochastic sampling in probe collection** (`collect.py` hardcoded
   `deterministic=False`, now `True`). Caused policies with higher
   action-entropy (fov-fix, uniform, matched, blind under stochastic)
   to STOP around step 4, producing quasi-static 4-step rollouts with
   near-zero target variance → any probe fit trivially → inflated R²
   across all conditions.

2. **World-frame GPS/compass as probe target**. Both are per-step
   observations the policy reads directly, so LSTM has no reason to
   re-encode them. The rich-encoder conditions (uniform, foveated,
   fov-learned) decode at chance under det; only encoder-bottlenecked
   conditions (blind, matched) encode them. Ego-relative **DtG** is
   the robust, universal target across all conditions.

## What claims were affected

| Old claim | Status | New claim |
|-----------|--------|-----------|
| H1: foveated > uniform in compensatory memory | ❌ reversed | **H1 (new): visual-encoder bottleneck → LSTM spatial compensation. Blind (GPS R²=0.95) and matched (0.78) develop cognitive-map-like encoding; rich-encoder conditions (uniform, foveated, fov-lrn) are at chance.** |
| "More pixels ≠ better decoding" hook (foveated 0.84 vs uniform 0.61) | ❌ both at chance | Replaced by stronger hook: matched (1-pixel-effective) 0.78 vs uniform (full RGB) chance. |
| H2: format-level divergence (CKA, transplant) | ✅ survives | Untouched. CKA/transplant are behavioural, not probe-dependent. |
| H3: fov-learned compass R²=0.94 | ❌ bug artefact | H3 downgraded to "pending fov-shifted convergence"; no current claim. |
| Wijmans blind-agent replication | ✅ strengthened | Blind GPS R²=0.95 replicates + extends via matched (new data point). |

## Data re-collected (done ✓)

5-condition deterministic-probe data + 5-fold CV analysis, all committed:

- `blind_gibson_det.npz` + analysis
- `matched_gibson_det.npz` + analysis
- `uniform_gibson_det.npz` + analysis
- `foveated_gibson_det.npz` + analysis (paper canonical: pre-corruption ckpt.36)
- `foveated_learned_gibson_det.npz` + analysis

## Paper changes (done ✓)

- Abstract, Intro H1 statement, Methods §3.3, Table 1 with CV error bars,
  §4.3 H1 complete rewrite, §4.4 H2 caveat trim, §4.5 H3 placeholder,
  Discussion "ruling out alternatives" + implications — all rewritten
  around the bottleneck framing.
- New hero figure: `fig/h1_bottleneck.pdf` (3-panel bar chart, GPS /
  compass / DtG × 5 conditions, CV error bars, bottleneck vs
  pass-through brackets).
- `main.pdf` compiles, 25 pages.

## What we're doing now (Izar, running)

| Job | What | Elapsed | Purpose |
|-----|------|---------|---------|
| `fov-shifted` | Training | ~18h / 72h | Clean H3 causal test: same foveation transform, different static gaze → does gaze location alter LSTM content? |
| `fov_s2`, `uni_s2`, `fov-lrn_s1`, `fov-lrn_s3` | Training | ~24h / 72h | Multi-seed error bars on H1 ordering |

## What we're waiting for (and why)

| Waiting on | Why it matters | Action when ready |
|-----------|----------------|-------------------|
| `fov-shifted` training | Only clean causal test for H3 (gaze-location variable isolated; same architecture as foveated-fixed) | Probe + compare to foveated_det + foveated_learned_det; populate §4.5 H3 section. |
| Multi-seed trainings (4 running on Izar) | Replace single-seed Table 1 numbers with mean±σ across seeds; tightens the headline "blind/matched >> others" claim | Re-probe + update Table 1 error bars + hero figure. |
| Friend's `encoder_capacity_scaling` | Turn 2-point bottleneck observation (blind, matched-48) into 7-point scaling curve. Adds quantitative scaling-law evidence. | Generate scaling curve figure → paper appendix. |
| Friend's `multiseed_robustness` gap (6 runs) | Fill the seeds Izar isn't covering (blind ×2, uniform s3, foveated s3, matched ×2) | Combine with Izar seeds for 3-seed error bars on all 5 conditions. |

## Not currently blocking, shelved

- `training_dynamics_observation.md`: fresh-restart (undertrained) vs
  pre-corruption fov-fix shows different LSTM encoding — potential
  training-dynamics support for the bottleneck mechanism. Single seed,
  large CV variance → needs multi-seed or intra-run ckpt sweep before
  paper-worthy. Shelved for appendix if corroborated.

## Next decision points

1. When fov-shifted trains: decide H3 section content (bottleneck
   framework makes H3 narrower: within the pass-through regime, does
   gaze location shift *which* non-spatial content the LSTM holds?).
2. When multi-seed data is in: update Table 1 + hero figure, freeze
   the main-text numbers.
3. When friend's scaling data arrives: draft appendix figure + text,
   integrate into main paper pitch.

Nothing else currently blocking forward motion on the paper.
