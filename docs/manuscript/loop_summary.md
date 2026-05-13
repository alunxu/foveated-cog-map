# Loop summary 2026-05-05 (overnight 8h cycle)

## Top-level outcome

- **Blind 250M-equivalent (ckpt.25, ~252M frames) probe-collected** for hp consistency with 4 sighted retrains
- **14 of 20 cross-pair transplant cells re-run** on retrained ckpts; **asymmetry direction confirmed**
- **8 post-blind analyses re-run on full ckpt.25 NPZ**: LOSO, MINE, subspace divergence, predictive horizon, 1-NN purity, cross-cond probe transfer, MLP probe, Skaggs (place-unit-count 99-shuffle null still running at end)
- **6 verified results integrated into main.tex** (commit 8116b6f); 2 results flagged for user review (subspace angles wider than paper claimed; predictive-horizon blind k=20 below paper threshold)
- **Paper compiles clean at 46 pages**

## Key data shifts

| Section | Paper (pre-loop) | Post-loop | Direction match? |
|---|---|---|---|
| Table 1 blind row | 340M, GPS 0.94±0.03 | 250M, GPS 0.934±0.04 | ✓ |
| §H1 MINE ratio | 1.5× | 1.3× | ✓ (still gradual) |
| §H2 transplant bn→rich drop | 0.3–0.5 | 0.17–0.33 (mean -0.23) | ✓ direction; magnitude tightened |
| §H2 transplant rich→bn drop | "largely preserved" | -0.07 to +0.06 (mean ≈0) | ✓ direction confirmed |
| §H2 1-NN purity | 1.000 / chance 0.25 (4 conds) | 1.000 / chance 0.20 (5 conds) | ✓ |
| §H2 subspace angles | 86°–89° | 69°–87° (mostly 77–87) | partial: near-orthogonal qualitatively, but range wider |
| §H3 LOSO blind | 0.92 / 0% neg | 0.90 / 2% neg | ✓ |
| §4.2 Skaggs blind | 1.38 | 1.39 | ✓ |

## Timeline

- ~04:30 UTC: Phase 0 starts (Izar SSH refresh, blind ckpt copy planning)
- 05:00 UTC: blind probe-collect ckpt.25 launched, 14 transplant cells launched
- 05:24 UTC: probe writes first checkpoint (ep 50, 232 MB partial NPZ); auto_fire_v1 mistakenly triggers on partial data
- 05:26 UTC: auto_fire_v1 killed; auto_fire_v2 launched waiting for COLLECT_DONE marker
- 05:35 UTC: cluster scripts updated for blind 250M-equivalent path; transplant.py donor-blind env-config bug fixed
- 06:18-08:31 UTC: 14 transplant cells land sequentially; bn→rich asymmetry trend visible from cell 4 onwards
- 08:31 UTC: blind ckpt.25 NPZ complete (3.06 GB, 221k steps, 239 scenes)
- 08:35 UTC: COLLECT_DONE detected manually (probe inner_cmd's `echo COLLECT_DONE` went to stdout but not log file; appended marker manually)
- 08:35–08:50 UTC: auto_fire_v2 ran 7 analyses sequentially: LOSO, subspace_div, cross-cond transfer, 1-NN purity, predictive horizon, MINE, place-unit count
- 08:46 UTC: separate jobs (mlp/lagk/skaggs/post-blind-lenses) complete
- 08:50 UTC: subspace_div re-run with mean-center-only (matched paper protocol)
- ~08:55 UTC: paper edits applied via apply_paper_edits.py + manual edits
- ~08:55 UTC: main.tex compiles clean; commit 8116b6f

## What ran

| Job category | Count | Status |
|---|---|---|
| Blind ckpt copy (Izar→RCP) | 35 ckpts + 4 tb events | ✓ Done |
| Probe-collect (5 conds × 1-5 ckpts) | 1 primary + 4 cross-training (c5 killed for being too slow, c10/15/20 in flight) | Primary ✓; cross-training partial |
| Transplant cells (5×5 - 5 self) | 20 attempted, 14 valid landed; 6 broken (cross-spatial-size: coarse↔{f,u,fl}) killed | 14/20 ✓; 6 broken |
| Excursion 5 conds | 4 sighted pre-existed; blind exc-5 Completed | ✓ |
| Shortcut 5 conds | 4 sighted pre-existed; blind sc-5 Completed | ✓ |
| Post-blind analyses (auto_fire_v2) | 7 ran (LOSO, subspace, cross, 1nn, horizon, MINE, place-unit count) | ✓ + place-unit running |
| Separate analyses (mlp/lagk/skaggs/post-blind-lenses) | 4 submitted via run_post_blind_analyses.sh | ✓ |

## Bugs found and resolutions

1. **transplant.py env-config**: when donor=blind, env was being built from donor's no-RGB config, recipient policy then loaded as blind, ckpt-load failed on compression-layer shape. **Fix**: detect blind via config-name, use recipient env when donor is blind. (Worked for blind→{coarse, fov, uni, fov_lp})
2. **transplant.py cross-spatial-size**: coarse↔{fov, uni, fov_lp} requires two envs (coarse 48×48 vs sighted 256×256) since shared env's obs space dictates compression-layer shape. **Status**: NOT fixed; 6 cells killed; partial 14-cell 5×5 used instead
3. **collect.py "COLLECT_DONE" marker**: the bash inner_cmd `echo COLLECT_DONE` writes to stdout but NOT to the .log file (only python's tee writes there), so my auto_fire_v2 grep on log missed it. **Workaround**: manually appended marker after probe Completed
4. **Subspace-divergence StandardScaler**: my loop_5cond_subspace.py used StandardScaler which flattens variance distribution, inflating K and shrinking principal angles. **Fix**: re-ran with mean-center only (paper convention); K dropped from 96-135 → 4-33, angles improved 60-67° → 69-87°

## Outputs

- `/tmp/rcp_analysis_v3/` — all JSONs from loop (61 files)
- `/tmp/loop_outputs/` — verify gate + integrated_changes.md + flagged_for_user.md
- `docs/manuscript/main.tex` — 6 verified edits applied (commit 8116b6f)
- `docs/manuscript/loop_summary.md` (this file)
- `docs/manuscript/integrated_changes.md` — paper-edit log
- `docs/manuscript/flagged_for_user.md` — divergent results

## Deferred (out of 8h scope)

- Place-unit count 99-shuffle null (still running; paper §4.2 uses simpler >1bit threshold from skaggs_rectified.json which IS up to date)
- Subspace evolution across training ckpts (blind cross-training NPZs c10/15/20 still in flight; c5 killed; can land later as supplementary fig)
- GPS-sensor ablation rerun (paper §4.5 already heavily hedged with old data; defer)
- 6 broken transplant cells (coarse↔{f,u,fl}); requires transplant.py refactor to support two envs
- Memory-init transplant rerun (no eval script; defer)
