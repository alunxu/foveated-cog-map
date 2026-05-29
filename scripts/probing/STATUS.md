# Probing script index

Cross-references generated from `git grep` across paper figures, cluster
scripts, and the manuscript. Run from the repo root:
`python3 -c "import re,os..."` cross-reference (74 total scripts).

Tier definitions:

- **Tier 1 — Core pipeline**: data-collection / canonical probe battery that *every* figure depends on
- **Tier 2 — Paper figure dependency**: referenced by a `scripts/paper_figures/make_*.py` or by `main.tex` directly
- **Tier 3 — Submission pipeline**: referenced by `scripts/cluster/*.sh` (Izar SLURM) — still in active probing workflow
- **Tier 4 — Working notes only**: referenced by docs/manuscript scratch files (`LOOP_PLAN.md`, `SLEEP_LOG.md`, `MASTER_TRACK.md`, sample/cogneuro_*) but not by paper figures or cluster scripts. Likely exploratory.
- **Tier 5 — Unreferenced**: not cited anywhere in the codebase; legacy / experiment scratch

---

## Tier 1 — Core pipeline (paper-critical)

| Script                              | Role                                                                       |
|-------------------------------------|----------------------------------------------------------------------------|
| `collect.py`                        | Hidden-state collection: rollout PointNav episodes, dump `h_t` + GPS + meta to NPZ |
| `analyze.py`                        | Canonical probing battery: Ridge GPS, MLP, Hewitt–Liang controls, layer-wise, distance-to-goal |

These two are the entry points for every probe-driven figure in the paper.

---

## Tier 2 — Paper figure dependency

### Main text

| Script                                       | Used by                                              | Paper figure |
|----------------------------------------------|------------------------------------------------------|--------------|
| `extra/compute_5cond_appendix.py`            | `paper_figures/render_5cond_appendix.py`             | Appendix figa9a/9b/13 (population coding battery) |
| `analyze_encoder_features.py`                | `paper_figures/make_encoder_feature_probe_figure.py` | Fig 9 (encoder-feature probe) |
| `extra/subspace_scrubbing.py`                | `paper_figures/make_consumption_2panel.py`           | Fig 5c (β-axis scrubbing) |
| `probe_depth_sweep.py`                       | `paper_figures/make_format_2panel.py`                | Fig 3 (depth-dependent decoding) |
| `extra/probe_transfer_5x5.py`                | `paper_figures/make_h2_probe_transfer.py`            | Appendix figa7b |
| `extra/analyze_extra_states.py`              | `paper_figures/make_extra_states_figure.py`          | (legacy)     |

### Cited in `main.tex` body

- `extra/predictive_horizon.py` (Appendix F predictive-horizon table)
- `extra/intrinsic_timescales.py` (Fig 4b unit-level τ violins; also `sample/cogneuro_frameworks/timescales_murray.md` background)
- `extra/leave_one_scene_out.py` (Fig 3b LOSO scene-invariance probe)
- `extra/temporal_generalisation.py`, `extra/temporal_generalisation_v2.py` (Fig 4a TGM)

---

## Tier 3 — Submission pipeline (Izar SLURM)

Referenced by `scripts/cluster/submit_*.sh`. These are still part of the
active probing pipeline, just orchestrated from the cluster side.

| Script                              | Submission script                                                |
|-------------------------------------|------------------------------------------------------------------|
| `analyze_cross.py`                  | `submit_cross.sh`, `regenerate_det_figures.sh`                  |
| `analyze_h3.py`                     | `submit_h3.sh`, `rerun_analyses_on_det.sh`                      |
| `analyze_legacy.py`                 | `submit_probe.sh`, `submit_probe_mig.sh`, `submit_probe_seeded.sh` |
| `collect_encoder_features.py`       | `submit_encoder_features.sh`                                    |
| `compute_scene_occupancy.py`        | `submit_occupancy_decoder.sh`                                   |
| `extended_lag_probe.py`             | `rerun_analyses_on_det.sh`                                      |
| `goal_vector_probe.py`              | `regenerate_det_figures.sh`, `rerun_analyses_on_det.sh`         |
| `length_matched_probe.py`           | `submit_length_matched.sh`                                      |
| `mlp_probe_sanity.py`               | `submit_mlp_probe.sh`, `submit_mlp_sanity.sh`                   |
| `population_coding_analysis.py`     | `submit_population_coding.sh`                                   |
| `temporal_probe.py`                 | `submit_temporal_probe.sh`                                      |
| `train_occupancy_decoder.py`        | `submit_occupancy_decoder.sh`                                   |
| `unaligned_cka.py`                  | `regenerate_det_figures.sh`, `rerun_analyses_on_det.sh`         |
| `extra/lagk_all_targets.py`         | (RCP only — `scripts/cluster_rcp/submit_phase_b_analysis.sh`)   |
| `run_mlp_probe_proper.py`           | (RCP only — `scripts/cluster_rcp/submit_phase_b_analysis.sh`)   |
| `visualize.py`                      | `scripts/eval/shortcut_with_trajectories.py`                    |

---

## Tier 4 — Working notes only (exploratory)

Referenced by `docs/manuscript/SLEEP_LOG.md`, `LOOP_PLAN.md`,
`MASTER_TRACK.md`, or `sample/cogneuro_*/` background notes — but NOT by
any paper figure, cluster submission, or `main.tex` body. Kept for git
history but inactive in the current paper pipeline.

- `cluster_quality.py`
- `skaggs_rectified.py`
- `run_subspace_evolution.py`
- `extra/ccgp_abstraction.py`, `extra/plot_ccgp.py`
- `extra/tangling.py`, `extra/plot_tangling.py`
- `extra/plot_tgm.py`, `extra/plot_timescales.py`
- `extra/make_cross_cond_figure.py`, `extra/pc_cumulative_analysis.py`
- `extra/per_scene_beta_stability.py`, `extra/position_axis_analysis.py`
- `extra/persistent_homology.py`, `extra/place_cell_v2.py`
- `extra/predictive_coding.py`, `extra/splitter_cells.py`
- `extra/run_100M_anchor.py`, `extra/run_cross_cond_transfer.py`
- `extra/run_twonn_id.py`

---

## Tier 5 — Unreferenced (legacy / experiment scratch)

Not cited anywhere in the codebase. Kept for git history.

| Script                                 | Likely purpose (from filename / first comment)                              |
|----------------------------------------|------------------------------------------------------------------------------|
| `analyze_subspace_divergence.py`       | Superseded by `extra/subspace_scrubbing.py`                                  |
| `compare_det.py`                       | Comparison of deterministic vs stochastic rollouts                           |
| `confidence_probe.py`                  | Confidence-of-position probe (exploratory)                                   |
| `fov_probe_diagnosis.py`               | Diagnostic for foveated probe (pre-canonical)                                |
| `masked_heading_probe.py`              | Heading-masked variant probe                                                 |
| `per_step_progression.py`              | Per-step probing accuracy curves (exploratory)                               |
| `probe_alternative_targets.py`         | Probing alternative regression targets                                       |
| `probe_cv_summary.py`                  | Cross-validation summary helper (exploratory)                                |
| `probe_deep_diagnostic.py`             | Deep-probe diagnostic (exploratory)                                          |
| `probe_scaler_test.py`                 | Probe-scaler ablation test                                                   |
| `extra/eigenspectrum_powerlaw.py`      | Powerlaw fit on PC eigenspectrum                                             |
| `extra/find_convergence.py`            | Training-convergence diagnostic                                              |
| `extra/fisher_information.py`          | Fisher-information geometry                                                  |
| `extra/make_convergence_diagnostic.py` | Plotter for find_convergence                                                 |
| `extra/make_twonn_figure.py`           | 2-NN intrinsic-dim figure                                                    |
| `extra/mine_capacity_accounting.py`    | MINE-based memory capacity accounting                                        |
| `extra/plot_behavioral_interventions.py` | Behavioural-intervention plot                                              |
| `extra/plot_module4_consumption.py`    | Pre-paper module-4 plot                                                      |
| `extra/plot_persistence.py`, `extra/plot_persistence_split.py` | Persistence diagram plots                             |
| `extra/plot_predcode.py`               | Predictive-coding plot                                                       |
| `extra/plot_splitter.py`               | Splitter-cell plot                                                           |
| `extra/run_extra_states_goalvec.py`    | Extra-states goal-vector runner (pre-canonical)                              |
| `extra/skaggs_proper_v3.py`, `extra/skaggs_remapping_analysis.py` | Skaggs spatial-info ablations                          |

---

## How the pipeline flows

```
                       trained ckpt.49.pth (per condition)
                                │
                                ▼
                        scripts/probing/collect.py
                                │  → probing_data/<cond>_det_ckpt49.npz
                                ▼
                        scripts/probing/analyze.py
                                │  → <cond>_det_analysis.json
                                ▼
   ┌─────────────────────┬──────┴──────┬─────────────────────┐
   ▼                     ▼             ▼                     ▼
extra/subspace_scrubbing  extra/predictive_horizon          extra/probe_transfer_5x5
extra/leave_one_scene_out extra/temporal_generalisation_*
extra/intrinsic_timescales extra/compute_5cond_appendix
                                │
                                ▼
                         results/probing_results/*.json
                                │
                                ▼
                         scripts/paper_figures/make_*.py
                                │
                                ▼
                         docs/manuscript/fig/*.pdf
```
