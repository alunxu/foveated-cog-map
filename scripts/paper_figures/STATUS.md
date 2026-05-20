# Paper figure script index

Map of `make_*.py` → submission paper figure. Keep this in sync when figures move.

## Main text (Fig 1–5, Fig 9)

| Paper figure                           | Script                                    | Notes                                                    |
|----------------------------------------|-------------------------------------------|----------------------------------------------------------|
| `fig1.pdf` (Setup pipeline)            | hand-composed in Keynote                  | Source: `fig1_setup_*.pdf` from `make_setup_pipeline.py` |
| `fig2_magnitude.pdf` (Magnitude)       | `make_magnitude_3panel.py`                | Default output is `fig_magnitude.pdf`; rename to `fig2_magnitude.pdf` after generation |
| `fig3_format_axis.pdf` (Format)        | `make_format_2panel.py`                   |                                                          |
| `fig4_temporal.pdf` (Temporal)         | `make_temporal_maps_figure.py`            |                                                          |
| `fig5_consumption.pdf` (Consumption)   | `make_consumption_2panel.py`              | `make_consumption_3panel.py` is the legacy 3-panel version |
| `fig9_world_model_probe.pdf` (Memory Maze DINOv2) | `make_encoder_feature_probe_figure.py` |                                                       |

## Appendix figures

| Paper figure                            | Script                                            |
|-----------------------------------------|---------------------------------------------------|
| `figa7a_transplant_5x5.pdf`             | `make_5x5_transplant_matrix.py`                   |
| `figa7b_h2_probe_transfer.pdf`          | `make_h2_probe_transfer.py`                       |
| `figa9a_per_unit_info.pdf`              | `render_5cond_appendix.py`                        |
| `figa9b_sparse_decoding.pdf`            | `render_5cond_appendix.py`                        |
| `figa10_goal_vector.pdf`                | `make_goal_vector_figure.py`                      |
| `figa13_pc_cumulative.pdf`              | `render_5cond_appendix.py`                        |
| `figa15_eigenspectrum.pdf`              | `render_5cond_appendix.py` (or sibling appendix renderer) |
| `figa17_shortcut_catalog.pdf`           | `make_shortcut_paired_trajectory_figure.py`       |
| `figa18_transplant_sweep.pdf`           | `make_transplant_sweep_figure.py`                 |

## Helpers (used by other scripts, not directly emitting a paper figure)

- `_style.py` — apply_paper_style() seaborn/matplotlib styling shared by every figure script
- `make_action_icons.py`, `make_motivation_icons.py` — pre-rendered SVG/PNG glyphs used in Fig 1 composite

## Legacy / experiment scratch (not in current paper)

These scripts produced figures from earlier drafts or one-off progress reports. They are kept for git history and possible revival; do **not** rely on them for the current paper.

- `make_bottleneck_figure.py` — pre-paper bottleneck figure
- `make_capacity_allocation_figure.py`, `make_capacity_allocation_v2.py` — early capacity-allocation framing figure (subsumed by Fig 2)
- `make_consumption_3panel.py` — legacy 3-panel version of Fig 5 (current uses 2-panel)
- `make_h1_mega_figure.py`, `make_h1h2_figures.py` — pre-canonical-ordering (H1/H2 naming, replaced by `fig2_magnitude` / `fig3_format_axis`)
- `make_information_allocation_figure.py`, `make_information_allocation_v2.py` — early information-allocation framing
- `make_pipeline_overview.py`, `make_setup_pipeline.py`, `make_setup_panels.py` — pre-Keynote attempts at Fig 1
- `make_probe_agent_figure.py` — legacy Fig 5b (replaced by 2-panel Fig 5)
- `make_progress_report_figures.py`, `make_progress_slides.py` — one-off progress-report assets
- `make_scrubbing_figure.py` — scrubbing now lives inside Fig 5c (`make_consumption_2panel.py`)
- `make_shortcut_canonical_figure.py`, `make_shortcut_figure.py`, `make_shortcut_scatter.py` — pre-canonical shortcut variants (current shortcut figures live in Fig 5 + `figa17`)
- `make_substitution_figure.py` — substitution-dynamics figure, deferred to follow-up
- `make_synthesis_figure.py` — pre-paper Fig 8 synthesis (the message is now distributed across Fig 1, §6 Discussion)
- `make_temporal_figure.py`, `make_temporal_probe_figure.py` — pre-`make_temporal_maps_figure.py`
- `make_trajectory_overlay_figure.py`, `make_training_dynamics_figure.py` — exploratory helpers, not in current paper
- `make_additional_figures.py` — mixed scratch
- `make_embedding_figures.py` — t-SNE / manifold figures, deferred
- `make_extra_states_figure.py` — figa3 extra-states (currently rolled into figa9)
- `make_foveation_strength_figure.py` — foveation σ sweep, deferred to follow-up
- `make_layerwise_figure.py`, `make_memory_length_figure.py`, `make_mp3d_generalization_figure.py`, `make_occupancy_decoder_figure.py`, `make_subspace_evolution_figure.py`, `make_temporal_probe_figure.py`, `make_training_curves.py`, `make_transplant_figure.py`, `make_tsne_figure.py`, `make_wjf_figure.py` — appendix/exploratory candidates whose outputs were not folded into the current paper

## How to regenerate the main-text figures

```bash
# Fig 2 (Magnitude)
python scripts/paper_figures/make_magnitude_3panel.py
mv docs/manuscript/fig/fig_magnitude.pdf docs/manuscript/fig/fig2_magnitude.pdf

# Fig 3 (Format)
python scripts/paper_figures/make_format_2panel.py

# Fig 4 (Temporal)
python scripts/paper_figures/make_temporal_maps_figure.py

# Fig 5 (Consumption)
python scripts/paper_figures/make_consumption_2panel.py
```

All four read from `/tmp/rcp_analysis/*_det_analysis.json` (probing summaries) and `results/shortcut_results/*_traj.{json,npz}` (shortcut trajectories). Probing summaries are produced by running the corresponding cluster jobs in `scripts/cluster/` (Izar SLURM) or `scripts/cluster_rcp/` (private RCP). Shortcut trajectories are produced by `scripts/eval/shortcut_with_trajectories.py`.
