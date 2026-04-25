# Master Track — NeurIPS 2026 submission

Single source of truth for: cluster jobs, experiment status, paper
claims, figure freshness, open questions, decision log.

**Last updated**: 2026-04-25 (commit `be369f5`).
Update this file when state changes — do NOT rely on memory.

---

## 0. Quick status (at-a-glance)

| Dimension | Status |
|---|---|
| Paper version | v1 in progress; main story stable; integration of latest experiments pending |
| Page count | 28 pages (NeurIPS limit 9 main + unlimited appendix) |
| Critical TODO in paper | Table 1 fov-lrn `\TODO{H3}`, App D scaling figure `\TODO{Pending data}`, §4.4 H3 paragraph (training in progress) |
| Cluster jobs running | 2 (fov_s2 + uni_s2 multi-seed training) |
| Cluster jobs pending | ~50 (F1-F4 + 5×5 transplant + per-step transplant + training dynamics + encoder features + pop coding) |
| Most recent finding | Temporal probe shows H1 is about TEMPORAL STABILITY of GPS code, not its existence (commit c904e8c) |
| Most recent paper change | Integrated temporal probe finding into §1, §4.1, §4.2, §5.1, §6 (commit 6408d6c) |

---

## 1. Cluster jobs

### 1.1 Running (verify with `ssh izar "squeue -u wxu"`)

| Job ID | Name | Started | Expected | Output |
|---|---|---|---|---|
| 2844326 | uni_s2 multi-seed | 2026-04-23 | ~2026-04-26 | `/scratch/izar/wxu/habitat_checkpoints/uniform_gibson_seed2/` |
| 2844327 | fov_s2 multi-seed | 2026-04-23 | ~2026-04-26 | `/scratch/izar/wxu/habitat_checkpoints/foveated_gibson_seed2/` |

### 1.2 Pending (in priority order)

| Batch | Jobs | Total | Per-job | Source script |
|---|---|---|---|---|
| F1-F4 foveation strength | 4 (2849136-9) | ~50-70h each | 250M frames training | `submit_train.sh + 4 configs` |
| 5×5 cross-condition transplant | 17 | ~1.5h each | midpoint=30 | `submit_5x5_transplant.sh` (A) |
| Per-step transplant extension | 12 | ~1.5h each | midpoint={200,400,800} × 4 pairs | `submit_5x5_transplant.sh` (H) |
| Training-dynamics probes | 22 | ~3h each | 5 conds × 4-5 ckpts | `submit_training_dynamics.sh` (J) |
| Encoder feature-map probes | 3 | ~3h each | matched/uniform/foveated | `submit_encoder_features.sh` (D) |
| Population coding analysis | 1 (2849188) | ~30-60 min | CPU mostly | `submit_population_coding.sh` |

### 1.3 Friend's H100 (separate cluster)

| Experiment | Status | Notes |
|---|---|---|
| Fov-shifted causal H3 | NOT YET STARTED | Friend has docs (`foveation_transform_fix_retrain.md`); blocks §4.4 H3 |
| Encoder-resolution scaling sweep | NOT YET STARTED | Friend has docs (`encoder_capacity_scaling.md`); blocks App D |
| Multi-seed gap fills (foveation-fix retrains) | NOT YET STARTED | Friend has docs |
| All other (F1-F4) | We're running them on Izar instead | Decision made 2026-04-25 to not split F1-F4 across clusters |

---

## 2. Experiment registry

Status legend:
- ❌ NOT_STARTED — script ready, not submitted
- 🟡 SUBMITTED — sbatch sent, queued or running
- 🔄 RUNNING — currently executing
- ✅ DONE — output exists; not yet integrated
- 📝 INTEGRATED — done + integrated into paper

### 2.1 Existing core (paper v1 backbone)

| ID | Name | Status | Output | Paper § |
|---|---|---|---|---|
| 1 | 5-condition deterministic Gibson probes | 📝 | `<cond>_gibson_det_analysis.json` | §4.1, §4.2, §4.3, §4.5 |
| 2 | 5-condition deterministic MP3D probes | 📝 | `<cond>_mp3d_det_analysis.json` | §4.5, Fig 7 |
| 3 | CKA cross-condition | 📝 | `cka_det.json` | §4.3, Fig 3 |
| 4 | Probe-transfer cross-condition | 📝 | `cross_transfer_det.json` | §4.3, Table 2 |
| 5 | Goal-vector probe | 📝 | `goal_vector_det.json` | §4.4, Fig 5a |
| 6 | Lag-k extended probe | 📝 | `extended_lag_det.json` + `extended_lag_det_lag20.json` | §4.2 |
| 7 | Memory transplant midpoint sweep (3 pairs) | 📝 | `*_to_*_mid<N>.json` | §4.3, Fig 4 |
| 8 | Shortcut discovery | 📝 | `data/shortcut/*` | §4.4, Fig 5b |
| 9 | MLP probe sanity | 📝 | `mlp_sanity_det.json` | §4.5, §5.2 |
| 10 | 1-NN cluster purity | 📝 | `cluster_quality_det.json` | §4.3 |
| 11 | Length-matched probe (I1) | 📝 | `length_matched_det.json` | §4.2 (in temporal-stability subsection) |
| 12 | Per-step temporal probe | 📝 | `temporal_probe_det.json` + `temporal_probe_evolution.{pdf,png}` | §4.2 (Fig `temporal_probe`) |

### 2.2 In-flight follow-ups (this batch, ~50 jobs)

| ID | Name | Job IDs | Status | Output (planned) | Paper § (planned) |
|---|---|---|---|---|---|
| F1 | Fov-fix v2 (clean restart 250M) | 2849136 | 🟡 | `foveated_v2_gibson` ckpts | §3.2 + §4.2 hedge |
| F2 | Fov-fix with normaliser enabled | 2849137 | 🟡 | `foveated_normaliser_gibson` ckpts | §3.2 + limitations |
| F3 | Log-polar foveation | 2849139 | 🟡 | `foveated_logpolar_gibson` ckpts | §3.2 + §4.2 (mechanism test) |
| F4 | Strong-Gaussian foveation σ_max=20 | 2849138 | 🟡 | `foveated_strong_gibson` ckpts | §3.2 (blur-strength sensitivity) |
| A | 5×5 cross-condition transplant matrix | 2849148-2849164 | 🟡 | 17 new `*_to_*_mid30.json` | §4.3 H2 (replace "the pairs we tested") |
| H | Per-step transplant midpoint sweep | 2849166-2849179 | 🟡 | 12 new `*_to_*_mid{200,400,800}.json` | §4.3 (new transplant-vs-midpoint figure) |
| J | Training-dynamics probes | 2849180-2849187 + others | 🟡 | 22 `<cond>_gibson_ckpt<N>_det.npz/json` | §4.6 NEW (training-dynamics figure) |
| D | Encoder feature-map probes | 2849191-2849193 | 🟡 | 3 `<cond>_gibson_encfeat_det.npz/json` | §3.2 + §4.5 ("can encoder feature map alone decode position?") |
| Pop | Population coding analysis | 2849188 | 🟡 | `population_coding_det.json` + `rate_maps_<cond>.pdf` + `population_coding_summary.pdf` | §4.6 NEW (population coding sub-section) |

### 2.3 Friend's H100 — pending (>3 days OR independent module)

| ID | Name | Status | Output | Paper § |
|---|---|---|---|---|
| F5a | Fov-learned clean transform retrain (seeds 0,1,3) | ❌ | 3 ckpts | Table 1 fov-lrn row + multi-seed |
| F5b | Fov-shifted causal control (H3) | ❌ | 1 ckpt | §4.4 H3 (replace `\TODO{H3}` + `\TODO{Training in progress}`) |
| F6 | Encoder-resolution scaling sweep (matched-{32,64,96,192}) | ❌ | 4 ckpts + probes | App D scaling curve |
| F7 | Multi-seed gap fills (blind×2, uni_s3, fov_s3, matched×2) | ❌ | Various | Table 1 error bars + §4.2 multi-seed footnote |

### 2.4 Future / out of v1 scope

| ID | Name | Status | Notes |
|---|---|---|---|
| O1 | Direct H1 mechanism test (GPS perturbation mid-rollout) | ❌ | Future work, listed as TODO in §4.2 "Why" |
| O2 | Cross-architecture transformer baseline | ❌ | Major work; future paper |
| O3 | Length-matching ablation in training (truncate trajectories) | ❌ | TODO in §3.3 |

---

## 3. Paper claim audit (per-section)

For each major claim: what data backs it, status of integration.

Status legend:
- ✅ Backed by deterministic data, integrated, hedged appropriately
- ⚠️ Hedged with "awaits replication" or TODO marker; needs follow-up
- 🆕 New finding from in-flight experiments; needs integration once data lands
- ❌ Not yet supported

### Abstract
| Claim | Status | Backed by |
|---|---|---|
| Encoder–memory race principle | ✅ | §4.2 H1 + temporal probe (Fig temporal_probe) |
| All 5 conditions encode top-layer GPS in mid-episode | ✅ | Temporal probe (commit c904e8c) |
| Bottleneck conditions maintain GPS code throughout | ✅ | Temporal probe + lag-k |
| Rich-encoder conditions overwrite GPS code as visual features take over | ✅ | Temporal probe |
| H2 format divergence | ✅ | CKA + transfer + 1-NN + transplant (4 metrics) |
| H3 gaze-location pending | ⚠️ | Friend's H100 (fov-shifted) |
| MP3D generalisation consistent | ✅ | §4.5 MP3D paragraph (commit 169171d) |
| Bio-analogy 3-fold parallel | ✅ | §5.3 (Chen 2016 disruption fixed in commit 9bdf355) |

### §1 Introduction
| Claim | Status |
|---|---|
| Wijmans (2023) baseline + question framing | ✅ |
| 5-condition design isolates 3 axes | ✅ |
| H1/H2/H3 hypothesis statements | ✅ |
| Findings: encoder–memory race + format divergence | ✅ |
| Contributions paragraph | ✅ |

### §2 Related Work
| Claim | Status |
|---|---|
| Cognitive maps in RL navigation | ✅ |
| Foveated vision / gaze policies in DL | ✅ |
| Probing methodology citations | ✅ |
| Bio precedent | ✅ |
| "What is new here" novelty | ✅ |

### §3 Methods
| Claim | Status |
|---|---|
| Agent architecture (Wijmans backbone) | ✅ |
| 5 visual conditions described | ✅ |
| Matched-compute attribution to 1×1 feature map | ⚠️ — F3 logpolar will sharpen; D will probe encoder feature directly |
| Probing methodology (deterministic, 5-fold CV) | ✅ |
| Sampling-protocol disclosure | ✅ |
| Foveation conditions disable normaliser (F2 will verify) | ⚠️ — F2 result pending |
| Behavioural probes (transplant, shortcut) | ✅ |

### §4 Results — RESTRUCTURED 2026-04-25 (commit pending)

New 6-section structure (was 5; foveation slot is new):
- §4.1 Five conditions, same task — slim summary table (5 cols)
- §4.2 Encoder–memory race (H1) — consolidated, mega-figure
- §4.3 Format-level divergence (H2) — transplant-led
- §4.4 Foveation: where it sits, where the cleaner tests sit — NEW SLOT for in-flight foveation experiments
- §4.5 Gaze location (H3) — kept independent
- §4.6 Boundaries and additional probes — short

| Sub | Claim | Status |
|---|---|---|
| §4.1 | Per-condition summary table (5 cols: cond / frames / SPL / succ / GPS R² / Compass R²) | ✅ slimmed |
| §4.2 | H1 finding + temporal stability + Layer-0 disambiguation + MLP probe + MP3D | ✅ all consolidated into §4.2 with single mega-figure (`fig:h1_mega`) |
| §4.2 | Proposed mechanism (encoder–memory race) | ✅ + TODO for direct causal test |
| §4.3 | H2 format divergence (transplant lead, then CKA / transfer / 1-NN) | ✅ |
| §4.3 | "Pairs we tested" caveat for transplant | ⚠️ — 5×5 matrix in flight (jobs 2849148-79) |
| §4.4 | Foveated-fix / fov-learned current placement (rich-encoder regime under σ_max=8 Gaussian blur) | ✅ |
| §4.4 | F1-F4 strength sweep design + signature | 🆕 PLANNED — disclosed; in training |
| §4.4 | F3 log-polar foveation design + signature | 🆕 PLANNED — disclosed; in training |
| §4.4 | Encoder feature-map probe design + signature | 🆕 PLANNED — disclosed; in probing (job 2849191-93) |
| §4.4 | Foveated-shifted control (links to H3) | 🆕 PLANNED — disclosed; in training |
| §4.5 | H3 learned-gaze collapse | ✅ |
| §4.5 | H3 fov-shifted causal control | ⚠️ TODO (in flight) |
| §4.5 | Shortcut discovery + "having vs using" anomaly | ✅ |
| §4.6 | Occupancy + place cells + per-unit info + goal vector → boundaries | ✅ slimmed |

### §5 Discussion
| Claim | Status |
|---|---|
| Encoder–memory race unification | ✅ |
| Alternative explanations rebuttal | ✅ |
| Bio precedent as analogy | ✅ |
| Implications for ML practice | ✅ |
| Limitations | ✅ |

### §6 Conclusion
| Claim | Status |
|---|---|
| Finding 1: encoder–memory race + temporal stability | ✅ |
| Finding 2: format divergence | ✅ |
| Finding 3: gaze-location test (pending) | ⚠️ |
| Bio analogy | ✅ |
| Open questions | ✅ |

### Appendix
| Section | Status |
|---|---|
| App A: supplementary figures (training, t-SNE, place_cells, goal_vector, layerwise, mp3d-companion) | ✅ |
| App B: gaze-diversity loss pilot | ✅ |
| App C: training stability (NaN bug) | ✅ |
| App D: foveation experiments status (NEW, links from §4.4) | ✅ — design disclosed, results pending |
| App E: encoder-capacity scaling | ⚠️ TODO (friend's H100) |

---

## 4. Figure registry

For each figure: source, freshness, paper-section, ready-to-publish.

**Restructure 2026-04-25**: figure budget went from 8 main + 2 tables → 4 main + 2 tables. Old standalone figures (h1_bottleneck, temporal_probe_evolution, layerwise_decay, mp3d_generalization, h3_content, place_cells, goal_vector_probe) consolidated or moved to appendix.

**Fig 2 simplified again 2026-04-25 (afternoon)**: H1 mega-figure went from 4 panels (panel a was itself 3 sub-bar-charts for GPS / Compass / DtG) → 3 cleaner panels: (a) grouped GPS bars × {Gibson, MP3D}, (b) temporal probe, (c) per-layer. Compass + DtG referenced via Table 1 + appendix supp figs. Total Fig 2 sub-axes: 6 → 3.

**Population coding 2026-04-25**: pulled `population_coding_det.json` + summary figure from Izar (job 2849188 done). Replaced broken `place_cells.pdf` with `population_coding_summary.pdf` in App A; updated §4.6 paragraph. **Data correctness flag**: old `place_cells.pdf` was showing impossible per-unit info (mean $\bar s = 65$ bits/unit, max 150) — but a 20×20 grid is bounded at $\log_2(400) \approx 8.6$ bits per unit. The new analysis (capped at ~2 bits max) is correct. The old figure's qualitative ordering (bottleneck > rich-encoder for n_above_1bit) was also REVERSED by the corrected analysis: rich-encoder conditions actually have MORE high-info units (16-18 vs blind's 1), but those units don't decode GPS — they encode position-correlated features. New paper §4.6 reflects this: blind has compressed broad spatial code, rich-encoder has higher-dim representations with peaked landmark-like units that don't translate to GPS readout. This is consistent with the encoder-memory race story (rich-encoder uses encoder for landmarks, doesn't compress to position).

### Main figures (after restructure)
| Filename | Source | Freshness | Section | Status |
|---|---|---|---|---|
| `fig_blind.png`, `fig_uniform.png`, `fig_foveated.png`, `fig_topdown.png` | Static illustrations | static | Fig 1 (setup, in §1) | ✅ |
| `h1_mega.pdf` (NEW) | 4-panel composite: hardcoded T1 + temporal_probe_det.json + 1d_multilayer + 1b_global_gps_compass × Gibson/MP3D | det | Fig 2 (§4.2 H1) | ✅ |
| `h2_cka_heatmap.pdf` + `transplant_sweep.pdf` (combined fig) | `cka_det.json` + `*_to_*_mid<N>.json` | det | Fig 3 (§4.3 H2) | ✅ + 🆕 will refresh after A/H land |
| `shortcut_bars.pdf` (was part of h3_content) | `data/shortcut/*` | det | Fig 4 (§4.5 H3) | ✅ |
| `training_curves.pdf` + `h2_hidden_embedding_tsne.pdf` | TB / det NPZs | det | App A supp fig | ✅ |
| `place_cells.pdf` + `goal_vector_probe.pdf` (now App) | det JSONs | det | App A supp fig | ✅ moved to appendix |
| `layerwise_decay.pdf` + `mp3d_generalization.pdf` (now App) | det JSONs (companions to h1_mega panels c, d) | det | App A supp fig | ✅ moved to appendix |
| `temporal_probe_evolution.pdf` (standalone, kept for completeness) | `temporal_probe_det.json` | det | now part of `h1_mega` panel (b); standalone retained | ✅ |

### Foveation slot figures (planned for §4.4)
| Filename | Source | Status |
|---|---|---|
| `foveation_strength_sweep.pdf` | F1-F4 probe results | 🆕 PLANNED — in flight |
| `logpolar_vs_blur.pdf` | F3 log-polar probes | 🆕 PLANNED — in flight |
| `encoder_feature_probe.pdf` | encoder feature-map probes job 2849191-93 | 🆕 PLANNED — in flight |
| `foveated_shifted_results.pdf` | fov-shifted training + probe | 🆕 PLANNED — in flight |

### Other in-flight
| Filename | Source | Section | Status |
|---|---|---|---|
| `population_coding_summary.pdf` | `population_coding_det.json` (job 2849188; **JSON LANDED 2026-04-25, awaits figure script**) | App or §4.6 | 🆕 |
| `training_dynamics.pdf` | `<cond>_gibson_ckpt<N>_det_analysis.json` × 22 | §4.6 / App | 🆕 In-flight (J) |
| `transplant_5x5_matrix.pdf` | 5×5 transplant SPL matrix | §4.3 | 🆕 In-flight (A, jobs 2849148-64) |
| `transplant_per_step.pdf` | extended midpoint sweep | §4.3 | 🆕 In-flight (H, jobs 2849166-79) |
| `scaling_sweep.pdf` | encoder-resolution sweep | App E | ❌ Friend's H100 |

---

## 5. Open questions / TODOs

### 5.1 Awaiting cluster results
- F1: under-training control for fov-fix (current uses ckpt.36 / 174M) — does fov-fix at full 250M still match uniform pass-through?
- F2: normaliser-invariance — was disabling RunningMeanAndVar a confound for fov-fix vs uniform comparison?
- F3: log-polar foveation — does *strong* foveation create real encoder bottleneck?
- F4: σ_max=20 — does blur-strength matter at all?
- A: complete 5×5 transplant matrix — replace "the pairs we tested" with full matrix
- H: per-step transplant — does cross-condition transplant cost grow with midpoint as predicted?
- J: training dynamics — when in training does encoder–memory race emerge?
- D: encoder feature-map — does matched-compute encoder output preserve position info even with 1×1 spatial collapse?
- Pop coding: 5 conditions' rate-map gallery + sparse decoding curve + intrinsic dim

### 5.2 Awaiting friend's H100
- F5a: fov-learned clean transform retrain (seeds 0,1,3) — replaces buggy-transform fov-lrn data
- F5b: fov-shifted causal H3 — populates §4.4
- F6: encoder-resolution scaling sweep — populates App D
- F7: multi-seed gap fills

### 5.3 Future / out-of-scope
- Direct H1 mechanism test (GPS perturbation mid-rollout)
- Transformer architecture replication
- Length-matching ablation in TRAINING (truncate trajectories during PPO)
- Per-unit place-cell visualisation animation across episode
- Cross-heading probe with larger N

### 5.4 Audit-flagged but not blocking
- Blind Layer-1 R² dip not explained (Fig 6 caption flagged)
- Matched-compute "channel info" — D will resolve
- Selectivity-with-within-episode-shuffle (B2 audit, MINOR)
- Lag-k extended to k=50, 100 (could go further)
- Large-N 1-NN purity (current 7500, could go to ~50k)

---

## 6. Decision log (chronological, why we did what)

### 2026-04-25 (afternoon): Results section restructure + foveation slot

**Why restructured**: The original §4 Results structure put H1 as a hypothesis-by-hypothesis layout (H1 → H2 → H3 → "additional analyses"), with H1's most important reinforcing evidence (per-layer probe, MLP probe, MP3D generalization) buried in "additional analyses" rather than alongside H1. A reviewer's natural objection ("Layer 0 GPS exists for ALL conditions — how can H1 distinguish?") was answered three subsections later. Also, foveation was scattered as one of five conditions with no dedicated narrative slot, mismatching the paper's title.

**What changed**:
1. New 6-subsection structure: §4.1 setup → §4.2 H1 (consolidated) → §4.3 H2 (transplant-led) → **§4.4 Foveation slot (NEW)** → §4.5 H3 → §4.6 Boundaries.
2. Consolidated H1 evidence into a single mega-figure `fig:h1_mega` with 4 panels: (a) current-state bars, (b) temporal probe, (c) per-layer, (d) MP3D. Replaces 4 separate figures.
3. Slimmed summary table from 11 columns → 6 columns. lag-5 / shortcut / MP3D columns moved to their respective subsections.
4. Place-cells, layerwise, goal-vector, MP3D-companion figures all moved to App A supplementary visualisations.
5. Foveation slot (§4.4) created with disclosed-design / pending-results paragraphs for: F1-F4 strength sweep, F3 log-polar, encoder feature-map probe, foveated-shifted control. New App D `app:foveation-status` lists submission state.

**Data fix surfaced in the process**: existing `make_mp3d_generalization_figure.py` used JSON field `gps_r2` (single train/test fit) instead of `gps_cv_r2_mean` (5-fold CV). This inflated the apparent rich-encoder negativity on MP3D in the standalone figure. Mega-figure panel (d) uses the correct CV-mean, which agrees with Table 1 / panel (a). Side effect: foveated MP3D GPS is `+0.35` (mild positive recovery, not "chance"), and foveated-learned MP3D is `-0.79` (less bad than `-2.43` Gibson). Paper text updated accordingly.

**Scripts added**: `scripts/paper_figures/make_h1_mega_figure.py`.

### 2026-04-22 to 04-24: Pre-fix work and bug discoveries

**The deterministic-sampling bug (root cause of original H1 reversal)**
- `scripts/probing/collect.py` hardcoded `deterministic=False` while every other
  eval script in the codebase uses `deterministic=True` with explicit
  "deterministic for eval" comments.  Under stochastic sampling, conditions
  with higher action entropy (fov-fix, uniform, matched, blind under their
  trained stochastic policies) sampled the STOP action with probability
  $\sim 0.25$/step → mean episode length $\sim 4$ steps → target variance two
  orders of magnitude below the episodic range → trivially high probe $R^2$
  across all conditions (any predictor fits a near-constant target).
- Fixed in commit `c81352e` (default to `True`, expose `--deterministic` flag).
  Re-collected all 5 conditions' det probes.  Numbers used in paper Table 1
  are all from the post-fix data.

**Original-vs-fixed claim status (after deterministic re-collection)**
| Original claim | Status | Evidence under det |
|---|---|---|
| H1: foveated > uniform compensatory memory | ❌ REVERSED | Both ~0 GPS R² |
| "More pixels ≠ better decoding" | ✅ Stronger | Matched 1×1 R²=0.78 vs uniform R²≈0 |
| H2: representational format divergence | ✅ Survives | CKA / transplant / transfer all behavioural-grade |
| H3: fov-learned compass 0.94 | ❌ Bug artefact | Det CV R² = -1.34 ± 3.14 |
| "LSTM learns cognitive map" | ✅ Strengthened | Blind 0.95, matched 0.78 |

**The TRUE finding** (replacing the original claim):
"Visual encoder capacity inversely determines linear top-layer LSTM spatial
encoding."  Decoupled cleanly via the matched-compute condition (visual
input present, but encoder feature map is 1×1 — bottleneck without "no
vision").

**Other audit-found bugs (categorised in former AUDIT_FINDINGS.md, all
fixed in commits 79c2db1, b7513d9, etc.)**
- **B1 (per-scene step-level split leakage)** in `analyze.py`: probe split
  was step-level within scene, leaking between train/test for episodes
  straddling the split.  Fix: episode-level split within each scene.
  Re-analysed (numbers within 0.02 R² of pre-fix) — claim unaffected.
- **G1 (shortcut figure caption mis-states aggregation)**: caption claimed
  `n=200` per bar (flat average), code does mean-of-20-scene-means.  Fixed
  caption.
- **G4 (transplant episode pinning)**: `env._current_episode = ep` not
  guaranteed to pin episode in all habitat-lab versions.  Fix: use
  `env._episode_iterator = iter([ep])` + assertion.  Aggregate SPL deltas
  (0.19-0.21) unaffected because averaged over 150 random eps.
- **E2 (foveation `_max_dist` static instead of per-sample)**: affects
  shifted-gaze conditions, off-centre gaze over-saturates peripheral
  eccentricity.  Fix: per-sample dynamic max_dist.  **Centred-gaze
  conditions (fov-fix, blind, matched, uniform) unaffected; fov-learned
  and fov-shifted require retrain (delegated to friend's H100,
  `foveation_transform_fix_retrain.md`)**.
- **G3, B2, B3, B5, C3, C4, D5, etc.**: minor / documentation-only fixes,
  no claim impact, all addressed in commits.

### 2026-04-24: Paper reframe
- Original H1 ("foveated > uniform compensatory memory") busted under deterministic data. Reframed to encoder–memory race using matched-compute as decoupling condition.
- Friend takes over fov-shifted training; we keep doing analysis on Izar.

### 2026-04-25: V1 polish + new findings
- Reviewed paper end-to-end; hedged 13 over-claims; added 7+ TODO markers for follow-up experiments.
- Lag-20 verified (commit a737aa5).
- Length-matched probe surfaced sub-finding: H1 ordering preserved across step-caps but rich-encoder GPS R² lifts substantially when length-capped.
- Per-step temporal probe confirmed: rich-encoder LSTM encodes GPS in mid-episode, overwrites in long tail. Reframed H1 around temporal stability (commit 6408d6c).
- User decision: F3 also on Izar (not friend) to keep cluster-noise consistent across conditions. Friend reserved for >3-day OR independent-module jobs.
- User decision: do all 5 follow-up components ourselves (A/D/H/J + pop coding) rather than delegate. Submitted ~50 jobs (commit be369f5).

---

## 7. How to update this doc

When state changes:
1. Update §0 Quick status (or §1 cluster jobs) first — that's what someone glances at.
2. Update the relevant experiment row in §2.
3. Update the relevant claim row in §3 if the change affects a paper claim.
4. Update §4 figure registry if a figure changes.
5. If a new TODO emerges, add to §5; if an old one resolves, move it to §6 decision log with date.
6. Bump the "Last updated" date at the top.

Never delete history — move resolved items to §6 with a date stamp.

Useful one-liners to refresh state:
```bash
# Cluster jobs
ssh izar "squeue -u wxu --format='%.10i %.20j %.5T %.10q %.10M'"

# Recent commits affecting paper
git log --oneline -10 -- docs/NeurIPS_2026/neurips_2026.tex

# Figure freshness (date-sorted)
ls -la docs/NeurIPS_2026/fig/*.pdf | sort -k 6,8

# Pending det probe analyses
ssh izar "ls /scratch/izar/wxu/probing_results/*_det_analysis.json | wc -l"
```
