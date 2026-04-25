# Master Track — NeurIPS 2026 submission

Single source of truth for: cluster jobs, experiment status, paper
claims, figure freshness, open questions, decision log.

**Last updated**: 2026-04-25 23:40 (commit `a8a1338`).
Update this file when state changes — do NOT rely on memory.

---

## 0. Quick status (at-a-glance)

| Dimension | Status |
|---|---|
| Paper version | v1 substantially complete; aggressive cut done (§3 / §4.4 / §5.2 / §5.3 → appendix); main text dense |
| Page count | **29 pages total**; main text 12-14 (Discussion p12, Conclusion p14, References p15+) |
| Submission deadline | **~2 weeks out** (per user 2026-04-25) — not rushed |
| Cluster: jobs RUNNING | 5 trainings (multi-seed seed2 + 3 foveation variants + normaliser) + 6 transplants |
| Cluster: jobs PENDING | 22 training-dyn probes + 4-ish transplants |
| Most recent landed | Encoder feature-map probes (3/3 conds), Phase B shortcut+trajectories (5/5 conds), 33-cell 5×5 transplant, 1 training-dyn ckpt (blind) |
| Most recent paper change | Aggressive cut + appendix moves (commits `7409ac8` + `8267f80`); main text reduced from 30→29p; ~1 page headroom freed for incoming results |
| Most recent submit batch | 22 training-dyn probes + 7 transplants (4 matched-recipient + 3 per-step missing) — 2026-04-25 23:30 (commit `a8a1338`) |

---

## 1. Cluster jobs

### 1.1 Running (verify with `ssh izar "squeue -u wxu"`)

Snapshot 2026-04-25 23:40:

| Job ID | Name | Elapsed | Time left | Output |
|---|---|---|---|---|
| 2844326 | uni_s2 multi-seed | 2d 6h | **~18h** | `uniform_gibson_seed2/` (21/36 ckpts ≈ 84%) |
| 2844327 | fov_s2 multi-seed | 2d 5h | **~18h** | `foveated_gibson_seed2/` (20/36 ckpts ≈ 80%) |
| 2849139 | F3 log-polar | 5h | 2.5d | `foveated_logpolar_gibson/` (3 ckpts) |
| 2849138 | F4 σ_max=20 strong | 7h | 2.5d | `foveated_strong_gibson/` (3 ckpts) |
| 2849136 | foveated_v2 (clean σ=8 retrain) | 8h | 2.5d | `foveated_v2_gibson/` (3 ckpts) |
| 2849... | F2 normaliser | very early | 3d | `foveated_normaliser_gibson/` (1 ckpt) |
| 2850299-2850305 | Transplant cells (matched-recipient + a few per-step) | 0-5min | 3h | `/scratch/izar/wxu/transplant_results/` |

### 1.2 Pending (just submitted, 2026-04-25 23:30, commit `a8a1338`)

| Batch | Jobs | Status | Per-job |
|---|---|---|---|
| Training-dynamics probes (J) | 22 (2850318-2850340) | PENDING (queued normal QOS, run in parallel ~10 at a time) | ~1-3h each |
| 4 matched-recipient transplant + 3 per-step missing | 7 | RUNNING (above) | ~1.5h each |

### 1.3 Already LANDED & integrated in paper

| Batch | Status | Paper section |
|---|---|---|
| 5×5 cross-condition transplant (excluding matched recipient) | 13/16 cross + diag, 33 cells total | Fig 3 right panel (§4.3) |
| Per-step transplant (mid200/400/800) | mostly done | App `fig:transplant_sweep_supp` |
| Encoder feature-map probes (D) | **3/3** (matched/uniform/foveated) | §4.4.4 + Fig 4 |
| Phase B shortcut + trajectories (ST) | **5/5** | §4.5 + Fig 6 |
| Population coding analysis | 1/1 | App A `fig:supp_pop_coding` |
| Topdown render for Fig 1d | 1/1 | Fig 1d (with floor plan) |

### 1.3 Friend's H100 — REQUIRED experiments (separate cluster, blocks paper)

These experiments need H100 because they require new training runs (2-7 days each on V100, faster on H100) that are too long for our Izar QOS, OR are independent modules best parallelised separately. \emph{None of them are in flight right now — they need explicit hand-off to the collaborator.}

| Code | Experiment | Why needed | Paper section affected | Effort estimate |
|---|---|---|---|---|
| **H100-A** | Fov-shifted causal H3 retrain (clean transform) | Old `foveated_shifted_gibson_buggy_transform` used the broken transform; clean H3 causal control needs a fresh retrain at the collapsed gaze location $(0.49, 0.62)$ | §4.5 H3 (currently `\TODO{Training in progress}`) + §4.4.4 in App D | 1 retrain × 2--3 days V100 (faster on H100) |
| **H100-B** | Encoder-resolution scaling sweep ($32, 48, 64, 96, 128, 192$ pixels at fixed encoder stack) | Tests H1 mechanism causally — directly varies encoder spatial output dimensionality with everything else held fixed | App E (currently `\TODO{Pending data}`); strengthens §4.2 mechanism + §5.4 implication (i) | 6 retrains × 2--3 days V100 = 12-18 V100-days. **H100 critical** |
| **H100-C** | Foveation-fix multi-seed (seeds 0, 1, 3) | Lets us promote the §4.5 2$\times$2 dissociation from "candidate" to "robust"; current claims rest on single-seed runs | §1 / §4.5 / §6 (hedging upgrade) | 3 retrains × 2--3 days each (Izar uniform_seed2 + fov_seed2 already running, will land tomorrow) |

**Hand-off status**: Friend has docs (`experiments/foveation_transform_fix_retrain.md`, `experiments/encoder_capacity_scaling.md`) but has not started training. **Critical path: H100-A and H100-B are the most blocking** — without them §4.4 H3 and App E remain TODO at submission.

**Decision (2026-04-25)**: F1-F4 σ-sweep + log-polar + normaliser were moved off H100 list onto Izar to keep results in one place; H1-H3 paper claims do NOT depend on F1-F4. The H100 list is now strictly the experiments friend's compute is best for or for which we lack queue capacity.

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
| §4.2 | Proposed mechanism (encoder–memory race), wording aligned with encoder probe | ✅ refined to "spatial-feature variety per step" framing (commit `cc1b7e7`); direct causal test still TODO |
| §4.3 | H2 format divergence (transplant lead, then CKA / transfer / 1-NN) | ✅ |
| §4.3 | "Pairs we tested" caveat for transplant | ✅ MOSTLY RESOLVED — 5×5 matrix at midpoint=30 with 13/16 cross-cells filled (commit `6ecc508`); 4 matched-recipient cells pending |
| §4.3 | Asymmetry pattern in 5×5 matrix (blind→uniform -0.38 vs uniform→blind +0.02) | ✅ NEW finding integrated (commit `6ecc508`) |
| §4.4 | Foveation table comparing fov vs uniform (converge on H1, diverge on H2 + behaviour) | ✅ Table 3 (commit `98b9627`) |
| §4.4 | F1-F4 strength sweep design + signature | 🆕 in training — script ready |
| §4.4 | F3 log-polar foveation design + falsifiable prediction (R² ≥ 0.3 if mechanism is encoder spatial output dim) | ✅ DISCLOSED (commit `70bd1fd`); in training, 2/36 ckpts |
| §4.4 | Encoder feature-map probe (matched/uniform/foveated all 3) | ✅ all landed (commit `409e6bd`); none of the 3 sighted encoders linearly decode GPS |
| §4.4 | Foveated-shifted control (links to H3) | 🆕 in training |
| §4.5 | H3 learned-gaze collapse | ✅ |
| §4.5 | H3 fov-shifted causal control | ⚠️ TODO (in flight) |
| §4.5 | Shortcut discovery + 2×2 dissociation (matched/uniform anomalies) | ✅ scatter (commit `4d9ee81`) + paired-traj fig (commit `d97eda9`) |
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

**Trajectory visualization 2026-04-25 (afternoon, Phase A + B)**:
- **Phase A (DONE, commit 84a81fc)**: Modified Fig 1 setup to include 5-condition trajectory overlay on the same Gibson val episode (scene 92, ep 414, selected because all 5 conditions succeeded with SPL within ±0.10 of their per-condition median). Bottleneck conditions take winding paths (blind 231 / matched 146 steps); rich-encoder conditions take direct paths (~80 steps). Fig 1 now visualises input ablation AND behavioural consequence on page 1.
- **Phase B (IN-FLIGHT, jobs 2849306-10)**: New `scripts/eval/shortcut_with_trajectories.py` re-runs the shortcut paired-episode eval but saves per-episode trajectories (positions, dtg). Submitted 5 jobs using the same ckpts as original shortcut eval. When complete: render paired-episode figure for §4.5 showing reset-memory vs persistent-memory trajectories on the same scene/goal — visualizes the "LSTM locks onto previous goal" interpretation of the matched/uniform anomalies.

**Population coding 2026-04-25**: pulled `population_coding_det.json` + summary figure from Izar (job 2849188 done). Replaced broken `place_cells.pdf` with `population_coding_summary.pdf` in App A; updated §4.6 paragraph. **Data correctness flag**: old `place_cells.pdf` was showing impossible per-unit info (mean $\bar s = 65$ bits/unit, max 150) — but a 20×20 grid is bounded at $\log_2(400) \approx 8.6$ bits per unit. The new analysis (capped at ~2 bits max) is correct. The old figure's qualitative ordering (bottleneck > rich-encoder for n_above_1bit) was also REVERSED by the corrected analysis: rich-encoder conditions actually have MORE high-info units (16-18 vs blind's 1), but those units don't decode GPS — they encode position-correlated features. New paper §4.6 reflects this: blind has compressed broad spatial code, rich-encoder has higher-dim representations with peaked landmark-like units that don't translate to GPS readout. This is consistent with the encoder-memory race story (rich-encoder uses encoder for landmarks, doesn't compress to position).

### Main figures (current)
| Filename | Section | Status |
|---|---|---|
| `fig_blind.png`, `fig_uniform.png`, `fig_foveated.png` + `trajectory_overlay.pdf` (with topdown) | Fig 1 setup + trajectory overlay (§1, §4.2 preview) | ✅ Fig 1d uses topdown floor plan, scene E9uDoFAP3SH ep 414 (commit `01c4933`) |
| `h1_mega.pdf` | Fig 2 (§4.2 H1) — 3-panel: bars / temporal / per-layer / MP3D | ✅ |
| `h2_cka_heatmap.pdf` + `transplant_5x5.pdf` | Fig 3 (§4.3 H2) — CKA heatmap + 5×5 cross-condition transplant matrix | ✅ — 33/37 cells populated; 4 matched-recipient cells pending (commit `6ecc508`) |
| `encoder_feature_probe.pdf` | Fig 4 (§4.4 foveation, encoder feature-map probe) — 3 conditions × GPS/Compass | ✅ matched/uniform/foveated all done (commit `409e6bd`) |
| `shortcut_scatter.pdf` | Fig 5 (§4.5 H3) — 2×2 dissociation: probe vs behavioural memory | ✅ (commit `4d9ee81`) |
| `shortcut_paired_traj.pdf` | Fig 6 (§4.5 H3) — paired-trajectory failure visualisation, 3 conditions | ✅ (commit `d97eda9`) |

### Appendix A figures (after restructure)
| Filename | Source | Status |
|---|---|---|
| `training_curves.pdf` + `h2_hidden_embedding_tsne.pdf` | TB / det NPZs | ✅ |
| `population_coding_summary.pdf` + `goal_vector_probe.pdf` | det JSONs | ✅ |
| `layerwise_decay.pdf` + `mp3d_generalization.pdf` | det JSONs (companions to h1_mega panels c, d) | ✅ moved to appendix |
| `temporal_probe_evolution.pdf` (standalone) | `temporal_probe_det.json` | ✅ — also in h1_mega panel (b) |
| `transplant_sweep.pdf` (NEW App location) | midpoint-sweep view of 3 representative pairs (commit `6ecc508`) | ✅ supports the t=30 main matrix |

### Foveation slot figures (placeholders for §4.4 sub-paragraphs; data in flight)
| Filename | Source | Status |
|---|---|---|
| `foveation_strength_sweep.pdf` | F1-F4 probe results | 🆕 SCRIPT READY (`make_foveation_strength_figure.py`); awaits training + probe |
| `logpolar_vs_blur.pdf` | F3 log-polar probes | 🆕 awaits log-polar checkpoint to be probable |
| `foveated_shifted_results.pdf` | fov-shifted training + probe | 🆕 awaits fov-shifted training |

### Other in-flight figures (scripts ready)
| Filename | Section | Status |
|---|---|---|
| `training_dynamics.pdf` | §4.6 / App | 🆕 SCRIPT READY (`make_training_dynamics_figure.py`); 1/22 ckpt landed |
| `transplant_5x5.pdf` (current) | Fig 3 right | ✅ partial; will refresh as 4 matched-recipient cells land |
| `scaling_sweep.pdf` | App E | ❌ Friend's H100 |

---

## 5. Open questions / TODOs

### 5.1 Awaiting cluster results (in flight on Izar)

| Code | Experiment | What it answers | Paper section affected | Submission |
|---|---|---|---|---|
| F1 | Foveation $\sigma_{\max}=2$ | Low-blur foveated $\to$ does it look like uniform? | §4.4.2 | submitted |
| F2 | Foveation $\sigma_{\max}=4$ | Same gradient, lower σ | §4.4.2 | submitted |
| F3 | Log-polar foveation | Spatial-sampling vs blur bottleneck distinction; predict LSTM GPS R² ≥ 0.3 (encoder 2×2 spatial output, between matched 1×1 and uniform 8×8) | §4.4.3 | training (14M/250M, ETA 2-3 days). **At ~125M (1.5 days) do early probe to check trend** |
| F4a | Foveation $\sigma_{\max}=12$ | High-blur foveated $\to$ approaches matched? | §4.4.2 | submitted |
| F4b | Foveation $\sigma_{\max}=20$ | Strongest foveation in sweep | §4.4.2 | submitted |
| F-shift | Foveated-shifted (gaze $(0.49, 0.62)$) | Gaze location as 2nd content axis (H3 causal) | §4.4.4 + §4.5 | submitted |
| F-norm | Foveated with normaliser re-enabled | Normaliser-invariance control | §3.2 implementation note | submitted |
| A | 5×5 transplant matrix (cross-pairs at midpoint=30) | Replace "we tested 3 pairs" with complete matrix | §4.3 | LANDED 33 cells; 4 matched-recipient cells re-submitted 2850299-2850305 |
| H | Per-step transplant (extended midpoints 200/400/800) | Does mismatch cost grow with midpoint? | App `fig:transplant_sweep_supp` | LANDED |
| J | Training dynamics (22 ckpt probes across 5 conds) | When does encoder-memory race emerge in training? | §4.6 / App | 1/22 LANDED; 22 re-submitted 2850318-2850340 normal QOS |
| D | Encoder feature-map probe (matched, uniform, foveated) | Where world-frame info sits encoder vs LSTM | §4.4.4 | **LANDED 3/3** + integrated |
| ST | Shortcut + trajectories (5 conds) | Phase B paired-episode trajectory figure for §4.5 | §4.5 | **LANDED 5/5** + integrated |

### 5.2 ⚠️ Friend's H100 — REQUIRED experiments (block paper, see §1.3 above)

| Code | Experiment | What it answers | Status |
|---|---|---|---|
| **H100-A** | Fov-shifted causal H3 retrain (clean transform, seeds 0/1/3) | Replaces buggy-transform fov-lrn baseline AND populates §4.4.4 + §4.5 H3 causal test | ⚠️ **NOT STARTED** — critical path |
| **H100-B** | Encoder-resolution scaling sweep ($32, 48, 64, 96, 128, 192$ at fixed encoder stack) | App E: causally tests the encoder-spatial-output mechanism by varying encoder resolution while holding everything else fixed | ⚠️ **NOT STARTED** — critical path |
| **H100-C** | Multi-seed gap fills for blind, matched, foveated, foveated-learned (seed 0, 1, 3) | Promotes single-seed dissociations to robust findings; uniform_seed2 + foveated_seed2 already on Izar (land tomorrow) | ⚠️ Partial: uniform/fov done on Izar, blind + matched + fov-lrn still need H100 |

### 5.3 Should-have experiments — not yet planned, would strengthen claims if added

For each: which paper claim is currently held by hedging that this experiment would tighten.

| Code | Experiment | What it tightens | Effort estimate |
|---|---|---|---|
| H1-causal | Train rich-encoder agent with GPS perturbed mid-rollout; or train bottleneck agent with GPS sensor removed | "Encoder–memory race" from candidate unifying account → mechanism (currently softened in §5.1) | 2-3 retrains; ~4 days V100 each |
| Pol-rely | Ablate GPS sensor mid-rollout at eval and measure SPL drop | Distinguishes "policy reads GPS code" from "policy reads non-GPS memory" — directly tests the matched-vs-uniform 2×2 anomaly | Eval-only; ~2 hours per cond |
| Multi-seed-shortcut | Re-run shortcut eval on multi-seed checkpoints | Promote 2×2 dissociation (matched + uniform anomalies) from candidate to robust finding | Depends on F5a/F7 |
| Length-match | Truncate every condition's probing data to common length, re-run probes | Eliminates "long blind episodes give artificially high R²" hypothetical confound | Eval-only; ~1 hour |
| 1-NN-large-N | 1-NN purity at $\sim 50$k pooled samples (currently 7500) | Bounds the "1-NN purity = 1.000" finding's sample-size effect | Analysis-only; few minutes |
| 1-NN-MP3D | 1-NN purity on MP3D-pooled hidden states | H2 robustness to dataset shift | Analysis-only; needs MP3D NPZs |
| Cross-head-N | Cross-heading probe with larger N (was tried, returned "insufficient samples") | Heading-invariance of GPS code | Bigger probe collection job; ~4 hours |
| Hybrid-sensor | Hybrid sensor (coarse-uniform-periphery + sharp-fovea, OR depth-only) | "Hybrid sensors might produce intermediate geometries" hint in §4.3 | New training condition; ~4 days V100 |
| Banino-grid | Test grid-cell periodicity / hexagonal autocorrelation in LSTM units | Compare to grid-cell literature directly | Analysis-only; ~1 hour |
| Deep-lag | Lag-$k$ probe to $k=50, 100$ | Bound the "persistent at lags ≥20" claim | Analysis-only |

### 5.4 Out-of-scope (paper-time, not future-work)

- **Transformer-architecture replication of all 5 conditions** — would test architecture-independence of encoder–memory race. Requires retraining every condition on a transformer backbone (≥5×4 days = month). Discussed in §5.4 Discussion as a prediction.
- **Non-navigation embodied tasks** — would test task-independence. New environment, new training pipeline. Out of scope.
- **Supervised visual learners (not RL)** — would test learning-objective-independence. Discussed in §5.4 as open empirical question.
- **Per-unit place-cell visualisation animation across episode** — nice-to-have for UI / website; not load-bearing.
- **Active-gaze with stochastic / info-seeking gaze decoder** — separate paper; explicitly disclaimed in §5.5 as architecture-specific to our minimal decoder.

### 5.5 Audit-flagged side observations (not blocking)
- Blind Layer-1 R² dip (0.61 vs Layer 0/2 ≈ 0.95) — Fig 2c notes this; not yet explained
- Matched-compute "channel info" alone might encode position — D in flight resolves
- Selectivity-with-within-episode-shuffle as alternative to label-permutation Hewitt-Liang (B2 audit, minor)
- Foveated-learned MP3D compass +1.75 swing (Gibson −1.34 → MP3D +0.41) — single-seed; multi-seed would test
- Population coding finding "rich-encoder peaked units encode position-correlated features" depends on threshold (1 bit); robustness under threshold sweep would help

---

## 6. Decision log (chronological, why we did what)

### 2026-04-25 (evening): Encoder feature-map probe + Phase B + 5×5 + H1 mechanism refinement

Big batch of cluster results landed late afternoon / evening:

**Encoder feature-map probe — 3/3 conditions** (commit `409e6bd`):
- matched encoder→GPS R² = -3.14 ± 5.91
- uniform encoder→GPS R² = -0.32 ± 0.08
- foveated encoder→GPS R² = -0.65 ± 0.28

NONE of the 3 sighted encoders linearly decode GPS. This was a more interesting outcome than the original 3 hypothesised possibilities. Sharpened H1 mechanism wording across abstract / §4.2 / §5.4 (commit `cc1b7e7`):

OLD framing: "rich encoder can re-derive position from current frame; bottleneck encoder cannot → LSTM compensates"
NEW framing: "no encoder linearly preserves GPS; what determines whether LSTM compensates is the encoder's spatial-feature variety per step (matched 1×1 vs uniform 8×8). LSTM Layer 0 reads GPS sensor directly; bottleneck conditions integrate it across time because they have minimal visual feature variety to substitute"

**Phase B paired-trajectory figure — 5/5 conditions** (commit `d97eda9`):
- New eval script `scripts/eval/shortcut_with_trajectories.py` saves per-episode positions
- Figure shows "memory locks onto old goal" failure mode: blind's persistent run oscillates around previous-episode goal location; uniform/foveated wander more diffusely
- Visualises the §4.5 2×2 dissociation finding behaviorally

**5×5 transplant matrix — 33 cells** (commit `6ecc508`):
- Replaces the 3-pair midpoint sweep as Fig 3 right panel
- 13/16 cross-pairs at midpoint=30 visible; 4 matched-recipient cells pending
- New findings: asymmetry (blind→uniform -0.38 vs uniform→blind +0.02), recipient ranking (uniform suffers most from foreign donors, blind least)
- Old transplant_sweep moved to App A as midpoint-stability evidence

**Topdown render fix** (commit `01c4933`):
- Fig 1d trajectory overlay now on actual Habitat occupancy map for the scene (E9uDoFAP3SH)
- Required 3 attempts: env._episode_iterator hack didn't work; env._dataset.episodes hack didn't work; finally fixed by setting config.habitat.simulator.scene + passing filtered dataset explicitly to habitat.Env

**F3 log-polar prediction** (commit `70bd1fd`):
- Wrote falsifiable prediction in §4.4.3: log-polar should give LSTM GPS R² ≥ 0.3 (because encoder 2×2 spatial output, between matched 1×1 and uniform 8×8). If mechanism is encoder spatial-output dimensionality, this should manifest in the LSTM
- F3 still training (~5% done at log time), ETA 2-3 days

**Cluster QOS workaround for short jobs** (decision):
- Topdown render needed debug QOS (priority 50000) to bypass the ~30 pending normal-QOS jobs ahead
- For future short single-purpose jobs (analyses on existing checkpoints), use --qos=debug + 1h walltime + 5 max submissions

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
