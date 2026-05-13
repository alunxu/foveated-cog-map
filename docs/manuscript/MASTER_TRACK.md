# Master Track — manuscript submission

Single source of truth for: cluster jobs, experiment status, paper
claims, figure freshness, open questions, decision log.

**Last updated**: 2026-04-30 05:30 (Fig 1 forced to top of page 2 by relocating figure source between §1 P2 bio-precedent and P3a literature-gap with `[t]` placement — verified via pdftotext: page 2 leads with ``(A) Training Architecture'' + ``(B) Memory Analysis''. Added §5.3.2 ``Epstein cog-map engagement — relational/topological analyses (PLANNED post-retrain)'' with detailed designs for EM1 hierarchical scene clustering, EM2 same/diff-scene pair classifier, EM3 topological reachability probe, EM4 graph reconstruction (deferred). Per-analysis: goal, inputs, procedure, output, acceptance criteria (compelling vs null), risk/contingency. Decision: WAIT for unified-retrain h₂ NPZs; pre-build EM1/EM2/EM3 scripts (~1h) while waiting. Earlier today: paper §1+§2+§3 audit pass committed; 3 commits. §1: dropped "information conservation" overclaim across 8 sites — abstract/Fig 4/§4.3/Table 2/§5.1 synthesis/App; renamed `fig:information_conservation` → `fig:information_allocation` + PDF + script; softened 6 intro overclaims. §2: fixed "four lines" → "five" + softened "tests it" → "provides controlled analog" + "argues against" → "is one extreme on continuum". §3: fixed lead paragraph wording to match new abstract.

**Audit-flagged TODO items NOT touched in this pass:**
- M2: 100M same-frame anchor numbers in §3 (linear R² = 0.96/0.84/0.71/0.23 vs ckpt20 data 0.956/0.880/0.784/0.025 — uniform 0.025 vs claimed 0.23 differs ~10×, may correspond to ckpt30=0.195 instead). Need to verify which exact ckpt = 100M for each cond.
- M3: 411 Gibson + 61 MP3D scene counts — datasets not on local FS to verify; standard Habitat numbers but worth a one-time cross-check
- M4: 32-d sensor + 32-d action embedding — DD-PPO Wijmans-default but worth a one-time code grep when the relevant module is loaded
- M5: 20 scenes × 10 pairs shortcut — direct number not found in eval scripts; should verify

**§1 cited PDFs verified (commit 9c354e3):**
- ✅ Ramakrishnan2025 SPACE — supports VLM-fail claim ("near chance level on classic animal-cognition tests")
- ✅ Wirth2017 gaze — supports place–view–action joint coding under foveation ("hippocampal activity was best fit by a fine-grained state space comprising current position, view, and action contexts")
- ✅ Rolls2024 view cells — supports primates-have-view-cells-because-of-foveation claim
- ⚠️ FIXED: GevaSagiv2016 — paper actually shows non-overlapping populations only in SUBICULUM (16/27 cells) while CA1 mostly does place-field shifts. Reworded §1 ¶2 from "non-overlapping hippocampal populations" → "modality-dependent hippocampal remapping (modality-specific subicular populations + CA1 place-field shifts)".
- ⚠️ FIXED: Kupers2014blind — review focuses on visual cortex cross-modal recruitment, NOT specifically hippocampal coupling. Added Fortin2008blind ("Wayfinding in the blind: larger hippocampal volume and supranormal spatial navigation") for the hippocampal-specific claim. Kept Kupers as broader cross-modal context citation. Fixed at both §1 ¶2 and §5.2 H1 bio precedent ¶.
Update this file when state changes — do NOT rely on memory.

---

## 0. Quick status (at-a-glance)

| Dimension | Status |
|---|---|
| Paper version | **v3** (cog sci/neuro pivot done; 41 pages; place-cell signature + comparative cognition + LOSO + multi-seed Table 2 + systematic convergence criterion landed) |
| Page count | **41 pages** (was 31 in v2); compiles with pdflatex (no errors/warnings) |
| Submission deadline | **2026-05-06** (6 days out) |
| Cluster: Izar jobs | 1 PENDING: uniform_gibson_ckpt38 probe (job 2867550, blocked by INJ_MAINTENANCE 08:00–12:00 today) |
| Cluster: RCP jobs RUNNING | 4: foveated-seed0pre (4-GPU torchrun, ~2090 fps actual, ETA ~5h to 70M+180M=250M-eff), coarse-s1 num_env=8 (~5070 fps, ETA ~12h), logpolar resumed (~1349 fps, ETA ~30h, monitoring SIGSEGV recurrence), uniform-s1 (4-GPU, just resubmitted, may go Pending) |
| Most recent landed | **§5.2 NEW**: Place-cell signature paragraph + Fig fig_place_cells.pdf (Skaggs MI per LSTM unit; blind 0.20 vs rich-encoder 0.10–0.12; broader spatial-coding population in blind: 504/512 vs ~422–434). **§5.2 NEW**: Sensory-niche framing across taxa (mole-rat / coarse-scalar / primate / insect-eye). **§5.4 NEW**: Achille-Soatto IB-invariance reading. **§4.2 NEW**: LOSO scene-invariance paragraph + fig_loso_cv.pdf as primary CAP mechanism. **§3 NEW**: systematic convergence criterion (max success/SPL/DtG plateau). **Table 2 NEW**: multi-seed (seed-0 + seed-2 × 4 conds). |
| Most recent paper change | Place-cell signature analysis (Skaggs adapted to LSTM hidden states): Blind has highest per-unit spatial MI and largest spatial-coding population, consistent with capacity-allocation under encoder bottleneck. |
| Critical blocker resolved | NaN bug surfaced on seed-0 foveated at ckpt.36 ($> 174$M frames cause silent weight corruption); fine-tune resume from ckpt.36 with `pretrained=True` + `lr=5e-5` + `total_num_steps=70M` (effective 250M total) running on RCP 4-GPU. |
| Diagnostics in flight | logpolar SIGSEGV recurrence watch (resubmitted ~28 min ago, healthy so far). |

---

## 1. Cluster jobs (current snapshot)

### 1.1 Izar running (snapshot 2026-04-28 21:30)

| Job ID | Name | Elapsed | Output |
|---|---|---|---|
| 2853101 | `uni_s2` multi-seed | ~36h, ~30h to wall | `uniform_gibson_seed2/` |
| 2853102 | `fov_s2` multi-seed | ~36h, ~30h to wall | `foveated_gibson_seed2/` |

`auto_resume.sh` keeps these alive across 72h walltime cycles.

### 1.2 RCP running — full Tier 1 + paper-critical Tier 2 + multi-seed (2026-04-28 21:30)

After EGL/fork blocker resolved via custom Docker image (`registry.rcp.epfl.ch/dhlab-wxu/habitat:v2`), RCP unblocked. 12 jobs in flight as of last check:

| RCP job | hc plan | Purpose | ETA |
|---|---|---|---|
| `dh-probe-8` | T14 fov-shifted | H3 static control (paper-critical) | ~7h |
| `dh-probe-11` | (scene_occ optimized v2) | WJ-C Stage 1 — fast version (~30s/scene vs old ~12min) | ~3h |
| `dh-probe-12` | T1 stoch-gaze σ-test | Diagnostic with σ_max=0.02 quasi-deterministic | ~6h |
| `dh-probe-13` | T2 K=64 | Scaling sweep (matched64 config) | ~7h |
| `dh-probe-14` | **T9 log-polar ⭐** | **Falsifiable H1 mechanism test** | ~7h |
| `dh-probe-15` | T10 fov-v2 | NaN-bug clean rerun | ~7h |
| `dh-probe-16` | T3 K=32 | Scaling sweep low | ~7h |
| `dh-probe-17` | T6 K=96 | Scaling sweep mid-high | ~7h |
| `dh-probe-18` | T7 K=192 | Scaling sweep high | ~7h |
| `dh-probe-19` | T4 blind seed=2 | Multi-seed, 342M frames | ~10h |
| `dh-probe-20` | T5 coarse seed=2 | Multi-seed | ~7h |
| `dh-probe-21` | T8 fov-learned seed=2 | Multi-seed (biggest paper-impact risk) | ~7h |

**`H100-*` no longer relevant**: friend's H100 was previously critical-path for H3 fov-shifted (T14) and scaling sweep (T2/T3/T6/T7). Now all running on RCP. Friend can pick up Tier 2 (T6/T7 already covered) or Tier 3 (σ-strength sweep T11/T12/T13, foveation completeness).

### 1.3 Stochastic-gaze nan diagnostic in flight

T1 `foveated_stochastic_gibson` (probe-5 → killed at update 240+, then probe-12 with σ_max=0.02). Probe-5 reward went `inf` → `nan` at update ~120 and stayed nan. Hypothesis: stochastic-gaze sampling occasionally produces "stuck pose" via random-action exploration, geodesic distance becomes `inf`, reward computation returns `nan`, gradients corrupt policy. Diagnostic test (probe-12, σ_max=0.02 quasi-deterministic): if healthy → σ size matters → can do annealing. If still nan → habitat-baselines structural issue.

---

## 2. Experiment registry

All paper-v1 backbone experiments (5-cond probes, transplant, shortcut, CKA, etc.) are integrated. Removed for conciseness — see §6 decision log for history.

Outstanding experiments organised by §5.7 (Wijmans replication, currently active) and below:

### 2.4 Future / out of v1 scope

| ID | Name | Notes |
|---|---|---|
| O1 | Direct H1 mechanism test (GPS perturbation mid-rollout) | Future work, listed as TODO in §4.2 "Why" |
| O2 | Cross-architecture transformer baseline | Major work; future paper |
| O3 | Length-matching ablation in training (truncate trajectories) | TODO in §3.3 |

---

## 3. Paper claim audit (per-section)

For each major claim: what data backs it, status of integration.

Status legend:
- ✅ Backed by deterministic data, integrated, hedged appropriately
- ⚠️ Hedged with "awaits replication" or TODO marker; needs follow-up
- 🆕 New finding from in-flight experiments; needs integration once data lands
- ❌ Not yet supported

### 3.0a Mechanism claims explicitly NOT supported by current data

Listed here so we don't drift into asserting them:
- **Direct causal test of encoder-memory race** (mid-rollout visual ablation in rich-encoder agents → does top-layer GPS code re-emerge?). Not run. Listed as TODO in §4.2 "Proposed mechanism".
- **"Spatial-feature variety per step" as the trigger**, vs. competing accounts (input resolution; encoder spatial output dim; encoder channel capacity). Resolution scaling sweep (App E, friend's H100) is the clean test; until it lands the framing is interpretation.
- **Architecture independence** of the principle (transformers, supervised learners, non-navigation). Untested. Hedged as such in paper.
- **Foveation specifically diverging from uniform "in subspaces invisible to a linear-GPS probe"** — this is a description of what we observed, not a tested claim. Subspace structure of the divergence is unprobed.
- **Foveated-shifted control predicts H3 effect** — the test is in training (H100-A); the prediction is open.

### 3.0b Methodological oversimplifications (paper has \pendnote flagging each)

Choices that could materially shift findings if revisited; flagged inline
as `\pendnote{...}` and listed here for follow-up audits.

| # | Choice | Where | Risk to claims | Tighter test |
|---|---|---|---|---|
| O1 | Foveation = Gaussian blur (σ_max=8 quadratic falloff) | §3.1 fn | High — Gaussian preserves encoder spatial output (8×8); biologically faithful foveation (log-polar / multi-scale) reduces it. If F3 log-polar matches uniform, the H1 mechanism story (encoder spatial output) needs reframing | F3 log-polar in training; multi-scale pyramid F-future |
| O2 | Top-layer h_2 only probed | §3.2 fn | Medium — c_t and h_0, h_1 not probed in main figs; if condition-specific structure lives in c_t but not h_2, H2 "disjoint subspaces" claim narrows | **2026-04-26 in flight (analyse_extra_states.py): Ridge probe on h_0/h_1/h_2/c_0/c_1/c_2 for all 5 conds.** Full CKA/transplant pipeline on those layers still future |
| O3 | Transplant single midpoint t=30 | §3.2 fn | Low (supp sweep already partly addresses) — sweep stable for t≥15; t≥100 (late-episode) not yet measured. Could matter if late-episode pattern differs | **2026-04-26 RESOLVED**: extended supp Fig 8 to {0, 15, 30, 60, 90, 200, 400, 800}; cross-self gap peaks in t∈[30,90] window and decays toward 0 at t≥400. t=30 is now justified, not arbitrary. Commit `ec212e1` |
| O4 | "Cognitive map" used loosely (= linear-decodable GPS code) | §1 Intro fn | Medium — much narrower than the broader cognitive-map concept (allocentric, relational, multi-scale, hierarchical). Limits scope of "we found cognitive maps" claim | Non-linear / relational probes (deeper MLP, kernel, contrastive) on h_t to test for richer structure |
| O5 | DD-PPO + PointGoal + Habitat + ResNet-18 (algo / task / sim / encoder) | §3.1 fn | Medium-high (scope) — H1/H2 patterns could shift under different RL algos, navigation tasks, simulators, or visual encoders. Each is a deliberate scope choice for Wijmans-comparability | Re-run 5-cond ablation on (a) ObjectGoal, (b) iGibson sim, (c) ViT or smaller-CNN encoder. Each is a separate paper |
| O6 | Linear CKA only, Ridge α=10 only | §4.3 + §3.2 implicit | Low — non-linear measures could reveal higher-order shared structure (already in §4.3 Boundaries); α sensitivity untested | Sweep α ∈ {0.1, 1, 10, 100, 1000}; aligned CKA; CKA with non-linear kernel |
| O7 | Episode-level 5-fold CV (vs. trajectory-balanced or scene-balanced) | §3.2 implicit | Low — under deterministic rollout the splits are reasonably balanced; could under- or over-estimate variance for short-/long-episode-dominant conditions | Trajectory-step-stratified CV; scene-level CV |
| O8 | Foveated (fix) at ckpt.36 (~174M frames), other 4 at final ckpt | §5.5 (ii) | Medium — foveated under-trained relative to others; if its converged ckpt would shift H1 numbers, the bottleneck-vs-rich partition sample for foveation is biased | Re-run foveated to 250M (clean retrain in flight as F1 control) |
| O9 | Behavioral metric = SPL drop only | §4.5 implicit | Low — SPL captures both success and path-efficiency. Other metrics (path length, time-to-goal, hesitation count) might tell a different per-condition story | Measure full behavioral signature: SPL, path length, decision entropy, hesitation rate |
| O10 | Encoder feature-map probe = compressed flatten (d=2048), not raw spatial map | §4.4 implicit | Low-medium — the compressed flatten loses some spatial structure that a raw 8×8×C feature-map probe might recover. If raw probes change the "no encoder linearly decodes GPS" claim, H1 mechanism reframes | Probe raw post-ResNet-18 spatial feature map directly (no compression) |
| O11 | Foveated-learned gaze collapsed to single point (0.49, 0.62) | §4.6 H3 | High (for H3) — H3 evidence depends on this single gaze location being representative; other learned gaze points might give different results. The foveated-shifted control IS the test, but only at this one location | Sweep gaze locations (e.g.\ 9-point grid in [0,1]²) with hardcoded foveation, see if pattern is gaze-location-monotonic |



---

## 4. Figure registry (current)

Figures pass-2 redesigned 2026-04-27 (Times font via `_style.py`; Makefile build; 28-29 pages).

| Filename | Section | Status |
|---|---|---|
| `fig1_setup.pdf` (with topdown floor plan) | Fig 1 (§1, §3) | ✅ |
| `fig2_h1_mega.pdf` (3-panel: bars / temporal / per-layer + MLP zone) | Fig 2 (§4.2 H1) | ✅ |
| `fig3_substitution_dynamics.pdf` (2-panel GPS/Compass training-dyn) | Fig 3 (§4.2 H1) | ✅ |
| `fig4_h2_probe_transfer.pdf` + `fig4_transplant_5x5.pdf` (uniform aspect) | Fig 4 (§4.3 H2) | ✅ coarse col/row 6/8 cells, last 2 in flight |
| `fig5_shortcut_canonical.pdf` (1×4 with map backgrounds) | Fig 5 (§4.4 behaviour) | ✅ |
| `fig6_synthesis_2axes.pdf` (3-axis quadrant scatter) | Fig 6 (§4.6 synthesis) | ✅ |
| `appfig_shortcut_catalog.pdf` (4×4 paired-traj catalog with maps) | App (§4.4 extended) | ✅ |
| `appfig7-12.pdf` (training curves, t-SNE, CKA, transplant sweep, extra states, pop coding, goal vector, layerwise) | App A | ✅ |

**Pending placeholders** (data in flight):
- F1-F4 σ-sweep, F3 log-polar foveation figures (scripts ready, await training)
- Foveated-shifted control figure (awaits H100-A retrain)
- Encoder-resolution scaling sweep figure (awaits H100-B)

**Wijmans replication figures** (§5.7, queued): probe-agent SPL bar chart, memory-length sweep curve, occupancy decoder visualization, t-SNE per-condition, Bug-baseline summary, excursion-forgetting analysis.

---

## 5. Open questions / TODOs

### 5.1 Awaiting cluster results (in flight on Izar)

| Code | Experiment | What it answers | Paper § |
|---|---|---|---|
| F1-F4 | Foveation σ-sweep $\{2, 4, 12, 20\}$ + F3 log-polar + F-norm normaliser | Encoder-memory race as continuous lever; F3 log-polar prediction $R^2 \geq 0.3$ | §4.4 (App D placeholder) |
| Multi-seed | uniform_seed2 / foveated_seed2 (running) + blind/coarse/foveated_learned seed=2 (queued) | All single-seed claims gated on N=2 replication | All §4 numbers |
| Transplant tail | Last 2 coarse-recipient cells (blind→coarse, uniform→coarse) | Final 2 cells of Coarse column in Fig 4b | Fig 4b |

### 5.2 ⚠️ Friend's H100 — REQUIRED experiments (block paper, see §1.2 above)

Reduced to 2 experiments (multi-seed moved off H100 to Izar 2026-04-26).

| Code | Experiment | What it answers | Status |
|---|---|---|---|
| **H100-A** | Fov-shifted causal H3 retrain (clean transform) | Populates §4.4.4 + §4.5 H3 causal test | ⚠️ **NOT STARTED** — critical path |
| **H100-B** | Encoder-resolution scaling sweep ($32, 48, 64, 96, 128, 192$ at fixed encoder stack) | App E: causally tests the encoder-spatial-output mechanism by varying encoder resolution while holding everything else fixed | ⚠️ **NOT STARTED** — critical path |

### 5.3 Should-have experiments — not yet planned, would strengthen claims if added

For each: which paper claim is currently held by hedging that this experiment would tighten.

| Code | Experiment | What it tightens | Effort estimate |
|---|---|---|---|
| H1-causal | Train rich-encoder agent with GPS perturbed mid-rollout; or train bottleneck agent with GPS sensor removed | "Encoder–memory race" from candidate unifying account → mechanism (currently softened in §5.1) | 2-3 retrains; ~4 days V100 each |
| Pol-rely | Ablate GPS sensor mid-rollout at eval and measure SPL drop | Distinguishes "policy reads GPS code" from "policy reads non-GPS memory" — directly tests the coarse-vs-uniform 2×2 anomaly | Eval-only; ~2 hours per cond |
| Multi-seed-shortcut | Re-run shortcut eval on multi-seed checkpoints | Promote 2×2 dissociation (coarse + uniform anomalies) from candidate to robust finding | Depends on F5a/F7 |
| Length-match | Truncate every condition's probing data to common length, re-run probes | Eliminates "long blind episodes give artificially high R²" hypothetical confound | Eval-only; ~1 hour |
| 1-NN-large-N | 1-NN purity at $\sim 50$k pooled samples (currently 7500) | Bounds the "1-NN purity = 1.000" finding's sample-size effect | Analysis-only; few minutes |
| 1-NN-MP3D | 1-NN purity on MP3D-pooled hidden states | H2 robustness to dataset shift | Analysis-only; needs MP3D NPZs |
| Cross-head-N | Cross-heading probe with larger N (was tried, returned "insufficient samples") | Heading-invariance of GPS code | Bigger probe collection job; ~4 hours |
| Hybrid-sensor | Hybrid sensor (coarse-uniform-periphery + sharp-fovea, OR depth-only) | "Hybrid sensors might produce intermediate geometries" hint in §4.3 | New training condition; ~4 days V100 |
| Banino-grid | Test grid-cell periodicity / hexagonal autocorrelation in LSTM units | Compare to grid-cell literature directly | Analysis-only; ~1 hour |
| Deep-lag | Lag-$k$ probe to $k=50, 100$ | Bound the "persistent at lags ≥20" claim | Analysis-only |

### 5.3.1 Cog sci / neuro pivot — proposals from deep-research (2026-04-30)

The paper is being repositioned for a cog-sci / neuroscience reviewer audience to compensate for the simple architecture (LSTM + ResNet18 + DD-PPO). Strategy: deep-RL agents as **model organisms** for testing cognitive-map / capacity-allocation theory, with neuroscience-standard analyses.  Tier-1 = high-impact & implementable on existing checkpoints; tier-2 = high-value supplementary; tier-3 = if time.

| Code | Proposal | Effort | Status | Tightens / what it adds |
|---|---|---|---|---|
| **A1** | Skaggs spatial-info bits per LSTM unit (with shuffle null) | 0.5d | **DONE 2026-04-30 02:50** (see Fig fig_place_cells.pdf, §5.2) | Place-cell signature: blind has higher per-unit MI ($0.20$ bits) and larger spatial population ($504/512$). Mirrors enhanced occipital--hippocampal coupling under sensory deprivation. |
| **A2** | Cross-scene preferred-bin preservation (rate-vs-global remapping analog) | 0.5d | **DONE 2026-04-30 02:50** (panel c of fig_place_cells.pdf) | Cross-scene Spearman ρ; coarse the only cond with non-trivial preservation ($+0.15$ vs others $+0.05$–$+0.07$). Nuance: bottleneck $\neq$ stable across rooms in normalised coords. |
| **E1** | Comparative-cognition reframe: 4 conds = 4 sensory niches | 0.25d | **DONE 2026-04-30 02:50** (§5.2 NEW Sensory-niche framing paragraph) | Blind=mole-rat, coarse=acoustic/scalar, foveated=primate, uniform=insect. Cite Geva-Sagiv 2015, Toledo 2020, Kimchi 2004. Pure narrative, 0 compute, big impact. |
| **D1** | MI capacity accounting (MINE estimator) of $I(h; \text{pos})$, $I(h; \text{pos} \mid \text{enc})$, $I(\text{enc}; \text{pos})$ | 1.5d | **PLANNED — Tier 1 must-do** | The CAP equation is currently a hypothesis. With MINE numbers showing constant total $\approx$ shifted split, CAP becomes empirically grounded. Direct test of Achille-Soatto IB framing. |
| **F1** | Activation patching (cross-cond representational transplant) | 1.5d | **PLANNED — Tier 1** | Causal complement to current correlational cross-cond transfer. Reviewers expect ≥1 causal experiment for NeurIPS 2025+. |
| **B1** | Detour test (Tolman 1948 in silico) | 1d | **PLANNED — Tier 1** | The cognitive-map litmus test. Foveated > Blind on detour would directly falsify CAP. Modern citation: Behrens 2018 ``What is a cognitive map?'' |
| **C1** | Predictive-horizon probing — decode $(x,y)_{t+k}$ from $h_t$ for $k = 1, 5, 10, 20, 50$ | 1d | **PLANNED — Tier 2** | Tests SR (Stachenfeld 2017) / TEM (Whittington 2020) predictions; adds temporal axis to current static probe story. |
| **G1** | Eigenspectrum power-law slope (Stringer 2019) | 0.5d | **PLANNED — Tier 2** | Refines the ID $\approx 3$ result; spectrum slope discriminates conditions even when ID is similar. |
| **F3** | Causal scrubbing of GPS-coding subspace (null-space project Ridge $\beta$, re-rollout) | 1d | **PLANNED — Tier 2** | Direct test of "which conditions \emph{causally use} the linear GPS subspace". |
| **G2** | Williams shape metric for cross-cond manifold distance | 1d | **PLANNED — Tier 2** | The rigorous answer to "are these representations the same?" Preferred over CKA in 2024+ literature. |
| F2 | Sparse autoencoder on $h_t$ per cond | 2d | Tier 3 (if time) | Interpretability lingua franca (Bricken 2023, Templeton 2024). |
| A3 | Grid-cell hexagonal periodicity test (Sargolini 2006) | 1d | Tier 3 (defensive) | Defends against the obvious Banino-2018 reviewer ask; expected NULL since our task lacks path-integration loss. |
| C2 | SR-eigenmap probe (compute empirical successor matrix; check $h_t$ aligns with eigenbasis) | 1.5d | Tier 3 | Stronger SR connection. |
| G3 | Topological data analysis (persistent homology Betti numbers) | 2d | Tier 3 (risky/cool) | Curto-Itskov style; bottleneck $\to$ clean room-topology Betti, rich-encoder $\to$ noisy. |
| D3 | Partial information decomposition (Williams-Beer) | 2d | Tier 3 | Sophisticated; decomposes (encoder, memory) $\to$ position into unique / redundant / synergistic. |

### 5.3.2 Epstein cog-map engagement — relational / topological analyses (PLANNED post-retrain)

**Why this section exists**: §1 footnote currently says "Full hierarchical/topological-graph analyses are left for future work" — engaging Epstein 2017's "relational/hierarchical" cognitive-map definition only via LOSO + predictive-horizon. If a quick-win (EM1+EM2) lands compellingly, that footnote becomes a Methods/Results paragraph with positive evidence rather than a hedge. Decision (2026-04-30): WAIT for unified-retrain h₂ NPZs to land before running, since pre-retrain checkpoints have known issues (foveated NaN-bug, varying frame counts, hp inconsistency). Quick-win EM1+EM2 then takes ~3h compute on clean data.

**Order of execution post-retrain**: EM1 (hierarchical clustering) → EM2 (same/diff scene classifier) → if both compelling, integrate into §4 Results as new subsection "Relational structure of the position code" → if time permits, EM3 (topological probe) → EM4 (graph reconstruction) deferred to journal version.

| Code | Analysis | Compute | Total time (incl. code+plot) | Tier | Engages |
|---|---|---|---|---|---|
| **EM1** | Hierarchical scene clustering of $\mathbf{h}_2$ | ~10 min CPU | ~1–2h | Quick-win | Relational binding (place→context) |
| **EM2** | Same-scene vs diff-scene pair classifier from $\mathbf{h}_2$ | ~20 min CPU | ~1–2h | Quick-win | Relational discriminability |
| **EM3** | Topological reachability probe (within-K-step) | ~30–60 min/cond × 5 | ~3–5h | Mid-add | Graph topology beyond metric |
| **EM4** | Full navmesh graph reconstruction from $\mathbf{h}_2$ | ~1–2h/cond × 5 | ~6–12h | Full / journal | Hierarchical map structure |

---

#### EM1 — Hierarchical scene clustering of $\mathbf{h}_2$

**Goal**: Test whether the policy-readable hidden state organises by scene (relational/contextual binding) vs by metric position alone. A rich-encoder agent that "uses visual landmarks as anchors" should cluster strongly by scene; a bottleneck agent that integrates GPS abstractly should cluster more weakly by scene (and more strongly by metric coordinate).

**Engages** Epstein 2017's "relational" axis: scene-conditional structure = the map binds local geometry to scene-level context.

**Inputs** (per condition):
- $\mathbf{h}_2$ pooled across $n=500$ Gibson held-out episodes (post-retrain NPZs at `/scratch/izar/wxu/probing_data/<cond>_det.npz`).
- Scene IDs per timestep (already saved alongside h₂ in same NPZ, key `scene_id` or `scene`).
- ~50–100 unique Gibson held-out scenes expected.

**Procedure**:
1. Subsample 5000 timesteps per condition (stratified by scene to avoid scene-frequency bias; min 20/scene).
2. Standardise each $\mathbf{h}_2$ feature (zero-mean, unit-variance) within condition to remove scale differences from the clustering.
3. Compute pairwise cosine + Euclidean distance matrices.
4. Run agglomerative clustering (Ward linkage, sklearn `AgglomerativeClustering`) for $k \in \{n_\text{scenes}/2, n_\text{scenes}, 2 n_\text{scenes}\}$.
5. Evaluate: ARI (adjusted rand index) and AMI (adjusted mutual info) between cluster labels and ground-truth scene labels.
6. Report per condition: ARI(scene), ARI(scene) at matched cluster count, dendrogram cophenetic correlation.
7. Repeat under shuffle null (shuffle scene labels) — report selectivity gap (real – shuffled).

**Output**:
- Table: 5 conditions × {ARI(scene), AMI(scene), gap-vs-null}; tucked into App or new Results subsection.
- Figure: dendrograms colour-coded by scene for blind vs uniform side-by-side (visual). Scene-labelled t-SNE companion (we already have t-SNE in App).

**Reuse**: `scripts/probing/cluster_quality.py` already does silhouette + 1-NN purity by *condition* — extend it to cluster by *scene*. New script: `scripts/probing/scene_clustering.py`. <50 LOC.

**Acceptance criterion (compelling)**:
- Significant condition-ranking on ARI(scene): rich-encoder ≥ 2× bottleneck ARI, with shuffle-null gap > 0.1 in rich-encoder. → Promote to §4.6 Boundaries new paragraph "Rich encoders bind position to scene context" + integrate Epstein footnote → in-text engagement.
- ARI(scene) ranking aligns with LOSO ranking from §4.6 (cross-validation: LOSO already shows scene-conditional rich-encoder code).

**Acceptance criterion (null)**:
- Conditions indistinguishable on ARI(scene), or ranking inverted vs LOSO. → Keep footnote as hedge; do NOT integrate. Note in App as "scene clustering ambiguous; LOSO is the cleaner relational test".

**Risk**: pooled h₂ across 500 episodes may have within-scene clusters dominated by episode-trajectory geometry rather than scene identity. Mitigation: subsample 1 timestep per episode before pooling to break trajectory autocorrelation.

---

#### EM2 — Same-scene vs different-scene pair classifier

**Goal**: Operationalise the relational claim as a binary classification: given two hidden states, can a simple model decide whether they came from the same scene? Cleaner reviewer-facing version of EM1 because it returns one number (AUC) per condition.

**Engages**: Same Epstein "relational" axis as EM1, with crisper statistical readout.

**Inputs**: Same NPZs as EM1.

**Procedure**:
1. Sample $N=10000$ pairs per condition: 50% same-scene, 50% different-scene (within-scene pairs sampled across different episodes when possible to avoid trajectory leakage).
2. For each pair $(\mathbf{h}_i, \mathbf{h}_j)$: compute pair representation $\mathbf{f}_{ij} = [|\mathbf{h}_i - \mathbf{h}_j|, \mathbf{h}_i \odot \mathbf{h}_j]$ (concatenate abs-diff and Hadamard product, standard pair-encoding for Siamese models).
3. Fit logistic regression on $\mathbf{f}_{ij} \to$ {same, diff} with 5-fold CV stratified by scene-pair identity.
4. Report: per-condition AUC, accuracy at threshold 0.5, calibration plot.
5. Baseline: shuffle-scene-label null; report AUC gap vs null.

**Output**:
- Table: 5 conditions × {AUC, AUC – null}.
- Figure: ROC curves overlaid (1 panel, 5 lines).

**Reuse**: New script `scripts/probing/scene_pair_classifier.py`, ~80 LOC. Reuses sklearn `LogisticRegression` + `roc_auc_score`.

**Acceptance criterion (compelling)**:
- Rich-encoder AUC ≥ 0.85 vs bottleneck AUC ≤ 0.65, with null-gap ≥ 0.3. → Integrate as a one-paragraph "scene-binding" subsubsection alongside LOSO with EM1. Updates Epstein footnote to a positive in-text claim.

**Acceptance criterion (null)**:
- AUC clusters tightly across conditions (e.g., all in [0.7, 0.85]). → Footnote stays hedged; report negative result in App.

**Risk**: AUC could be inflated by trivial trajectory leakage if pairs from the same episode dominate within-scene class. Mitigation: enforce different-episode constraint for within-scene pairs.

---

#### EM3 — Topological reachability probe

**Goal**: Decode "is state $s'$ reachable from state $s$ within $K$ navmesh steps?" from $(\mathbf{h}_s, \mathbf{h}_{s'})$ pairs. Tests whether $\mathbf{h}_2$ encodes graph topology (reachability/connectivity), not just metric distance.

**Engages**: Stachenfeld 2017 successor-representation predictive map + Epstein "topological" axis. Tighter than predictive-horizon (Appendix~\ref{app:predictive_horizon}, current) because it decouples reachability from raw metric distance.

**Inputs**:
- h₂ pooled over Gibson held-out episodes (post-retrain).
- Habitat navmesh per scene (ground-truth shortest-path distance via `pathfinder.find_path(s, s')`).
- $K \in \{5, 10, 20\}$ step thresholds.

**Procedure**:
1. For each scene, sample pairs $(s_i, s_j)$ from the same scene at varying ground-truth navmesh-step distance.
2. Label: $y_{ij} = 1$ if $d_\text{navmesh}(s_i, s_j) \leq K$; else $0$. Stratify pair sampling so 50% reachable, 50% not, balanced across $d_\text{navmesh}$ bins to avoid the trivial "Euclidean distance proxies for reachability" shortcut.
3. **Critical control**: Also sample pairs with the same Euclidean distance distribution but different reachability (e.g., across-wall pairs with low Euclidean distance but high navmesh distance). The probe must differentiate these → tests topology beyond metric.
4. Train MLP probe (2-layer, hidden 256) on pair representation $\mathbf{f}_{ij}$ → $P(y=1)$. Episode-level 5-fold CV.
5. Report per condition × $K$: AUC, AUC on the across-wall control set, AUC vs Euclidean-distance baseline (logistic regression on $\|\mathbf{p}_i - \mathbf{p}_j\|$).

**Output**:
- Table: 5 conditions × 3 K-thresholds × {AUC, AUC-control, AUC-vs-Euclidean-baseline}.
- Figure: bar chart per condition, K=10; control set vs main set.

**Reuse**: New script `scripts/probing/topological_probe.py`. Needs habitat navmesh access — ~150 LOC including pair sampling. Estimated ~30–60 min/condition (5 condition × 3 K = 15 fits at 2 min each + pair generation).

**Acceptance criterion (compelling)**:
- Rich-encoder AUC > Euclidean-baseline AUC by ≥ 0.1 on the across-wall control set in at least one condition. → Promote to "agent encodes navigable topology, not just Euclidean position" claim, integrate into §4 Results.

**Acceptance criterion (null)**:
- All conditions perform at Euclidean-baseline level on across-wall controls. → Honest negative result in App; supports "linear-readable position" framing without overclaiming topological structure.

**Risk** (high):
- Habitat navmesh may not align with the agent's actually-traversed graph (agents can clip walls, get stuck) → ground truth itself noisy. Mitigation: filter to clean episodes (success=1, no off-mesh steps).
- 30–60 min/cond compute estimate is rough; if MLP probe is large or pairs ≫ 10k, could blow up. Mitigation: cap $N=20000$ pairs; logistic regression baseline first before MLP.
- Interpretation ambiguity: even if rich-encoder agents can decode reachability, this might just reflect them learning visual landmarks of door/wall locations rather than abstract topology. Mitigation: note this as a confound in the paragraph.

---

#### EM4 — Full navmesh graph reconstruction (DEFERRED to journal version)

**Goal**: Reconstruct the navmesh adjacency graph of a scene from h₂ alone. Most ambitious — directly tests "the hidden state IS a cognitive map" rather than "encodes some property of the map".

**Procedure** (sketch, not committed to running for NeurIPS):
1. Sample $K=200$ candidate node states per scene (stratified by spatial location).
2. Use EM3-trained reachability probe (with $K_\text{step}=5$) to predict adjacency for all $K^2 = 40000$ pairs.
3. Build predicted graph; threshold edges at probe-prob > 0.5.
4. Compare to ground-truth navmesh graph: graph edit distance, spectral graph similarity (Laplacian eigenvalue $\ell_2$), per-edge AUC.

**Why deferred**:
- ~6–12h compute (5 conditions × 1–2h each), and depends on EM3 working.
- High risk of distracting from main "linear readability" finding.
- Reviewer ROI lower than EM1/EM2 because graph reconstruction is a well-known hard problem; partial reconstruction would invite "but what about the missing edges" critique.

**Acceptance criterion to revisit for NeurIPS**: Only if EM3 lands compellingly AND there is unused compute slack post-final-figures (≥1 day before submission). Otherwise queue for journal extension.

---

**Pre-build pipeline (~1h, can do BEFORE retrain finishes)**:
- `scripts/probing/scene_clustering.py` (EM1) — adapt `cluster_quality.py`.
- `scripts/probing/scene_pair_classifier.py` (EM2) — new, ~80 LOC.
- `scripts/probing/topological_probe.py` (EM3) — new, ~150 LOC, needs Habitat navmesh hookup verification.
- All 3 read `data/probing_results/<cond>_det.npz` (or post-retrain equivalent path); verify they expose `scene_id` per timestep before retrain finishes.
- Smoke-test on existing pre-retrain blind NPZ (we don't trust the numbers but the pipeline should run end-to-end).

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

### 5.7 Wijmans 2023 replication / extension plan (queued 2026-04-27)

After re-reading Wijmans et al. 2023 ("Emergence of Maps in the Memories of Blind Navigation Agents", ICLR), the following experiments port their methodology to our 5-condition setup. Each strengthens a specific claim in our paper. **Priority: do all in order. Deadline 2026-05-06.**

| Code | Experiment | Wijmans figure | Tightens our claim | Effort |
|---|---|---|---|---|
| **WJ-B** | **Probe agent**: train a 2nd agent (same arch as recipient) initialised with the recipient's final memory $(\mathbf{h}_T, \mathbf{c}_T)$, task it with SecondNav(S→T). Measure probe SPL vs agent SPL. | Wijmans Fig 3A/B | §4.5 dissociation: rich-encoder memory may be USEFUL even though linearly unreadable. If uniform/foveated probe SPL > agent SPL → memory contains policy-relevant info → "linearly unreadable ≠ useless". | 2-3 days; eval-only (no retrain) |
| **WJ-A** | **Memory-length sweep**: at eval time, clip the LSTM hidden-state to the last $k$ steps for $k \in \{1, 4, 16, 64, 256, 1000\}$; re-probe GPS at each $k$. | Wijmans Fig 2 | H1: bottleneck conditions need long-horizon memory to integrate GPS. Curve shape per condition shows whether the GPS code accumulates gradually or arises locally. | 1-2 days; analysis-only |
| **WJ-C** | **Occupancy grid decoder**: train a decoder to predict allocentric free-space occupancy maps from $(\mathbf{h}_T, \mathbf{c}_T)$ for each condition. Report IoU per condition + side-by-side ground-truth/predicted visualisation. | Wijmans Fig 4 | H1 mechanism: bottleneck encodes metric maps; rich-encoder does not. **Strongest reviewer evidence** for "memory contains a map" claim — currently we only show linear-probe R² which is more abstract. | 3-5 days; new decoder + train + per-condition analysis |
| **WJ-E** | **t-SNE per condition**: t-SNE of $(\mathbf{h}_t, \mathbf{c}_t)$ pooled across $n=500$ episodes per condition. Colour by action × collision-state (Wijmans 4-cluster) OR by distance-to-goal bin. | Wijmans Fig 1C | H2 visual demonstration: per-condition embeddings should look qualitatively different. Easy + impactful. | 0.5 days; analysis-only |
| **WJ-D** | **Bug algorithm baseline**: implement the clairvoyant Bug variant (always-right / always-left / oracle) on our scenes. Add to summary table. | Wijmans Fig 1B / Table 1 | Pre-empts "task too easy" reviewer concern; contextualises 96-99% success rates. | 1 day; classical algorithm + eval pipeline |
| **WJ-F** | **Forgetting / excursion analysis**: train decoder $f_k(\mathbf{h}_t, \mathbf{c}_t) \to s_{t-k}$ for $k \in [1, 256]$. Compare excursion vs non-excursion error per condition. | Wijmans Fig 5A/B | H2: do conditions differ in WHAT they remember? Foveated wandering trajectories may forget more than blind wall-following. | 2-3 days; per-condition decoder + excursion labelling |

**Order of implementation** (user-confirmed 2026-04-27): WJ-B (probe agent) → WJ-A (memory length) → WJ-C (occupancy grid) → WJ-E (t-SNE) → WJ-D (Bug baseline) → WJ-F (excursion forgetting).

Rationale: B + A directly strengthen the §4.5 dissociation and H1 (the story's weakest pillar); C is the highest-impact figure (occupancy maps as direct mechanism evidence); E/D are quick wins; F is the most exploratory.

### 5.6 Mined-from-existing-data side observations (single-seed; verify with multi-seed)
- **Persistent-failure terminal locations** (Table 4 in §4.5, commit `fab08b3`): only uniform's persistent-memory failures cluster around the previous-episode goal location (margin +1.83m); blind/coarse/foveated terminal positions are closer to the new goal but not at it (n=27/35/16 same-floor failures). This refines the "having-vs-using" 2×2 dissociation: uniform's memory anchors on visual landmarks; blind's memory interferes through position-mis-reporting rather than location-anchor.
- **LSTM gain** (mentioned in §4.4, commit `2a4b9fe`): LSTM top-layer GPS R² minus encoder feature-map GPS R² is +3.9 for coarse, +0.0 for uniform, +0.7 for foveated. Mid-magnitude foveated gain is consistent with foveation supplying less navigation-useful visual structure than uniform.
- **Failure-episode asymmetry** (NOT YET in paper, low confidence at n=12, single seed): 98 episodes uniquely fail in bottleneck conditions (rich-encoder succeeds), but only 12 episodes uniquely fail in rich-encoder conditions (bottleneck succeeds). Of the 12, scene 91 contributes 3, suggesting at least one "rich-encoder-unfriendly" scene. 6 of the 12 are short-geodesic ($<7$m), so failure isn't path-length-driven. Worth re-checking post multi-seed; if pattern holds, indicates rich-encoder agents have a small but non-zero failure mode that bottleneck doesn't share — visual landmark misreading on specific scene types.

---

## 6. Decision log (chronological, why we did what)

### 2026-04-29: Narrative pivot — modern-AI-puzzle-first framing

**Why pivot**: original narrative led with bio precedent ("hippocampal sensor remapping") and treated the agent's older LSTM architecture defensively. Reviewer concern: "so what?" / why does an old-architecture finding matter for current research? User directive: connect findings to modern visual intelligence — VLM spatial-reasoning failures, foundation visual encoders, transformer world models, embodied LLM agents — and frame the older architecture not as a limitation but as a deliberately controlled experimental chassis where the encoder is the only varying component (frontier systems' architectural complexity confounds encoder effects with everything else).

**Central re-frame**: visual perception and downstream memory are not separable modules but a co-trained system; the encoder's structure shapes what memory can encode about space (a content-level claim about the perception–memory interface). The bio precedent stays as supporting evidence, not as the main motivation. Implications point outward to (i) a content-level alternative to the training-distribution explanation of VLM spatial-reasoning failures, (ii) re-framing foveated-perception research as cognition-enabling rather than only compute-saving, and (iii) a falsifiable prediction for transformer-based world models and embodied LLM agents.

**Title**: "How Visual Sensor Shapes the Format of Spatial Memory in Navigation Agents". Alternative declarative form: "Sensor Structure Shapes the Format of Spatial Memory in Navigation Agents" (kept for now in question form; both fit narrative).

**Abstract restructure** (290→229 words after several rounds): single paragraph, modern-AI-puzzle opener (foundation encoders / VLMs / world models / embodied LLM agents share an implicit pipeline premise) → VLM spatial-failure as motivating empirical pattern → structural reading (encoder + memory as co-trained system) → 1-sentence bio precedent (foveated primates ↔ echolocating bats) → comparative chassis "varied along a single axis of encoder spatial bandwidth" (no enumeration) → 3 findings (counter-intuitive substitution, format incompatibility, recall-vs-use dissociation) with `\footnote` flagging findings as provisional pending finer memory-analysis experiments → 2-sentence implications (VLM-failure alternative + falsifiable prediction for transformers/world-models/embodied-LLMs). No `\citep{}` references in abstract.

**Intro restructure** (~700→~540 words, 5 paragraphs): ¶1 pipeline-vs-co-trained reframing (different angle from abstract; not verbatim repeat). ¶2 bio precedent compact 4-species (kept different scale from abstract's 1 sentence). ¶3 prior emergent-maps lit (Wijmans/Banino/Cueva) "typically reported one condition at a time" → comparative chassis framing → 4-cond high-level enumeration → "older architecture is precisely the point" closing. ¶4 (NEW, merged from former ¶4 hypotheses + ¶5 contribution): three measurements with concrete predictions for each (magnitude/H1, format/H2, behavioural memory-reliance/dissociation). ¶5 interpretive close (bottleneck-vs-rich-encoder split + foveation hybrid). Removed: explicit "We test two hypotheses" framing (felt over-engineered for ML audience; redundant with contribution paragraph; H1/H2 tags retained as section anchors).

**Fig 1 update**: bottom-row pipeline schematic replaced with `\fbox{\parbox}` placeholder ("to be drawn"); caption trimmed to drop the obsolete bottom-block description. All figures and tables changed from `[!htbp]`/`[h]` to `[!t]` (force top-of-page). Page count 31→30.

**Cleanup**: 5→4 condition consolidation finished — caught and fixed three stale "five" references at lines 159 ("All five agents"), 441 ("for all 5 conditions"), 541 ("5-condition spectrum"), and 215 ("Five conditions, five linearly disjoint memory geometries (H2)" → "Four conditions, four..."). All `(1×1)` / `(fix)` annotations dropped from figures and text. `foveated_learned` condition fully removed from main paper + supplementary.

**Releases**: HuggingFace `alunxu/spatial-memory-checkpoints` (public, 4 conds × 5 ckpts each, 829 MB, ultra-minimal README to protect idea). GitHub link kept for code.

**Loop check (WJ-A/D/C)**:
- WJ-A memory-length sweep: 5 conds × 5–7 K-points landed in `/tmp/wj_data/wja_v3/`; figure rendered at `fig/fig_memlen_sweep.pdf` (4 conds, K = 1…full episode). Modest finding: all conds need long memory to recover GPS; rich-encoder marginally faster mid-K saturation. NOT yet integrated to main.tex (numbers somewhat inconsistent with §4.2 substitution claim — discrepancy in protocol; flagged for follow-up).
- WJ-D Bug baseline: already integrated at line 159 (SPL 0.07, success 0.13).
- WJ-C scene_occupancy: 106 ground-truth occupancy maps landed (inputs to decoder); decoder itself not yet trained.
- Job 2862280 (Izar): analyze.py for blind ckpt5/25 + encoder_zeroed × 3 NPZs in flight.

**TODO from this pivot** (tracked in §5):
- Refresh Related Work §2 to genuinely engage with the new narrative (VLM spatial failures, foundation encoders, transformer world models, embodied LLM agents, encoder–memory interface lit). Currently §2 is organised around the OLD narrative (cognitive maps in RL nav + foveated vision + hidden-state probing methodology). Need fresh deep research; download relevant PDFs to `Project/literature/`; read carefully; rewrite without speculation/overclaim.

### 2026-04-28 (evening): RCP unblocked, Tier 1 trainings launched, paper revision pass 1+2

**RCP EGL/fork blocker resolved via custom Docker image** (commits between):
- Persistent issue: every smoke (smoke11–smoke25) crashed at first `env.step()` in
  rollout collection — `ForkServerProcess-1` died silently with EOFError /
  BrokenPipe. Non-trivial, EGL/fork interaction inside container.
- Fix: built custom image `registry.rcp.epfl.ch/dhlab-wxu/habitat:v2` via
  kaniko on RCP — preinstalls all libs habitat-sim needs at OS level (apt
  packages incl. `libgl1-mesa-glx`, `libegl1-mesa`, `libglib2.0-0`, etc.) plus
  conda habitat env + habitat-lab + torch CUDA + protobuf/pillow pinned.
- Verified via dh-spatial-smoke-v2b: forkserver child process started,
  habitat_sim loaded, Simulator constructed, env.step() succeeded.
- After Docker image: full Tier 1 + critical Tier 2 + multi-seed launched on
  RCP as `dh-probe-{8,11,12,13,...,21}`. 12 jobs in flight.

**Paper revision pass 1 (section-by-section)** — caption / main-text consistency:
- Captions stripped of interpretive content; reduced to panel descriptors +
  methodology. Interpretive claims moved to main text.
- Main text: avoid restating figure data; describe trends, claims,
  implications, reasoning, interpretation.
- Section title rewrites for Results: §4.1 "Comparable navigation skill across
  all five conditions"; §4.2 "Substitution: rich encoders crowd out the
  integrated GPS code (H1)"; §4.3 "Five conditions, five linearly disjoint
  memory geometries (H2)"; §4.4 "Foveation: same H1 readout as uniform,
  distinct elsewhere"; §4.6 "Gaze location as a candidate fourth content axis
  (H3)"; §4.7 "Boundaries: null results that bracket H1 / H2".
- Abstract: added Finding 3 (probe-readable vs policy-used dissociation);
  framing aligned with §4.2 title ("substitute" instead of "race"); added
  explicit "recurrent (LSTM) PointNav agent" case-study label.
- Intro ¶6 (supplemental refs) deleted to remove appendix refs from intro.
  Intro & Conclusion no longer reference any appendix (per user feedback).

**Paper revision pass 2 (cross-section consistency)**:
- 0 undefined refs / 0 ?? in PDF.
- 1 unnecessary `\TODO` (length-matching ablation) removed.
- 1 legitimate `\TODO` (App E scaling sweep, "Pending data") kept — will fill
  in after dh-probe-13/16/17/18 land.
- 11 `\uncertain{}` blocks (intentional hedging) kept.
- 8 active `\hcpending{}` claims; mapped to RCP jobs in flight.

**WJ-A K-truncation methodology confound diagnosed**:
- `--reset-every K` actively zeroes LSTM hidden state every K steps. NOT a
  probe-time filter. This creates two confounds: (1) post-reset 1-step hidden
  states are sensor-passthrough (any condition with GPS sensor scores ~0.9
  R² at K=1, can't distinguish conditions); (2) at K=large, dataset mixes
  post-reset transients with long-accumulation states, probe trained on
  bimodal distribution overfits — matched K=1000 gives R² = -1.51 vs Table 1
  no-reset = +0.78.
- Decision: WJ-A figure does NOT enter paper. §4.2 "memory-length sweep"
  remains commented out. The H100-B encoder-resolution scaling sweep is the
  cleaner causal H1 test (different agents trained from scratch at different
  K, no reset intervention).
- Tried v2 (probe-time filter on full-episode NPZ): episode-level CV gives
  negative R² for all conditions (cross-scene generalization fails);
  step-level CV would match Table 1 baseline but trivial leakage. Probably
  needs separate paper.

**WJ-F variance-matched analyzer (v2) reveals real forgetting signal**:
- v1 (per-segment 5-fold R²) showed all conditions recover>warmup with
  Δ +0.14–0.23 — variance artefact (R² normalised by target variance).
- v2 (`scripts/eval/excursion_analyze_v2.py`): one Ridge probe trained on
  full-episode pooled hidden states (episode-level 5-fold CV), tested per
  segment with MAE / position-spread (scale-invariant).
- Result: all 4 conditions show forgetting; blind +0.17 (smallest, sensor
  passthrough robust), uniform +0.31, foveated +0.34, coarse +0.43 (largest,
  integrated GPS map fragile to action-stream noise).
- Integrated to paper §4.7 with new figure `appfig_wjf_excursion.pdf`.

**WJ-E t-SNE figure** (`appfig_tsne.pdf`) generated from existing Izar
subsamples and integrated to paper appendix.

**Build pipeline switched to `xelatex`**: TeXLive 2026/Homebrew has a
non-deterministic segfault during 3rd-pass `pdflatex` (hyperref `\pdfendlink`
nesting on figure floats). `xelatex` builds cleanly. Makefile updated.

---

### 2026-04-25 (evening): Encoder feature-map probe + Phase B + 5×5 + H1 mechanism refinement

Big batch of cluster results landed late afternoon / evening:

**Encoder feature-map probe — 3/3 conditions** (commit `409e6bd`):
- coarse encoder→GPS R² = -3.14 ± 5.91
- uniform encoder→GPS R² = -0.32 ± 0.08
- foveated encoder→GPS R² = -0.65 ± 0.28

NONE of the 3 sighted encoders linearly decode GPS. This was a more interesting outcome than the original 3 hypothesised possibilities. Sharpened H1 mechanism wording across abstract / §4.2 / §5.4 (commit `cc1b7e7`):

OLD framing: "rich encoder can re-derive position from current frame; bottleneck encoder cannot → LSTM compensates"
NEW framing: "no encoder linearly preserves GPS; what determines whether LSTM compensates is the encoder's spatial-feature variety per step (coarse 1×1 vs uniform 8×8). LSTM Layer 0 reads GPS sensor directly; bottleneck conditions integrate it across time because they have minimal visual feature variety to substitute"

**Phase B paired-trajectory figure — 5/5 conditions** (commit `d97eda9`):
- New eval script `scripts/eval/shortcut_with_trajectories.py` saves per-episode positions
- Figure shows "memory locks onto old goal" failure mode: blind's persistent run oscillates around previous-episode goal location; uniform/foveated wander more diffusely
- Visualises the §4.5 2×2 dissociation finding behaviorally

**5×5 transplant matrix — 33 cells** (commit `6ecc508`):
- Replaces the 3-pair midpoint sweep as Fig 3 right panel
- 13/16 cross-pairs at midpoint=30 visible; 4 coarse-recipient cells pending
- New findings: asymmetry (blind→uniform -0.38 vs uniform→blind +0.02), recipient ranking (uniform suffers most from foreign donors, blind least)
- Old transplant_sweep moved to App A as midpoint-stability evidence

**Topdown render fix** (commit `01c4933`):
- Fig 1d trajectory overlay now on actual Habitat occupancy map for the scene (E9uDoFAP3SH)
- Required 3 attempts: env._episode_iterator hack didn't work; env._dataset.episodes hack didn't work; finally fixed by setting config.habitat.simulator.scene + passing filtered dataset explicitly to habitat.Env

**F3 log-polar prediction** (commit `70bd1fd`):
- Wrote falsifiable prediction in §4.4.3: log-polar should give LSTM GPS R² ≥ 0.3 (because encoder 2×2 spatial output, between coarse 1×1 and uniform 8×8). If mechanism is encoder spatial-output dimensionality, this should manifest in the LSTM
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
  with higher action entropy (fov-fix, uniform, coarse, blind under their
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
| "More pixels ≠ better decoding" | ✅ Stronger | Coarse 1×1 R²=0.78 vs uniform R²≈0 |
| H2: representational format divergence | ✅ Survives | CKA / transplant / transfer all behavioural-grade |
| H3: fov-learned compass 0.94 | ❌ Bug artefact | Det CV R² = -1.34 ± 3.14 |
| "LSTM learns cognitive map" | ✅ Strengthened | Blind 0.95, coarse 0.78 |

**The TRUE finding** (replacing the original claim):
"Visual encoder capacity inversely determines linear top-layer LSTM spatial
encoding."  Decoupled cleanly via the coarse (matched-compute) condition
(visual input present, but encoder feature map is 1×1 — bottleneck
without "no vision").

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
  conditions (fov-fix, blind, coarse, uniform) unaffected; fov-learned
  and fov-shifted require retrain (delegated to friend's H100,
  `foveation_transform_fix_retrain.md`)**.
- **G3, B2, B3, B5, C3, C4, D5, etc.**: minor / documentation-only fixes,
  no claim impact, all addressed in commits.

### 2026-04-24: Paper reframe
- Original H1 ("foveated > uniform compensatory memory") busted under deterministic data. Reframed to encoder–memory race using coarse (matched-compute) as decoupling condition.
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
git log --oneline -10 -- docs/manuscript/main.tex

# Figure freshness (date-sorted)
ls -la docs/manuscript/fig/*.pdf | sort -k 6,8

# Pending det probe analyses
ssh izar "ls /scratch/izar/wxu/probing_results/*_det_analysis.json | wc -l"
```
