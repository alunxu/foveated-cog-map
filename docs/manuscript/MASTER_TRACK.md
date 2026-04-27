# Master Track — manuscript submission

Single source of truth for: cluster jobs, experiment status, paper
claims, figure freshness, open questions, decision log.

**Last updated**: 2026-04-27 06:50 (Wijmans replication plan added §5.7).
Update this file when state changes — do NOT rely on memory.

---

## 0. Quick status (at-a-glance)

| Dimension | Status |
|---|---|
| Paper version | v1 polished; figure pass complete (Fig 2-6 + appendix); refs working; 28-29 pages |
| Page count | **28-29 pages**; build clean via `make` (Makefile in docs/manuscript/) |
| Submission deadline | **2026-05-06** (9 days out) |
| Cluster: jobs RUNNING | 2 trainings (uni-s2 / fov-s2) + transplant tail (matched-recipient cells, last 2) |
| Cluster: jobs PENDING | 0 |
| Most recent landed | All 10 topdown maps (Fig 5 backgrounds) ✓; 6/8 matched transplant cells ✓ (patch validated) |
| Most recent paper change | Footnote color audit (only pending claims red); Makefile for reliable bibtex; Fig 5 redesigned 4×4 + maps |
| Next experiments queued | WJ-B (probe agent) → WJ-A (memory length) → WJ-C (occupancy decoder) → WJ-E (t-SNE) → WJ-D (Bug baseline) → WJ-F (excursion forgetting). See §5.7. |

---

## 1. Cluster jobs (current snapshot)

### 1.1 Running (verify with `ssh izar "squeue -u wxu"`)

Snapshot 2026-04-27 06:50:

| Job ID | Name | Elapsed | Output |
|---|---|---|---|
| 2853101 | uni_s2 multi-seed | ~4h (cycle ~2d total) | `uniform_gibson_seed2/` |
| 2853102 | fov_s2 multi-seed | ~4h | `foveated_gibson_seed2/` |
| 2857277 | uniform→matched transplant | ~45min | `/scratch/izar/wxu/transplant_results/` |
| 2857272 | blind→matched transplant | ~45min | (matched-recipient column completion) |

Cron `auto_resume.sh` keeps uni-s2/fov-s2 alive across 72h walltime cycles.
Cron `probe_hc_arrival.sh` waits for friend's hc-trained checkpoints.

### 1.2 Friend's H100 — REQUIRED experiments (separate cluster, blocks paper)

These experiments need H100 because they require new training runs (2-7 days each on V100, faster on H100) AND we want to avoid splitting/merging multi-cluster results.

| Code | Experiment | Why needed | Paper section affected | Effort estimate |
|---|---|---|---|---|
| **H100-A** | Fov-shifted causal H3 retrain (clean transform) | Clean H3 causal control: foveated agent with hardcoded gaze at $(0.49, 0.62)$, the collapsed-gaze location | §4.5 H3 (currently `\TODO{Training in progress}`) + §4.4.4 in App D | 1 retrain × 2--3 days V100 |
| **H100-B** | Encoder-resolution scaling sweep ($32, 48, 64, 96, 128, 192$ pixels at fixed encoder stack) | Tests H1 mechanism causally — directly varies encoder spatial output dimensionality with everything else held fixed | App E (currently `\TODO{Pending data}`); strengthens §4.2 mechanism + §5.4 implication (i) | 6 retrains × 2--3 days V100 = 12-18 V100-days. **H100 strongly preferred for parallelism** |

**Removed from H100 list (2026-04-26)**: ~~H100-C Multi-seed gap fills~~ — decision to keep all multi-seed work on Izar to avoid split/merge complexity. Submitted blind/matched/foveated_learned seed=2 retrains via Izar normal QOS (jobs 2850374-76); uniform/foveated seed=2 already in flight on cs-503 QOS. All 5 conditions will have N=2 multi-seed coverage by ~3 days from now.

**Decision (2026-04-25 / 26)**: 
1. F1-F4 σ-sweep + log-polar + normaliser run on Izar (paper claims don't depend on them; they only sharpen).
2. **All multi-seed retrains run on Izar** (avoid cross-cluster merging headaches).
3. H100 list is now strictly: H100-A + H100-B (the 2 experiments truly best for friend's H100). 

**Hand-off status**: Friend has docs (`experiments/foveation_transform_fix_retrain.md`, `experiments/encoder_capacity_scaling.md`). **Critical path: H100-A and H100-B are the most blocking** — without them §4.5 H3 fov-shifted and App E scaling sweep remain TODO at submission.

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
| `fig4_h2_probe_transfer.pdf` + `fig4_transplant_5x5.pdf` (uniform aspect) | Fig 4 (§4.3 H2) | ✅ matched col/row 6/8 cells, last 2 in flight |
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
| Multi-seed | uniform_seed2 / foveated_seed2 (running) + blind/matched/foveated_learned seed=2 (queued) | All single-seed claims gated on N=2 replication | All §4 numbers |
| Transplant tail | Last 2 matched-recipient cells (blind→matched, uniform→matched) | Final 2 cells of Coarse column in Fig 4b | Fig 4b |

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
- **Persistent-failure terminal locations** (Table 4 in §4.5, commit `fab08b3`): only uniform's persistent-memory failures cluster around the previous-episode goal location (margin +1.83m); blind/matched/foveated terminal positions are closer to the new goal but not at it (n=27/35/16 same-floor failures). This refines the "having-vs-using" 2×2 dissociation: uniform's memory anchors on visual landmarks; blind's memory interferes through position-mis-reporting rather than location-anchor.
- **LSTM gain** (mentioned in §4.4, commit `2a4b9fe`): LSTM top-layer GPS R² minus encoder feature-map GPS R² is +3.9 for matched, +0.0 for uniform, +0.7 for foveated. Mid-magnitude foveated gain is consistent with foveation supplying less navigation-useful visual structure than uniform.
- **Failure-episode asymmetry** (NOT YET in paper, low confidence at n=12, single seed): 98 episodes uniquely fail in bottleneck conditions (rich-encoder succeeds), but only 12 episodes uniquely fail in rich-encoder conditions (bottleneck succeeds). Of the 12, scene 91 contributes 3, suggesting at least one "rich-encoder-unfriendly" scene. 6 of the 12 are short-geodesic ($<7$m), so failure isn't path-length-driven. Worth re-checking post multi-seed; if pattern holds, indicates rich-encoder agents have a small but non-zero failure mode that bottleneck doesn't share — visual landmark misreading on specific scene types.

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
git log --oneline -10 -- docs/manuscript/main.tex

# Figure freshness (date-sorted)
ls -la docs/manuscript/fig/*.pdf | sort -k 6,8

# Pending det probe analyses
ssh izar "ls /scratch/izar/wxu/probing_results/*_det_analysis.json | wc -l"
```
