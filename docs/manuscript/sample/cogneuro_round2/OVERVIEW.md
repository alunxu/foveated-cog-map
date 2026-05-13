# Cogneuro Round 2 — 8-12 NEW post-hoc analyses for the sensor-structure paper

**Cache.** All methods read `/tmp/rcp_analysis_v3/{cond}_det_RCP.npz` (5 conditions × ~500 episodes × per-step `h_t`, `c_t`, `agent_pos`, `agent_dir`, `goal_vec`, `distance_to_goal`, `step_in_episode`, `episode_id`, `scene_id`). No retraining. No new rollouts (except where noted with explicit "needs new rollouts" flag — those are deferred).

**Backbone story.** Three axes are already in the paper:
- **Magnitude** — how much linear-readable position information lives in `h_t` (probes, PR).
- **Format** — which subspace carries it (Procrustes, LOSO, TwoNN, transplant).
- **Consumption** — which route the policy uses (transplant asymmetry).

Round-1 added a fourth implicit axis — **temporal organisation** (TGM, intrinsic timescales). Round-2 adds two new axes that I will argue are *necessary* to make the comparative-cogneuro framing work:
- **Computation / dynamics** — *how* the recurrent state moves through state space (fixed points, predictive coding, transfer entropy, Fisher information).
- **Topology / geometry of the map** — *what shape* the cognitive-map manifold takes (persistent homology, Riemannian curvature, splitter-cell mixed selectivity).

---

## Meta-narrative

> *"This paper takes a first step toward a comparative cognitive neuroscience of artificial recurrent agents: a systematic application of the systems-neuroscience analytic toolkit to a controlled, multi-condition population of RL navigation agents that differ only in their sensor. Where biological comparative neuroscience must work across species (sensor differences confounded with everything else), our silicon analogue holds architecture, task, training, and behaviour fixed and varies only the front-end sensor. The same battery — linear and non-linear readout, generalisation across contexts, intrinsic timescales, fixed-point geometry, predictive-coding error, persistent homology of the state manifold, splitter-cell mixed selectivity, transfer-entropy flow, Fisher information — that is used to map cortex and hippocampus across rats, bats, monkeys, and humans is here brought to bear on five matched populations of artificial agents. The comparative method, not any single test, is the contribution: a portable battery that lets future work assay any encoder–memory–policy system along the same axes and ask the same questions."*

**Abstract sentences (drop-in).** 1-2 sentences for the abstract:

> *Beyond the empirical capacity-allocation result, we treat this controlled five-condition silicon analogue as a testbed for porting the systems-neuroscience analytic toolkit — population geometry, intrinsic timescales, fixed-point dynamics, predictive coding, splitter-cell mixed selectivity, and persistent-homology of the state manifold — to a comparative analysis of artificial recurrent agents. The matched-condition design isolates sensor structure as the single variable, providing the missing controlled counterpart to cross-species hippocampal comparisons.*

---

## Ranked shortlist (10 NEW candidates)

Ranked by **expected strength × pre-registerability × implementation cost**. Tier S = priority for the 4-hour subset; Tier A = strong but slightly more implementation; Tier B = ambitious / supplementary.

| # | Method | Tier | Risk | Hours | Predicted axis | Pre-reg sharpness |
|---|---|---|---|---|---|---|
| 1 | **Sussillo–Barak slow-point search** (`slow_points.md`) | S | LOW–MED | 3 | Computation/dynamics | HIGH — # of slow points, dimensionality of slow manifold, Jacobian eigenvalue spectrum should monotonically vary with sensor bandwidth |
| 2 | **Predictive-coding residual analysis** (`predictive_coding.md`) | S | LOW | 2 | Computation | HIGH — prediction-error magnitude in `h_{t+1} − f̂(h_t, a_t)` should be larger for blind/coarse than for foveated/uniform |
| 3 | **Persistent homology of episode trajectory** (`persistent_homology.md`) | S | LOW–MED | 3 | Topology | HIGH — Betti-0 / Betti-1 of the rate map should match map topology only when readout is linearly decodable; ladder across conditions |
| 4 | **Splitter / journey-coding cells** (`splitter_cells.md`) | S | LOW | 2 | Format | HIGH — fraction of `h_t` units showing trajectory-conditional firing is **higher** in blind/coarse, where memory must integrate longer paths |
| 5 | **Transfer entropy: sensor → h_t → policy** (`transfer_entropy.md`) | S | MED | 3 | Consumption | HIGH — TE(visual_features → h_t) increases with encoder bandwidth; TE(GPS_history → h_t) decreases. Confirms two-route claim |
| 6 | **Fisher information / linear Fisher (Beck/Brunel)** (`fisher_info.md`) | A | LOW | 2 | Magnitude (geometry-aware) | HIGH — linear Fisher about position is monotone in encoder bandwidth, but with **anti-monotone trend** for `h_t`-only Fisher (capacity-allocation prediction) |
| 7 | **Forget/input/output gate statistics** (`gate_analysis.md`) | A | LOW | 1.5 | Computation | MED — mean forget-gate openness should be **lower** for blind (longer integration) and **higher** for sighted (more updating) |
| 8 | **Cueva neural geometry of time** (`time_geometry.md`) | A | MED | 3 | Format (temporal axis) | MED–HIGH — time-on-trajectory vs time-in-episode geometry separates blind from sighted: blind should have more compressed time manifold |
| 9 | **Communication subspace (Semedo 2019)** (`comm_subspace.md`) | B | MED | 3 | Consumption | HIGH — dimensionality of LSTM-layer-2 → policy-head subspace should be **smaller** in blind (compact policy-relevant code), larger in sighted |
| 10 | **Avalanche / criticality statistics** (`criticality.md`) | B | HIGH | 4 | Dynamics | LOW — too HP-sensitive; flagged as supplementary "did we find criticality?" check |

Two **alternates** kept on bench (not in main 10 because of HIGH risk or scope):
- **Topological data analysis with Mapper** — alternative to persistent homology, more visual but more HP-sensitive.
- **Causal-abstraction probing (Geiger 2021)** — interpretability rather than cogneuro framing; better for a follow-up.

---

## 4-hour execution order (the "do this tonight" subset)

Pick **3 methods** that together cover **all three new axes** (computation, topology, mixed-selectivity / format) with LOW–MED risk:

1. **Splitter cells** (2h) — easiest, clear pre-reg, builds directly on existing `agent_pos` + `episode_id` indexing already in the npz.
2. **Predictive-coding residual** (2h) — needs only a 1-step dynamics regressor `f̂: (h_t, a_t) → h_{t+1}`; same indexing.
3. **Sussillo–Barak slow points** (3h) — needs the frozen LSTM forward pass from a checkpoint; if the checkpoint is on RCP and the cache npz lacks the LSTM weights, slip this to fallback.

**Fallback if no LSTM checkpoint forward access:** swap (3) for **Fisher information** (2h) which only needs the cached `h_t` array and `agent_pos` labels.

That gives a **runtime-independent 6-hour core** plus a 1-hour buffer for figures.

---

## Mapping to the magnitude / format / consumption / dynamics / topology axes

| Method | Magnitude | Format | Consumption | Dynamics | Topology |
|---|:-:|:-:|:-:|:-:|:-:|
| Slow-point search | | x | | **xx** | x |
| Predictive coding | x | | | **xx** | |
| Persistent homology | | **x** | | | **xx** |
| Splitter cells | | **xx** | | | |
| Transfer entropy | | | **xx** | x | |
| Fisher info | **xx** | x | | | |
| Gate statistics | | | | **xx** | |
| Time geometry | | **xx** | | x | x |
| Communication subspace | | x | **xx** | | |
| Criticality | | | | **x** | |

`xx` = primary axis the method speaks to; `x` = secondary contribution.

---

## What "first move toward comparative cogneuro" buys the paper

If 4 of these 10 land cleanly **and agree** with magnitude/format/consumption, the paper's contribution graduates from:

> *"a controlled study of how sensor bandwidth shapes one specific readout (linear position decode)"*

to:

> *"a portable, multi-axis comparative-analysis battery for artificial recurrent agents, demonstrated on a sensor axis but applicable to any matched-condition population (architecture comparisons, training-curriculum comparisons, biological-vs-silicon comparisons)."*

That is the "foundational" framing the user is pursuing — and the paper has the rare asset (5 matched-condition populations with cached states and per-step labels) that makes the methodology test possible. **The contribution is the battery applied to a controlled axis, not any single tail probability.**

See `narrative.md` for the §1 + §6 paragraph drafts. See `risks.md` for the per-method risk classification.
