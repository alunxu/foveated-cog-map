# Cogneuro Frameworks for the 5-Condition LSTM PointNav Paper

**Date:** 2026-05-06 (deadline night)
**Compute budget:** ~6-8 h tonight, **no retraining**.
**Data on hand:** for each of 5 conditions {blind, coarse 1x1, foveated 4x4 sigma=8, uniform 4x4, foveated_logpolar}, 100 episodes x 200-500 steps with cached:
- `h_t` (512-d top-layer LSTM hidden), `c_t`, all-layer h/c
- per-step labels: `agent_pos (x,y,z)`, `agent_dir`, `goal_vec`, `step_in_episode`, `episode_id`, `scene_id`, `distance_to_goal`, `local_occupancy 5x5`

**Paper backbone (lens through which every method is judged):**
> *Capacity allocation principle.* Sensor structure shapes the cognitive map along three orthogonal axes:
> - **magnitude** (how much state is encoded)
> - **format** (what geometry / which subspaces it lives on)
> - **consumption** (how it is used by the policy)

Every shortlisted method below is rated on whether it strengthens this lens (S = sharpens an axis, O = orthogonal-but-additive evidence, R = redundant with existing tools, X = high risk).

Existing tools we already have (do not duplicate):
- linear & MLP probes (LOSO) for {pos, head, goal, dist_to_goal, occupancy}
- Procrustes shape distance, principal angles, PR, TwoNN ID, Stringer alpha
- 5x5 memory transplant
- DSA pilot (failed; HP-sensitive — see `risks.md`)
- Skaggs spatial info (rectified)
- per-step progression / extended-lag probes / unaligned CKA

---

## Ranked shortlist (12 candidates)

| Rk | Method | Axis it tests | Cost | Risk | File |
|----|--------|---------------|------|------|------|
| 1 | **Cross-condition generalisation performance (CCGP)** — Bernardi+2020 | **format** (S) | 2 h, 80 LOC | low | [`ccgp_bernardi.md`](ccgp_bernardi.md) |
| 2 | **Temporal generalisation matrix (TGM)** — King & Dehaene 2014 | **format / consumption** (S) | 2 h, 60 LOC | low | [`tgm_king_dehaene.md`](tgm_king_dehaene.md) |
| 3 | **Manifold capacity (MFTMA)** — Chung+2018 / Cohen+2020 | **magnitude / format** (S) | 3 h, 200 LOC (use schung039 repo) | medium | [`manifold_capacity_chung.md`](manifold_capacity_chung.md) |
| 4 | **Demixed PCA (dPCA)** — Kobak+2016 | **format** (S) | 2 h, 100 LOC (machenslab/dPCA pip) | low | [`dpca_kobak.md`](dpca_kobak.md) |
| 5 | **Tensor component analysis (TCA)** — Williams+2018 | **format** (O) | 2 h, 80 LOC (tensortools) | low-med | [`tca_williams.md`](tca_williams.md) |
| 6 | **Intrinsic timescale hierarchy** — Murray+2014 | **magnitude / format** (S) | 1 h, 40 LOC | low | [`timescales_murray.md`](timescales_murray.md) |
| 7 | **Replay / preplay sequenceness (TDLM)** — Liu+2019, Pfeiffer & Foster 2013 | **consumption** (O) | 3 h, 150 LOC | medium-high | [`sequenceness_liu_tdlm.md`](sequenceness_liu_tdlm.md) |
| 8 | **Representational drift across episodes** — Driscoll+2017, Rule+2020 | **format** (O) | 1.5 h, 60 LOC | low | [`drift_driscoll_rule.md`](drift_driscoll_rule.md) |
| 9 | **Tangling Q metric** — Russo+2018 | **consumption / dynamics** (O) | 1.5 h, 50 LOC | low | [`tangling_russo.md`](tangling_russo.md) |
| 10 | **Mixed-selectivity dimensionality (Rigotti 2013)** | **format** (R/S boundary) | 1 h, 40 LOC | low | [`mixed_selectivity_rigotti.md`](mixed_selectivity_rigotti.md) |
| 11 | **Time-cell / ramp / persistent-activity catalogue** — MacDonald+2011, Eichenbaum 2014 | **magnitude** (O) | 2 h, 80 LOC | low | [`time_cells_eichenbaum.md`](time_cells_eichenbaum.md) |
| 12 | **Output-potent / output-null subspaces** — Kaufman+2014, Saxena & Cunningham 2019 | **consumption** (S) | 3 h, 150 LOC | medium | [`null_potent_kaufman_saxena.md`](null_potent_kaufman_saxena.md) |

---

## Top-3 picks (do these first)

### Pick #1: Cross-Condition Generalisation Performance (CCGP) — Bernardi et al. 2020 (Cell)
**Why first:** This is the single best-fit method for our paper. CCGP measures whether a learned dimension is *abstract* (factorised, disentangled) by testing if a linear decoder trained on one slice of conditions generalises to held-out slices. It maps directly onto the **format** axis — coarse vs foveated vs blind should differ in *how disentangled* their {pos, head, goal} subspaces are, even when raw probe accuracy is similar (which is exactly the gap our reviewer #2 was complaining about).

**Concrete protocol** for our data:
1. Define 4 binary "dichotomies" of episodes: e.g. `(scene_id even/odd) x (goal_in_left_half / right_half)`.
2. For each variable V in {pos_x quantile-binned, heading-octant, dist-to-goal-binned}, train a linear decoder on 3 of the 4 quadrants and test on the 4th. Average across the 4 holdouts -> CCGP(V).
3. Compare CCGP(V) to within-quadrant CV decoding accuracy. The gap = "abstraction index".
4. Predict (pre-registered): foveated > uniform > coarse > blind for goal-vector CCGP, but the ordering should *invert* or flatten for low-level pose, because foveated buys generalisable goal-space at the cost of sensor-specific pose binding.

Implementation notes:
- ~80 LOC, sklearn `LogisticRegression` + `LinearDiscriminantAnalysis`.
- Add to `scripts/probing/extra/ccgp_abstraction.py` next to `leave_one_scene_out.py` (which has the LOSO splitter already).
- Runtime: ~10 min per condition on CPU.

What success/failure tells us:
- **Success** (foveated has highest CCGP for goal-relative variables): we have a *quantitative* claim about the format axis. Reviewer #2's "so what?" is answered — sensor structure shapes geometry, not just magnitude.
- **Null** (CCGP flat across conditions): magnitude axis is the only real axis. We have to weaken the format claim. **This is OK** — Bernardi gives us the language to say "abstraction index does not differ".

### Pick #2: Temporal Generalisation Matrix (TGM) — King & Dehaene 2014 (TICS)
**Why second:** TGM is the cleanest *format/dynamics* probe we don't yet have. We train a decoder of variable V at time t, then test it at time t' for all (t, t'). The shape of the resulting square matrix diagnoses three regimes:
- **Diagonal-only** = transient code (different basis at each step).
- **Square block** = stable maintained code (same basis throughout).
- **Off-diagonal extending forward** = code is launched and then *held* (working-memory style).

This is *exactly* what distinguishes blind (must hold goal in memory) from sighted-with-goal-in-FOV (re-derived each step). It maps onto **format** (stable vs transient) and onto **consumption** (when the agent reads the code).

**Concrete protocol:**
1. Truncate episodes to first T=100 steps. Take h_t for all 100 ep x 100 step.
2. For each pair (t_train, t_test) in [0..T-1]^2, train a ridge regression on h_{t_train} -> goal_vec (using 80 ep), test on h_{t_test} of held-out 20 ep. Score = R^2 or angular error.
3. Plot 100x100 heatmap per condition. Pre-registered prediction: blind shows broad off-diagonal extension (memory-held), sighted shows diagonal band (sensor-derived each step), coarse shows the *most* off-diagonal because it has neither route.
4. Bonus: 5x5 cross-condition TGM = "transplanted decoder generalises in time" — extends the memory transplant.

Implementation notes:
- ~60 LOC pure numpy + sklearn.
- Add to `scripts/probing/extra/temporal_generalisation.py`.
- Runtime: ~5 min/condition.

What success/failure tells us:
- **Success** (qualitatively different TGM shapes across conditions): a one-figure proof that *format* differs. Doesn't depend on numerical scale, so robust.
- **Null** (all conditions identical squares or diagonals): we lose a panel but still have a methodologically sound figure.

### Pick #3: Manifold Capacity (MFTMA) — Chung, Lee, Sompolinsky 2018 / Cohen+2020 (Nat. Comm.)
**Why third (with risk caveat):** The single most direct test of the **magnitude** axis on our backbone. MFTMA gives capacity alpha_M, manifold radius R_M, manifold dimension D_M, separately. Cohen+2020 demonstrated that alpha_M tracks classification difficulty in CNN layers; we'd compute it on h_t with manifolds defined by *episode-id* (each episode = one manifold of states the agent traversed).

**Concrete protocol:**
1. Use the official `schung039/neural_manifolds_replicaMFT` repo (well-documented, MIT licence).
2. Define manifolds: 100 manifolds per condition, one per episode. Sample P=50 hidden states per episode.
3. Compute alpha_M, R_M, D_M for each condition. Pre-registered prediction: alpha_M monotone in sensor information (foveated > uniform > coarse > blind), but R_M and D_M may *not* be monotone — that's the format/magnitude split made geometric.

**Risk note:** MFTMA has known sensitivity to (a) number of manifolds M, (b) points per manifold P, (c) the random-projection dimension. *Do an HP sweep first* (M in {50,100,200}, P in {25,50,100}). If results are HP-stable: keep it. If HP-unstable: drop it (DSA-style HP-shopping risk — see `risks.md`).

Implementation notes:
- ~200 LOC if we want full ablation; ~80 LOC for a single config.
- Add to `scripts/probing/extra/manifold_capacity.py`.
- Runtime: ~30 min/condition (GPU helps but CPU fine for 5 conds).

What success/failure tells us:
- **Success** (alpha_M ordering matches sensor entropy, R_M *inverts* for foveated): a clean magnitude-vs-format dissociation. Big win.
- **HP-unstable**: walk away and use the runtime budget on TGM extensions. **Pre-register the HP sweep**.

---

## What we deliberately exclude (and why)

- **TEM / Whittington 2020**: needs retraining a model. Out of budget.
- **Banino 2018 grid-cell rate-map analysis**: Schaeffer 2022 already shows this is post-hoc cherry-picked. Reviewers will hate it. Leave the existing place-cell v2 analysis as is.
- **Rao-Ballard predictive coding**: requires architectural changes (top-down prediction error units). N/A.
- **Information bottleneck (Tishby)**: we already cite IB in discussion; no clean numerical estimator on h_t that would survive review.
- **DSA round 2**: known HP-sensitive failure mode (see `risks.md`).
- **Place-cell-style firing fields beyond Skaggs**: already done in `place_cell_v2.py`.

---

## Recommended execution order tonight

```
T0+0:00  CCGP        (2 h)          -> Pick #1 figure, big win for format axis
T0+2:00  TGM         (1.5 h)        -> Pick #2 figure, format/consumption
T0+3:30  Tangling Q  (1 h)          -> cheap win, dynamics complement
T0+4:30  Timescales  (1 h)          -> hierarchy panel
T0+5:30  MFTMA HP    (1.5 h)        -> only commit if HP-stable
T0+7:00  buffer / writing
```

CCGP + TGM together would be a major upgrade to the paper without any retraining. Tangling and timescales are 1-hour bonus panels. MFTMA is the high-variance but high-reward bet — pre-register the HP sweep so a null is still publishable as a methodology note.

See `risks.md` before committing to any single method.
