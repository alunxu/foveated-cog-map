# Pre-registration: Architecture-agnostic encoder-bandwidth probe

**Filed before any results are inspected.** This file fixes the analysis plan and
the predicted result shape, to prevent the DSA-style narrative-shopping that
sank the previous experiment. If results disagree with what is written here, we
report the null and do not iterate on probe HPs.

## Hypothesis (single sentence)

The bandwidth → linear-readable position-code prediction from our 5-condition
Habitat PointNav results reproduces in a *different architecture, environment,
and policy regime*: a small LSTM trained on top of frozen DINOv2-Base patch
features, on Memory Maze 9x9 offline trajectories.

## What is being varied (single primary axis)

* **Input resolution / spatial bandwidth** of the image fed to a *single* frozen
  DINOv2-Base encoder. Memory Maze frames are 64×64 RGB.

## Five conditions, mapped to our paper's 5 sensors

| condition | implementation | analogue in paper |
|-----------|----------------|--------------------|
| `blind` | RGB replaced by zeros (encoder still runs; gives a constant token) | blind |
| `coarse` | bilinear downsample 64→14, upsample 14→56 | coarse 1×1 encoder |
| `foveated` | gaussian blur (σ=4 px) at native 56×56 | foveated 4×4 with σ=8 blur |
| `uniform` | native 56×56 (downsampled from 64) | uniform 4×4 unblurred |
| `foveated_logpolar` | log-polar warp at 56×56 (central magnification) | foveated-logpolar |

Predicted ordering by *encoder bandwidth* (information available about pixel
content): `blind ≤ coarse ≤ foveated < foveated_logpolar ≲ uniform`.

## Recurrent integrator (held fixed across conditions)

* `nn.LSTM(input=768+6, hidden=512, num_layers=2, batch_first=True)`
* 768 = DINOv2-Base CLS dim; 6 = one-hot last action (Memory-Maze action space).
* Trained for 30k steps, sequence length 100, batch 16, MSE next-feature
  prediction loss, Adam lr=3e-4.
* Trained *separately per condition* on the cached features for that condition.

## Probes (held fixed across conditions; matches Pasukonis 2023 §3.3)

* **Linear probe:** `nn.Linear(512 + 512 + 6, 2)` on `concat(h_t, c_t, last_action)
  → agent_pos`.
* **4-layer MLP probe:** `1024-1024-1024-1024` units, ELU activations,
  LayerNorm before each hidden layer, on the same input.
* **Train objective:** MSE on `agent_pos` (units = maze cells, range ~[0, 9]).
* **Eval window:** average per-step R² over steps 250-500 (i.e., second half of
  the trajectory). Trajectory length T=500 chosen to fit the 8h compute budget
  on Mac MPS; this differs from Pasukonis (T=1000, eval 500-1000) by halving
  the integration horizon. The "second half" framing — matching the burn-in
  removal of the published protocol — is what we preserve.
* **Train/test split:** trajectory-level split of locally generated Memory-Maze
  9x9 trajectories. Pasukonis Drive folder was rate-limited (24h cooldown), so
  we generate locally with `MUJOCO_GL=glfw` on Mac. ~600 train + 200 eval
  trajectories with the explorer policy in `01_generate_data.py`. Random seeds
  are disjoint between splits.
* **Probe optimisation:** Adam lr=1e-3, 30k gradient steps, batch 256 frames.

## Predicted result shape (frozen at registration time)

For *linear* probe R² on `agent_pos`:
* **Monotone-increasing in bandwidth.** R²(blind) ≤ R²(coarse) ≤ R²(foveated)
  < R²(foveated_logpolar) ≲ R²(uniform).
* The blind condition serves as the floor (R² near 0).

For *MLP* probe R² on `agent_pos`:
* **Substantially flatter** than the linear curve across non-blind conditions.
* Rationale: under information-bottleneck framing, the 4-layer MLP can recover
  position even when the format is non-linear in the LSTM hidden state.

For the **gap** `R²_MLP − R²_linear`:
* **Largest at low-bandwidth conditions** (coarse, foveated), smaller at
  high-bandwidth (uniform). Mirrors Habitat result.

## Decision rules (frozen)

1. **Strong outcome** → integrate into §5.4 as architecture-agnostic
   confirmation. Required: linear R² is monotone non-decreasing across
   `blind < coarse < uniform`, AND `R²_MLP − R²_linear` is positive at coarse
   and ≤ half its coarse value at uniform.
2. **Moderate outcome** → integrate as supporting evidence with explicit
   caveats. Required: linear R² is monotone, but the MLP-recovery condition
   fails (or the gap is roughly constant).
3. **Null** → report in §5.6 Limitations as "we tested the prediction outside
   Habitat and the bandwidth-ordering signal did not reproduce", and do NOT
   iterate on probe HPs / encoder choice / training-step count to recover it.

## Sanity checks (run BEFORE the main analysis)

a. **Static-encoder baseline.** Linear probe of agent_pos *directly* on the
   DINOv2 CLS feature (no LSTM). Should be near-zero for blind, increasing in
   bandwidth. Confirms the signal exists at all.
b. **LSTM convergence.** Plot training loss; only continue to probe if loss
   has stabilised within 2× of its final value at the eval point.
c. **Trajectory-level shuffling control.** Shuffle agent_pos labels across
   trajectories within the eval set; both linear and MLP R² should be ≤ 0
   (chance level). If above zero, the probe is leaking time-of-day cues.

## Out-of-scope (do not chase if it doesn't work)

* Encoder family (DINOv2 vs VC-1 vs R3M).
* LSTM architecture variations beyond the registered config.
* Memory-Maze 15x15 (only 9x9 in 8h).
* Re-training DINOv2 (frozen).

---

**Author:** experiment loop, 2026-05-06.
**Status:** filed before code runs. Modify only by appending dated revision notes.
