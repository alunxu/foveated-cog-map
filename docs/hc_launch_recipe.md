# HC (High-Compute) Launch Recipe — Stochastic Gaze + Scaling Sweep

This document is for the friend / collaborator running net-new
experiments on high-compute (hc) hardware (e.g. H200 cluster).
Created 2026-04-26.

## Two experiments to run

| # | Experiment | Configs | Net training jobs | ETA on 1× hc GPU |
|---|---|---|---|---|
| **1** | Stochastic gaze | `foveated_stochastic_gibson` | 1 | ~3 days |
| **2** | Encoder-capacity scaling sweep | `matched{32,64,96,192}_gibson` | 4 | ~3 days each (parallel-able) |

Both are NET-NEW value (Izar already covers σ_max sweep, log-polar,
foveated_shifted, fov_v2, multi-seed seed=2). Skip those — see
"What NOT to redo" at the bottom.

---

# Experiment 1: Stochastic gaze

## What this experiment tests

The existing `FoveatedLearnedGazePolicy` (deterministic sigmoid MLP) collapses
to a fixed point at `(0.49, 0.62)` under pure task supervision. The paper §4.6
(H3, gaze location) currently rests on:

- `foveated (fix)`: static center gaze
- `foveated (learned)`: collapsed deterministic gaze (failed module)
- `foveated_shifted` (in training on Izar): static at the collapsed position

The remaining gap: **does a working learned-gaze policy reproduce the
collapsed-gaze behaviour, or does it discover a different gaze-content
coupling?** Stochastic gaze is the smallest design change that should
prevent collapse.

## Design summary

`FoveatedStochasticGazePolicy` (in `src/habitat/foveated_stochastic_policy.py`):

```
gaze decoder output: 4 values  (μ_x_logit, μ_y_logit, σ_x_raw, σ_y_raw)
  μ = sigmoid(μ_logit)                          ∈ [0, 1]
  σ = 0.05 + 0.25 · sigmoid(σ_raw)              ∈ [0.05, 0.30]
train: gaze = (μ + σ · ε).clamp(0.05, 0.95)     ε ~ N(0, 1)  (reparameterized)
eval:  gaze = μ                                 (deterministic)
```

Bounded σ replaces the previous `gaze_diversity_loss` aux loss. The
σ_min=0.05 floor enforces ~5% image-range exploration **per env** at
all times (cannot collapse). The σ_max=0.30 cap prevents pure-random
gaze that would destroy useful foveation.

Reparameterization ensures gradient flows from PPO loss → gaze sample
→ μ and σ. No auxiliary loss needed.

Same slow-gaze approximation as the deterministic learned-gaze policy:
one sample per env per rollout segment (broadcast across timesteps).

## Files added

| File | Purpose |
|---|---|
| `src/habitat/foveated_stochastic_policy.py` | Policy + Net classes |
| `src/habitat/__init__.py` | Registers `FoveatedStochasticGazePolicy` |
| `habitat_configs/ddppo_pointnav_foveated_stochastic_gibson.yaml` | Hydra training config (250M frames, identical to `foveated_learned_gibson` except policy class) |

## Smoke test (Izar GPU; should be portable to hc cluster)

```bash
# Run the smoke test:
python /tmp/smoke_stochastic_gaze.py

# Expected output:
#   [TRAIN] gaze_mu shape: (4, 2), range [..., ...]
#   [TRAIN] gaze_sigma shape: (4, 2), range [0.05, 0.30]
#   [TRAIN] ✓ All bounds checks passed
#   [TRAIN] ✓ Stochastic sampling verified
#   [EVAL]  ✓ Deterministic + gaze == μ at eval
#   [GRAD]  ✓ Reparameterization works (gradient flows to decoder)
#   ALL SMOKE TESTS PASSED
```

If smoke test passes, the policy is ready for training.

## Launch on hc cluster

The training config matches the existing foveation experiments: 250M
environment frames, ResNet-18, 3-layer LSTM-512.

### Single-node sbatch

```bash
# Standard training (no resume)
sbatch scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_foveated_stochastic_gibson

# Multi-seed for replication (after primary run lands)
sbatch scripts/cluster/submit_train_seeded.sh pointnav/ddppo_pointnav_foveated_stochastic_gibson 2
sbatch scripts/cluster/submit_train_seeded.sh pointnav/ddppo_pointnav_foveated_stochastic_gibson 3
```

Expected wall time on hc hardware (~3-4× faster than V100, e.g. H200):
- 250M frames target × ~1k frames/sec × 1 hc GPU = ~70h ≈ 3 days
- With multi-GPU hc allocation: ~24h

Adapt SBATCH directives to friend's cluster QoS / partition / gpu type
as needed.

### Hyperparameter notes

The config uses the same DD-PPO settings as the existing 5 conditions:
- `total_num_steps: 2.5e8` (250M)
- `lr: 2.5e-4`, `entropy_coef: 0.01`
- `num_steps: 128` (rollout segment length)
- `num_environments: 6`
- `num_recurrent_layers: 3`, `hidden_size: 512`
- `backbone: resnet18`

Stochastic-gaze hyperparameters (in `FoveatedStochasticGazePolicy.__init__`):
- `sigma_min=0.05`, `sigma_max=0.30`
- `gaze_clip_lo=0.05`, `gaze_clip_hi=0.95`
- `gaze_hidden=64`

If gaze still doesn't show interesting structure after 50M frames
(see "What success looks like"), reduce `sigma_min` to `0.02` so the
policy can use more focused gaze when confident.

## What success looks like

After training, run the standard probing pipeline:

```bash
# Deterministic rollouts (eval mode → gaze = μ, deterministic):
sbatch scripts/cluster/submit_probe_deterministic.sh \
    pointnav/ddppo_pointnav_foveated_stochastic_gibson \
    /scratch/.../habitat_checkpoints/foveated_stochastic_gibson/latest.pth

# Then standard analysis:
python scripts/probing/analyze.py \
    --data .../foveated_stochastic_gibson_det.npz \
    --out .../foveated_stochastic_gibson_det_analysis.json
```

### Success criteria (in priority order)

1. **Gaze does NOT collapse**: per-env gaze trajectory should show
   meaningful variance (std > 0.05 over time within an episode). The
   per-env `gaze_mu` distribution across the full deterministic
   rollout should NOT be a delta function.
2. **Comparable task performance**: SPL ≥ 0.75 (vs `foveated (learned)`
   ≈ 0.82). Some loss is acceptable (exploration cost), but >10%
   regression is a problem.
3. **Different H1/H2 signatures from `foveated (fix)`**: top-layer
   GPS R², compass R², MP3D held-out shift, and shortcut SPL drop
   should differ from the static-center foveated condition.
4. **NEW H3 result**: gaze `mu` shows scene-dependent structure
   (not the same `(0.49, 0.62)` regardless of input). If yes,
   strong evidence for active gaze content axis.

### Failure signatures (worth investigating before paper)

- **σ collapses to its minimum (0.05)**: PPO is pushing σ down. Could
  mean the policy prefers determinism and the noise hurts. Try
  `sigma_min=0.02`.
- **σ stays at maximum (0.30)**: PPO can't find informative gaze, so
  noise is "free". Try lower entropy coefficient (`entropy_coef=0.005`)
  or longer training.
- **Task SPL < 0.5**: stochastic gaze is hurting navigation too much.
  Lower `sigma_max` to 0.15.
- **Gaze μ identical across envs**: the decoder is collapsed despite
  σ floor. Check that `gaze_decoder` parameters are receiving gradient
  (rerun smoke test).

## Integration into paper (if results land before deadline)

Adds a 6th condition to the comparison set (5 visual conditions + 1
stochastic-gaze variant). Updates §4.6 H3 from "in progress" to a real
test.

If results agree with `foveated_learned` (deterministic): suggests
H3 effect is robust to gaze-module design. Strong claim.

If results disagree: investigate WHY (paper's narrative would need
revision). Check if the task simply doesn't reward varied gaze.

---

# Experiment 2: Encoder-capacity scaling sweep

## What this experiment tests

The paper's H1 (encoder–memory race) currently rests on **5 conditions
at fixed input resolutions** (blind: 0px, coarse: 48px, uniform: 256px,
foveated: 256px-with-blur). The scaling sweep adds the missing
intermediate points:

```
0px      48px        96px     128px       192px         256px
↑         ↑           ↑         ↑           ↑              ↑
blind   coarse    matched96  matched128  matched192    uniform
        (=matched)
```

Already trained on Izar: `blind`, `matched` (48px), `matched128`,
`uniform`. Missing: **matched32, matched64, matched96, matched192**.

The hypothesis: GPS probe R² decreases monotonically with input
resolution, from ~0.95 at no-encoder to ~0 at 256×256, crossing into
the rich-encoder regime around 64–128 px. Currently in paper
`Appendix~\ref{app:scaling}` — text is `\TODO`.

This is paper-section-completing: §1, §3.1, §4.2, §5.5, §6 all
reference `app:scaling` as "in progress / pending". Without it, the
paper claims correlation; with it, the paper claims a continuous
causal lever.

## Files needed (already in repo at commit 4585f45)

| File | Status |
|---|---|
| `habitat_configs/ddppo_pointnav_matched32_gibson.yaml` | ✓ |
| `habitat_configs/ddppo_pointnav_matched64_gibson.yaml` | ✓ |
| `habitat_configs/ddppo_pointnav_matched96_gibson.yaml` | ✓ |
| `habitat_configs/ddppo_pointnav_matched192_gibson.yaml` | ✓ |

Each uses the standard `WijmansPointNavPolicy` — no foveation. Only
difference vs `matched_gibson.yaml` is the rgb/depth `height:` and
`width:` (32/64/96/192) and the checkpoint folder name.

## Smoke test

These reuse the existing `WijmansPointNavPolicy` which has been
extensively tested. No new code; just new resolution. The matched128
checkpoint already exists on Izar — successful precedent for
intermediate-resolution variants.

## Launch on hc cluster

```bash
# Sequential (each job uses 1 hc GPU):
for res in 32 64 96 192; do
    sbatch scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_matched${res}_gibson
done

# Parallel (each job uses 1 hc GPU, all 4 run at once if you have ≥4 cards):
# Same as above; SLURM will schedule them concurrently if resources allow.
```

Expected wall time per run: ~3 days at 250M frames on 1 hc GPU.

If walltime per job is limited (e.g. <72h on your cluster), the
training will resume from `latest.pth` automatically when re-launched
to the same `checkpoint_folder` (habitat-baselines built-in).

## What success looks like

After all 4 runs complete (or resume to convergence), run probing:

```bash
for res in 32 64 96 192; do
    sbatch scripts/cluster/submit_probe_deterministic.sh \
        pointnav/ddppo_pointnav_matched${res}_gibson \
        /scratch/.../habitat_checkpoints/matched${res}_gibson/latest.pth
done
```

Then make the figure:

```bash
# Use existing make_h1_mega_figure.py with extended condition list,
# OR write a new make_scaling_figure.py that plots GPS R² vs input resolution.
# (Friend can pass JSON paths back; figure script will run on user's side.)
```

### Expected curve (paper hypothesis)

```
GPS R² (5-fold CV)
1.0 ┤ ●────●───●         ← bottleneck plateau (blind, 32px, 48px)
    │           \●
0.5 ┤            \●      ← transition (64-128 px)
    │             \●
0.0 ┤              \●─●  ← rich-encoder plateau (192px, 256px)
    └──────────────────────
   0  32  48 64 96 128 192 256  input resolution (px)
```

If observed: **direct causal H1 evidence**. The paper's H1 claim
upgrades from "correlation across 5 hand-picked conditions" to
"continuous causal lever via encoder spatial output dimensionality".

If the curve is flat or non-monotonic: **investigate first** (probably
training-budget mismatch — H1 mechanism predicts dependence on
encoder spatial output dim, not pixel count per se). Don't write up
as a contradiction without re-checking that all conditions converged.

## Integration into paper

Closes `app:scaling` `\TODO`. Affected sections:

- **§1** (line 98): "scaling-sweep experiment that would test the H1
  mechanism causally is in progress" → results landed
- **§3.1** (line 119): "encoder-resolution scaling sweep
  (Appendix~\ref{app:scaling}) addresses this directly"
- **§4.2** (line 169): "Whether the correlation reflects causation is
  what the §\ref{sec:foveation} experiments and the
  Appendix~\ref{app:scaling} resolution sweep test directly"
- **§5.5** (line 305): scaling sweep is one of the limitations
- **§6** (line 310): pending experiments list

If results agree with hypothesis: causal H1 fully settled.

If results disagree: paper needs to soften "encoder spatial output
dimensionality" framing back toward correlation-only.

---

# What NOT to redo on hc cluster

Izar already covers these. Do NOT duplicate:

| Experiment | Izar status | Why skip |
|---|---|---|
| σ_max=2/4/12/20 strength sweep | 4 jobs (σ=20 RUNNING, others PENDING) | full coverage |
| F3 log-polar | RUNNING ~18h | covered |
| foveated_shifted (H3 control) | PENDING normal QOS | covered |
| fov_v2 (clean re-run) | RUNNING ~21h | covered |
| Multi-seed seed=2 | 5 conditions RUNNING / queued resume | covered |

If hc cluster has spare capacity AFTER experiments 1+2, the next-most-valuable
addition would be **multi-seed seed=3** for the original 5 conditions
(blind, matched, uniform, foveated, foveated_learned). This builds toward
the canonical 3-seed setup that gates the strength of every quantitative
claim. But this is lower priority than experiments 1 and 2.

---

## Contact / sync

Paper repo: see latest §4.6 / Appendix `app:foveation-status` for the
H3 narrative. Substitution mechanism evidence (commit b9487aa) is
the most recent positive finding to be aware of for context.
