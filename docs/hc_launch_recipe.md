# HC (High-Compute) 9-Day Launch Recipe — NeurIPS 2026 Final Push

**Deadline**: 2026-05-06 (NeurIPS abstract / submission cutoff).
**Today**: 2026-04-27.
**Net working days**: ~9.

**Hardware**: Friend has **4× H100 + 1× H200** (= 5 GPUs in parallel).
**Strategy**: All foveation variants, scaling sweep, stochastic gaze, and 3
of 5 multi-seed runs go to hc. Izar finishes only `uni-s2`, `fov-s2`
(already >50% complete); Izar's role becomes **probing + analysis** of
hc-trained checkpoints once they land.

This document tells the collaborator **exactly which 14 trainings to
launch, in which order, on which GPU**, plus how to ship the trained
checkpoints back to Izar for downstream probing.

---

## Tier 1 — Must-have, launch DAY 1 (5 GPUs in parallel)

These are the trainings whose results materially change paper claims.
If only some hc trainings make the deadline, these MUST.

| Slot | Job | Config | What it tests in the paper |
|---|---|---|---|
| **GPU 1 (H200)** | Stochastic gaze | `pointnav/ddppo_pointnav_foveated_stochastic_gibson` | §4.6 H3 dynamic gaze test (currently only static control); upgrades H3 from "in flight" to a real test |
| **GPU 2** | Scaling sweep K=64 | `pointnav/ddppo_pointnav_matched64_gibson` | Causal H1: 64×64 input → ~2×2 encoder output, predicted ~ between coarse and uniform |
| **GPU 3** | Scaling sweep K=32 | `pointnav/ddppo_pointnav_matched32_gibson` | Causal H1: 32×32 input → 1×1 encoder (matches coarse-bandwidth lower bound) |
| **GPU 4** | Multi-seed bld-s2 | `pointnav/ddppo_pointnav_blind_gibson` (seed=2) | Replication of blind (R²=0.95 on Izar single seed) |
| **GPU 5** | Multi-seed mtc-s2 | `pointnav/ddppo_pointnav_matched_gibson` (seed=2) | Replication of coarse (R²=0.78 on Izar single seed) |

> ⚠️ **K=128 is already done on Izar** (50 ckpts converged earlier; probing
> running now). DO NOT relaunch K=128 on hc.

**Launch all 5 on Day 1**:

```bash
cd /path/to/cs503-project

# Stochastic gaze (GPU 1, H200 preferred)
sbatch <your-sbatch-flags> scripts/cluster/submit_train.sh \
    pointnav/ddppo_pointnav_foveated_stochastic_gibson

# Scaling sweep K=64, K=32 (GPUs 2, 3) — K=128 ALREADY DONE on Izar
sbatch <your-sbatch-flags> scripts/cluster/submit_train.sh \
    pointnav/ddppo_pointnav_matched64_gibson
sbatch <your-sbatch-flags> scripts/cluster/submit_train.sh \
    pointnav/ddppo_pointnav_matched32_gibson

# Multi-seed bld-s2, mtc-s2 (GPUs 4, 5)
sbatch <your-sbatch-flags> scripts/cluster/submit_train_seeded.sh \
    pointnav/ddppo_pointnav_blind_gibson 2
sbatch <your-sbatch-flags> scripts/cluster/submit_train_seeded.sh \
    pointnav/ddppo_pointnav_matched_gibson 2
```

ETA per run: 2-3 days on H100/H200 (vs ~5 days on V100).

---

## Tier 2 — Launch DAY 3-4 (when GPUs free up from Tier 1)

Filling out scaling sweep + last multi-seed condition + F3 falsifiable.

| Slot | Job | Config | Purpose |
|---|---|---|---|
| GPU 1 | Scaling sweep K=96 | `pointnav/ddppo_pointnav_matched96_gibson` | Mid-sweep |
| GPU 2 | Scaling sweep K=192 | `pointnav/ddppo_pointnav_matched192_gibson` | High-end (close to uniform) |
| GPU 3 | Multi-seed fov_lrn-s2 | `pointnav/ddppo_pointnav_foveated_learned_gibson` (seed=2) | Replication of foveated (learned), the H3 anchor |
| GPU 4 | Foveated log-polar (F3) | `pointnav/ddppo_pointnav_foveated_logpolar_gibson` | Falsifiable test for §4.4 H1 mechanism (predicts GPS R² ≥ 0.3) |
| GPU 5 | Foveated v2 (clean re-run) | `pointnav/ddppo_pointnav_foveated_v2_gibson` | §5.5(ii) clean replacement for ckpt.36 buggy NaN-corruption fov-fix |

```bash
sbatch <flags> scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_matched96_gibson
sbatch <flags> scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_matched192_gibson
sbatch <flags> scripts/cluster/submit_train_seeded.sh pointnav/ddppo_pointnav_foveated_learned_gibson 2
sbatch <flags> scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_foveated_logpolar_gibson
sbatch <flags> scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_foveated_v2_gibson
```

---

## Tier 3 — Launch DAY 6-7 (nice-to-have foveation completeness)

If Tier 1+2 finish ahead of schedule. Skip if running tight.

| Slot | Job | Config | Purpose |
|---|---|---|---|
| GPU 1 | Foveated σ=20 (F4 strong) | `pointnav/ddppo_pointnav_foveated_strong_gibson` | F4 strength endpoint (high blur) |
| GPU 2 | Foveated σ=2 | `pointnav/ddppo_pointnav_foveated_sigma2_gibson` | F1 strength low end |
| GPU 3 | Foveated σ=12 | `pointnav/ddppo_pointnav_foveated_sigma12_gibson` | F1c mid |
| GPU 4 | Foveated shifted | `pointnav/ddppo_pointnav_foveated_shifted_gibson` | H3 static control (gaze hardcoded at (0.49, 0.62)) |
| GPU 5 | (spare) | — | Re-run a partial Tier1/2 if any failed |

```bash
sbatch <flags> scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_foveated_strong_gibson
sbatch <flags> scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_foveated_sigma2_gibson
sbatch <flags> scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_foveated_sigma12_gibson
sbatch <flags> scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_foveated_shifted_gibson
```

---

## Skip unless extra time

| Job | Why skip |
|---|---|
| Foveated σ=4 | Close to σ=2 / σ=8, marginal value |
| fov_normalised | Already in paper appendix as separate F2 design; not critical |

---

## What runs on Izar (do NOT relaunch on hc)

| Job | Status on Izar |
|---|---|
| `uniform_gibson_seed2` | 56% (140M / 250M frames) — finishing in ~6 days on V100 |
| `foveated_gibson_seed2` | 52% (130M / 250M) — finishing in ~6-9 days |

These are too far along to restart on hc. Izar's `auto_resume.sh` keeps
them rolling through 72h walltime cycles. Friend doesn't need to do
anything for these.

---

## Daily schedule overview

```
Day 1     5 Tier 1 trainings start (parallel on 5 GPUs)
Day 3-4   Tier 1 finishes → Tier 2 starts
Day 6-7   Tier 2 finishes → Tier 3 starts (if time)
Day 8-9   Tier 3 finishes; ship checkpoints back to Izar
Day 9     Final integration (wxu) + paper submit
```

---

## How to ship trained checkpoints back to Izar

Once a training run hits 250M frames (or hits Day 8 partial), zip the
final `latest.pth` + tensorboard dir and rsync to Izar. wxu will run
the probing pipeline:

```bash
# On hc cluster, for each completed run:
RUN_NAME=foveated_stochastic_gibson      # example
CKPT_DIR=/path/to/hc/checkpoints/${RUN_NAME}

# Just send the latest checkpoint + the last few intermediate ones for
# across-checkpoint probing (we found those very informative — see
# fig/fig3_substitution_dynamics.pdf).
rsync -avz \
    "${CKPT_DIR}/latest.pth" \
    "${CKPT_DIR}/ckpt.10.pth" \
    "${CKPT_DIR}/ckpt.20.pth" \
    "${CKPT_DIR}/ckpt.30.pth" \
    "${CKPT_DIR}/ckpt.49.pth" \
    "${CKPT_DIR}/tb/" \
    izar.epfl.ch:/scratch/izar/wxu/habitat_checkpoints/${RUN_NAME}/
```

(Replace the `izar.epfl.ch:...` part with whatever ssh alias / path
your cluster uses to reach Izar. wxu can adapt.)

---

## Sanity checks before launching

1. **Pull latest repo state**:
    ```bash
    cd /path/to/cs503-project
    git pull origin main
    ```
    Latest commit at submission prep is `9684d3e` or later.

2. **Verify configs are present**:
    ```bash
    ls habitat_configs/ddppo_pointnav_{foveated_stochastic,matched32,matched64,matched96,matched128,matched192,foveated_logpolar,foveated_v2,foveated_strong,foveated_sigma2,foveated_sigma12,foveated_shifted}_gibson.yaml
    ```
    Should list 12 files.

3. **Verify policies are registered** (1-line Python check):
    ```bash
    python -c "
    import sys; sys.path.insert(0, '/path/to/cs503-project')
    import src.habitat
    from habitat_baselines.common.baseline_registry import baseline_registry
    for name in [
        'FoveatedStochasticGazePolicy',
        'FoveatedSigma2WijmansPolicy',
        'FoveatedSigma4WijmansPolicy',
        'FoveatedSigma12WijmansPolicy',
        'FoveatedStrongWijmansPolicy',
        'FoveatedLogPolarWijmansPolicy',
        'FoveatedShiftedGazePolicy',
    ]:
        cls = baseline_registry.get_policy(name)
        print(f'{name}: {cls is not None}')
    "
    ```
    All should print `True`.

4. **Smoke test** for the new stochastic gaze policy (under 1 min):
    ```bash
    python /path/to/cs503-project/scripts/cluster/smoke_policy.py \
        FoveatedStochasticGazePolicy
    ```
    (Or equivalent — see `scripts/cluster/smoke_policy.py`.)

---

## Failure signatures to watch for

If you see any of these, ping wxu before continuing:

| Symptom | Likely cause | Action |
|---|---|---|
| `nan_sanitised > 0` in TB metrics | Numerical instability (rare on H100/H200; we patched this) | Continue training; the fix absorbs it |
| Stochastic gaze: σ collapses to 0.05 | PPO is pushing exploration down | Continue, log it; may be expected behavior |
| Stochastic gaze: SPL < 0.5 by 50M frames | Too much gaze noise hurting navigation | Continue but flag — may need to reduce sigma_max |
| Scaling sweep: matched-K converges to coarse-1×1 R² | Encoder collapsing earlier than expected | Continue, this would be the substitution mechanism's prediction (good!) |

---

## Communication

- Friend: launch all of Tier 1 ASAP; monitor with `squeue` etc.
- wxu (here, on Izar): finish probing on uni-s2/fov-s2 once they land,
  prepare paper figures + tex placeholders for incoming hc results,
  process hc checkpoints as they arrive.
- Daily check-in (text/Discord) on what's converged + what's stuck.

Paper integration target: Day 8-9 final pass.

Goal: a clean NeurIPS submission with multi-seed + scaling sweep +
stochastic gaze H3 result, all on consistent hardware.

— wxu, 2026-04-27 02:30 CEST
