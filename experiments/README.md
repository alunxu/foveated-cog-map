# Experiments

Self-contained experiment briefs.

## Where the active plan lives

The 14-training plan that the friend's hc cluster runs is consolidated
into a single comprehensive doc:

→ **[docs/hc_experiment_plan.md](../docs/hc_experiment_plan.md)** —
   per-training motivation, prediction, and ship-back protocol.

The earlier per-experiment briefs (`multiseed_robustness.md`,
`encoder_capacity_scaling.md`, `foveation_*.md`) that lived here have
been **deleted** as duplicates of `docs/hc_experiment_plan.md`.  Git
history retains them if you need to recover an old version.

## Compute split

DD-PPO training is the bottleneck (~250M frames, 10–15h on H100 / 2–3
days on V100). Probing rollouts and analysis are cheap (~0.5–4h).
Allocate accordingly:

- **Training** (long): run on H100/H200 (friend's cluster).
- **Probing + analysis** (short): run wherever is convenient; the
  checkpoints sync as `.pth` files, which are small.

## First-time cluster setup

If running these experiments on a cluster for the first time, follow
[`setup.md`](setup.md) end to end: conda environment → Habitat stack →
Gibson dataset → repo install → sanity-check dry-run.  Total time
~30–60 min.

For dataset-only details (Gibson + MP3D + merged layout) see
[`docs/DATASET_SETUP.md`](../docs/DATASET_SETUP.md).

## What's still here

| File | Purpose |
|------|---------|
| [`setup.md`](setup.md) | One-time cluster setup (conda env + Habitat install + blind-policy patch + sanity dry-run). |
| [`training_dynamics_observation.md`](training_dynamics_observation.md) | Historical observation that informed the substitution-mechanism narrative; raw artefact paths preserved. The observation itself is now integrated into paper §4.4 (substitution dynamics figure). |

## Adding a new experiment

If a new experiment is large enough to need its own brief (motivation +
plan + integration), prefer adding a section to
[`docs/hc_experiment_plan.md`](../docs/hc_experiment_plan.md) so the
friend has a single source of truth.  Standalone files in this directory
are reserved for setup helpers and historical observation notes.
