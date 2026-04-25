# Experiments

Self-contained experiment briefs. Each file describes one intention,
the plan, and the expected outcome. Anyone with SLURM + our repo
synced can pick one up and run it.

## Compute split

DD-PPO training is the bottleneck (~250M frames, 10–15h on H100 / 2–3
days on V100). Probing rollouts and analysis are cheap (~0.5–4h).
Allocate accordingly:

- **Training** (long): run on H100/H200 (friend's cluster).
- **Probing + analysis** (short): run wherever is convenient; the
  checkpoints sync as `.pth` files, which are small.

This keeps the long, parallelisable work on the faster hardware and
reserves Izar for short, sequential, interactive work.

## First-time cluster setup

If running these experiments on a cluster for the first time, follow
`setup.md` end to end: conda environment → Habitat stack →
Gibson dataset → repo install → sanity-check dry-run. Total time
~30–60 min.

## Open

| File | Intention |
|------|-----------|
| `foveation_transform_fix_retrain.md` | **High priority.** Re-train the 4 shifted-gaze conditions under the fixed `torch_foveation` transform. Blocks H3 (fov-shifted) and updates the paper's fov-learned row. |
| `foveation_strength_ablation.md` | **High priority.** Train fov-v2 (clean restart, full 250M; F1), fov-strong (σ_max=20; F4), and fov-logpolar (real spatial-sample reduction; F3) to check whether our paper's "fov-fix ≈ uniform pass-through" finding is robust to (i) under-training, (ii) blur strength, (iii) foveation model class. Critical for defending the encoder-memory race claim. |
| `foveation_normaliser_invariance.md` | Train fov-fix with `RunningMeanAndVar` enabled (F2), to test whether the normaliser-disabled implementation is a confound in the fov-fix vs uniform comparison. |
| `encoder_capacity_scaling.md` | Turn the 2-point bottleneck observation (blind, matched-48) into a 7-point scaling curve by training intermediate-resolution variants (matched-{32, 64, 96, 192}). |
| `multiseed_robustness.md` | Add seeds per condition for cross-seed error bars on the H1 ordering. Izar is already running some seeds; this doc covers only the remaining gap. |

## Observations (not actively running)

| File | Observation |
|------|-------------|
| `training_dynamics_observation.md` | Under-trained fov-fix (~115M) shows mid-regime LSTM GPS encoding (R²=0.46) while fully-trained fov-fix (~174M) is at chance. Candidate training-dynamics support for the bottleneck hypothesis; needs corroboration (multi-seed or intra-run ckpt sweep) before paper-ready. |

## Adding a new experiment

One file per experiment, named by intention (`<what_it_tests>.md`).
Include: motivation, plan (configs + submit commands), expected
outcome, and how the output integrates with the paper.
