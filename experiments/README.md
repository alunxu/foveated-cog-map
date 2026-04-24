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

## Open

| File | Intention |
|------|-----------|
| `encoder_capacity_scaling.md` | Turn the 2-point bottleneck observation (blind, matched-48) into a 7-point scaling curve by training intermediate-resolution variants (matched-{32, 64, 96, 192}). |
| `multiseed_robustness.md` | Add seeds per condition for cross-seed error bars on the H1 ordering. Izar is already running some seeds; this doc covers only the remaining gap. |

## Adding a new experiment

One file per experiment, named by intention (`<what_it_tests>.md`).
Include: motivation, plan (configs + submit commands), expected
outcome, and how the output integrates with the paper.
