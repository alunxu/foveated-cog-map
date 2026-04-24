# Experiments

Self-contained experiment briefs. Each file describes one intention,
the plan, and the expected outcome. Anyone with SLURM + our repo
synced can pick one up and run it.

## Open

| File | Intention |
|------|-----------|
| `encoder_capacity_scaling.md` | Turn the 2-point bottleneck observation (blind, matched-48) into a 7-point scaling curve by training intermediate-resolution variants (matched-{32, 64, 96, 192}). |
| `multiseed_robustness.md` | Add 2 seeds per condition for cross-seed error bars on the H1 ordering. |

## Adding a new experiment

One file per experiment, named by intention (`<what_it_tests>.md`).
Include: motivation, plan (configs + submit commands), expected
outcome, and how the output integrates with the paper.
