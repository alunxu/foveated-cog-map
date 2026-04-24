# Observation: training-dynamics signal for the bottleneck hypothesis

Status: single-seed, noisy. Candidate for paper appendix once
corroborated by multi-seed data (`multiseed_robustness.md`) or
extended by probing intermediate checkpoints.

## The observation

Two foveated-fixed checkpoints probed under identical deterministic
protocol give different LSTM spatial-encoding signatures:

| Checkpoint | Frames trained | GPS CV R² | Compass CV R² | DtG CV R² |
|------------|---------------:|-----------|---------------|-----------|
| Fresh-restart `foveated_gibson/latest.pth` (ckpt.23) | ~115M | **+0.46 ± 0.39** | **+0.36 ± 0.37** | +0.89 ± 0.02 |
| Pre-corruption `foveated_gibson_corrupt_job2836021/ckpt.36` | ~174M | +0.06 ± 0.88 | +0.07 ± 0.69 | +0.82 ± 0.09 |

The under-trained (115M) fov-fix agent shows **non-zero LSTM spatial
encoding** (GPS 0.46, compass 0.36). The fully-trained (174M) one is
at chance (consistent with the main paper Table, foveated = pass-
through regime).

## Why this matters

Consistent with the bottleneck hypothesis read dynamically:

- **Early training**: visual encoder weights are still random/partial;
  the encoder cannot resolve world-frame pose yet, so LSTM
  compensates (mid-regime GPS encoding).
- **Late training**: visual encoder has converged to a rich state
  that extracts pose from each frame; LSTM no longer needs to encode
  it and drifts to the pass-through regime.

Equivalently: the bottleneck mechanism is *not* a one-shot property
of the final architecture, but a running equilibrium that shifts as
the encoder learns. The final state reflects which side wins the
race — and in foveated/uniform, the full-RGB encoder wins, so LSTM
ends up pass-through.

Blind and matched-compute never have that race: blind has no encoder;
matched's $1{\times}1$ feature collapse caps encoder capacity
throughout. So their LSTMs stay in compensatory mode from the start.

## Why it's not paper-ready yet

- **Single seed**: CV $\sigma = 0.39$ on the 115M-step data is large,
  and the whole observation is a single training run. Needs 2–3
  additional seeds.
- **Only 2 time-points**: fresh-restart at ~115M and pre-corruption
  at ~174M are not directly comparable because they are different
  training runs with different init seeds and RNG states. A within-
  run trajectory (probe every 20M frames of the same run) would be
  the clean test.
- **Noisy mechanism claim**: the interpretation "race between
  encoder and LSTM" is narrative; the experiment that would support
  it is: train with frozen visual encoder at various quality levels,
  measure LSTM encoding at each. We don't have that data.

## Minimal experiment to promote this to paper evidence

Probe every 10th intermediate checkpoint of the fully-trained
foveated-fixed run (either the corrupt pre-corruption series or the
fresh-restart, whichever is handier) and plot GPS/compass/DtG $R^2$
vs training frames. Expected: GPS+compass $R^2$ starts high, decays
to chance by ~200M frames; DtG stays flat.

Implementation: loop over `ckpt.{5, 10, 15, 20, ..., 49}.pth` and
run `submit_probe_deterministic.sh` on each (probe time ~2h each on
H100). 10 checkpoints × 2h = 20h of compute, easily parallelisable.

Would also want to do this on matched-compute (should show *slower*
decay since its encoder can't fully compensate) and uniform (should
show fast decay, similar to foveated).

## Paper placement

If confirmed, goes into an appendix section like
"Training-dynamics support for the bottleneck hypothesis", alongside
the main Table 1. The main text result (bottleneck ordering at
converged weights) stands on its own; this adds a mechanistic
"it's a race, not a one-shot" reading.

Not currently blocking the paper. Shelved until multi-seed data or
the intra-run ckpt sweep is available.

## Raw artefacts

- `/scratch/izar/wxu/probing_results/foveated_gibson_analysis.json`
  (fresh-restart, 128K steps, 24 Apr 18:34)
- `/scratch/izar/wxu/probing_results/foveated_gibson_det_analysis.json`
  (pre-corruption ckpt.36, 108K steps, 24 Apr 03:00)
- `/scratch/izar/wxu/probing_data/foveated_gibson.npz` (fresh-restart)
- `/scratch/izar/wxu/probing_data/foveated_gibson_det.npz` (pre-corruption)
