# Experiment: foveation strength ablation (F1 / F4 / F3)

## Intention

Our paper's H1 finding is that fov-fix behaves like uniform (both
"rich-encoder pass-through": top-layer GPS R² ≈ 0). That implies
foveation as we implement it doesn't create an encoder bottleneck.
**But our foveation model is just an eccentricity-dependent Gaussian
blur on a uniform pixel grid — it degrades peripheral spatial
frequency, but the spatial sample structure is still 256×256, and the
ResNet-18 still has plenty of cells to extract position from.**

This experiment tests whether the "fov-fix ≈ uniform" finding is
robust to:

- **F1 (fov-v2)**: more training time + clean NaN-patch from start, to
  rule out under-training (current fov-fix uses ckpt.36 / ~174M frames,
  forced by a NaN-corruption window in the original run).
- **F4 (fov-strong)**: stronger Gaussian blur (σ_max = 20 vs the
  default 8), to test whether even within the Gaussian-blur model
  class a stronger periphery suppression moves us toward the
  bottleneck regime.
- **F3 (fov-logpolar)**: a different foveation model entirely —
  log-polar resampling — that actually *removes* peripheral spatial
  samples (variable spatial sampling, the way real primate foveation
  works), instead of just blurring them.

## What's being trained where

All three retrainings go to friend's H100 cluster.  Single seed each
for v1 (multi-seed if compute allows; multiseed_robustness.md covers
the gap-fill plan separately).

| Run | Config | Cluster | Notes |
|-----|--------|---------|-------|
| `foveated_v2_gibson` | `pointnav/ddppo_pointnav_foveated_v2_gibson` | **friend** | Clean restart from scratch; NaN gradient patch active throughout (already in `wijmans_policy.py`); same hyperparameters as fov-fix |
| `foveated_strong_gibson` | `pointnav/ddppo_pointnav_foveated_strong_gibson` | **friend** | Identical to fov-fix except σ_max=20 (vs 8) |
| `foveated_logpolar_gibson` | `pointnav/ddppo_pointnav_foveated_logpolar_gibson` | **friend** | Log-polar foveation (`LogPolarFoveationTransform`); n_rho=64, n_theta=64 by default |

Each run is ~12-18h on H100/H200 for 250M frames.  Submit in parallel
to the four-GPU pool if available.

## Prerequisite

The new policies and foveation transform are in:
- `src/habitat/torch_foveation.py` — `LogPolarFoveationTransform` class
- `src/habitat/foveated_strong_policy.py` — F4 policy
- `src/habitat/foveated_logpolar_policy.py` — F3 policy + encoder

All three policies are registered in `src/habitat/__init__.py`.

**Sync the latest code first**:

```bash
cd ~/cs503-project
git pull
# Verify all three new policies are registered:
grep -E "FoveatedStrongWijmansPolicy|FoveatedLogPolarWijmansPolicy|FoveatedNormalisedWijmansPolicy" src/habitat/__init__.py
# Expect 3 import lines
# Verify unit tests pass for the foveation transforms:
python3 tests/test_torch_foveation.py
# Expect "All tests passed."
```

## Submit

```bash
# F1: fov-v2 (clean restart, full 250M with NaN patch active)
sbatch --gres=gpu:1 --time=24:00:00 \
    scripts/cluster/submit_train.sh \
    pointnav/ddppo_pointnav_foveated_v2_gibson

# F4: fov-strong (sigma_max=20)
sbatch --gres=gpu:1 --time=24:00:00 \
    scripts/cluster/submit_train.sh \
    pointnav/ddppo_pointnav_foveated_strong_gibson

# F3: fov-logpolar (log-polar resampling, real spatial bottleneck)
sbatch --gres=gpu:1 --time=24:00:00 \
    scripts/cluster/submit_train.sh \
    pointnav/ddppo_pointnav_foveated_logpolar_gibson
```

## Probe

After each training completes (or at ckpt.49), run the deterministic probe:

```bash
# F1 fov-v2
sbatch scripts/cluster/submit_probe_deterministic.sh \
    pointnav/ddppo_pointnav_foveated_v2_gibson \
    data/checkpoints/foveated_v2_gibson/ckpt.49.pth 500

# F4 fov-strong
sbatch scripts/cluster/submit_probe_deterministic.sh \
    pointnav/ddppo_pointnav_foveated_strong_gibson \
    data/checkpoints/foveated_strong_gibson/ckpt.49.pth 500

# F3 fov-logpolar
sbatch scripts/cluster/submit_probe_deterministic.sh \
    pointnav/ddppo_pointnav_foveated_logpolar_gibson \
    data/checkpoints/foveated_logpolar_gibson/ckpt.49.pth 500
```

Output:
- `probing_data/<condition>_det.npz`
- `probing_results/<condition>_det_analysis.json`

## Expected outcome (decision tree)

For each of F1, F4, F3, the relevant question is whether top-layer
GPS R² stays at chance (≈ 0) like fov-fix, or moves toward
matched-compute's R² ≈ 0.78.

**F1 (fov-v2 250M, clean restart):**
- GPS R² ≈ 0 (matches current fov-fix at R²=0.06): current finding
  is robust to under-training; the bottleneck framing stands.
- GPS R² substantially > 0: current fov-fix at ckpt.36 was
  under-trained, and the "fov-fix ≈ uniform pass-through" finding is a
  training-budget artefact. Paper needs reframing.

**F4 (fov-strong σ_max=20):**
- GPS R² ≈ 0: stronger Gaussian blur within the same model class is
  not enough to push fov-fix into the bottleneck regime. Suggests
  Gaussian blur is fundamentally too weak a foveation simulation; F3
  log-polar test is needed to make the "foveation matters" claim.
- GPS R² closer to matched-compute: foveation strength matters as a
  continuous variable; the "fov-fix ≈ uniform" finding is a
  blur-strength artefact at σ_max=8.

**F3 (fov-logpolar):**
- GPS R² ≈ 0: even with real spatial-sample reduction (variable
  sampling, not just blur), foveation does not push fov-fix into the
  bottleneck regime. Strongly supports the encoder-memory race
  framing as orthogonal to foveation specifically — what matters is
  feature-map size at the encoder *output*, not input.
- GPS R² closer to matched-compute (≥ 0.5): foveation, when
  implemented strongly enough to actually reduce spatial samples,
  *does* shape recurrent spatial memory. The paper's "fov-fix ≈
  uniform" finding then needs to be reframed as specific to weak
  Gaussian-blur foveation, and the encoder-memory race claim becomes
  *more* general (it now includes a foveation-driven instantiation).

## Integration with paper

These three results jointly determine whether the current paper's
"fov-fix is in the rich-encoder pass-through regime" claim is robust
or a setup artefact.  Each result independently sharpens or
reframes:

- F1 = under-training control
- F4 = blur-strength continuum within Gaussian-blur model class
- F3 = different foveation model class

If all three end at GPS R² ≈ 0, the paper's current framing is
strongly defensible. If any of them lifts GPS R² substantially, we
will reframe the relevant claim — see decision tree above.

## Cluster cost summary

3 jobs × 12-18h on H100/H200 = ~36-54 GPU-hours. Parallelisable across
4 GPUs (alongside the in-progress fov-shifted retrain etc.).
