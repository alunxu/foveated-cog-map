# Experiments for friend's A100/H100/H200 GPUs

**Context**: We just confirmed a new core finding via deep-dive analysis:

> **Visual encoder capacity inversely determines LSTM spatial encoding.**

5-condition results (5-fold CV, deterministic probe on Gibson held-out):

| Condition | Input res | ResNet18 feat map | GPS R² | Compass R² |
|-----------|-----------|-------------------|--------|-----------|
| Blind | — | — | **0.95 ± 0.02** | 0.81 ± 0.08 |
| Matched-48 | 48×48 | 1×1 × 256 | **0.78 ± 0.10** | 0.64 ± 0.10 |
| Uniform | 256×256 | 8×8 × 256 | -0.31 ± 0.86 | 0.36 ± 0.23 |
| Foveated | 256×256 blurred | 8×8 × 256 | +0.06 ± 0.88 | 0.07 ± 0.69 |
| Fov-learned | 256×256 | 8×8 × 256 | -2.43 ± 3.98 | -1.34 ± 3.14 |

Clean monotonic pattern. But we have only **2 points in the bottleneck
regime** (blind, matched-48) and **3 saturated points** at full capacity.
To make the finding a scaling law, we need intermediate points.

## Priority 1 — Input-resolution sweep (the scaling-law experiment)

**Goal**: Fill the curve between blind (0 pixels) and uniform (256×256).

**Training runs** (4 new, each ~250M frames):

| Config | Input res | ResNet18 out | Pixels approx |
|--------|-----------|--------------|---------------|
| `matched32_gibson` | 32×32 | 1×1 | 1024 |
| (Matched-48, already trained) | 48×48 | 1×1 | 2304 |
| `matched64_gibson` | 64×64 | 2×2 | 4096 |
| `matched96_gibson` | 96×96 | 3×3 | 9216 |
| (Matched-128, training on Izar) | 128×128 | 4×4 | 16384 |
| `matched192_gibson` | 192×192 | 6×6 | 36864 |
| (Uniform-256, already trained) | 256×256 | 8×8 | 65536 |

**Configs already committed** in `habitat_configs/`:
- `ddppo_pointnav_matched32_gibson.yaml`
- `ddppo_pointnav_matched64_gibson.yaml`
- `ddppo_pointnav_matched96_gibson.yaml`
- `ddppo_pointnav_matched192_gibson.yaml`

**Submit**:

```bash
# On friend's cluster (with our repo synced):
for res in 32 64 96 192; do
    sbatch --gres=gpu:1 --time=24:00:00 \
        scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_matched${res}_gibson
done
```

**Expected time on H100/H200**: Each ~10-15h for 250M frames. 4 trainings
in parallel → all done in ~15h.

**Each training output**: `checkpoints/matched${res}_gibson/ckpt.{0..49}.pth`

**Then probe them**:
```bash
for res in 32 64 96 192; do
    sbatch scripts/cluster/submit_probe_deterministic.sh \
        pointnav/ddppo_pointnav_matched${res}_gibson \
        checkpoints/matched${res}_gibson/ckpt.49.pth 500
done
```

**Probe time**: ~2-4h each on H100. 4 parallel → 4h total.

**Expected result**: GPS R² monotonically decreases with resolution:
`matched32 > matched64 > matched96 > matched128 > matched192 > uniform`.
Would turn 2 data points into a clean scaling law — strong paper figure.

## Priority 2 — Multi-seed robustness (error bars on main claim)

**Goal**: 2 additional seeds per condition for error bars on blind/matched
/uniform/foveated/fov-learned.

**10 runs**, each ~250M frames (~15h on H100):

```bash
for seed in 1 2; do
    for cfg in blind uniform foveated foveated_learned matched; do
        sbatch --gres=gpu:1 --time=24:00:00 \
            scripts/cluster/submit_train_seeded.sh \
            pointnav/ddppo_pointnav_${cfg}_gibson ${seed}
    done
done
```

**Expected time**: 10 runs × 15h, parallel on ~5-10 GPUs → 15-30h.

## Priority 3 (optional) — Other encoder configurations

### 3a. Wider/narrower encoder (control for param count)

Modify `wijmans_policy.py` to parameterize `resnet_baseplanes` (currently 32):

```python
# Add config field: rl.ddppo.resnet_baseplanes
# Sweep: 8, 16, 32, 64
```

Tests whether the bottleneck effect is specifically from spatial feature
collapse or general encoder capacity. If narrow-encoder (baseplanes=8, 128×128
input) still gives chance-level GPS R², the spatial-collapse interpretation
is supported.

### 3b. Transformer backbone replacement

Replace LSTM with a transformer-based state encoder. Tests whether the
bottleneck→encoding relationship is LSTM-specific. Reviewer-concern
mitigation. Would take 2-4 weeks to set up — probably defer past NeurIPS
deadline.

## Why this matters for the paper

**Current state** (2-point observation):
> "Blind and matched-compute both encode spatial info in LSTM; uniform
> and foveated don't."

**With resolution sweep** (scaling law):
> "LSTM spatial encoding monotonically decreases with visual input
> resolution, from R²=0.95 at no-vision to R²≈0 at 256×256. At
> ~48×48 the transition occurs. This quantitative relationship is the
> computational signature of a capacity-driven compensation mechanism."

The scaling law is both a stronger empirical claim AND a quantitative
prediction reviewers can immediately verify/contest. 5-7 datapoints
along a curve is much more convincing than 2 isolated conditions.

## Summary for friend

**If compute is limited**: Just do Priority 1 (4 runs: matched-32, 64,
96, 192). ~15h on H100, gives us the scaling-law figure.

**If compute is abundant**: Priority 1 + Priority 2 (14 runs total).
Scaling law + error bars on main claim. ~15-30h with good parallelization.

**Even more abundant**: Add Priority 3a (encoder-width sweep). Shows
bottleneck effect is spatial-collapse specific, not just overall capacity.

## Integration with our workflow

Friend's outputs (checkpoints) will need to sync back to Izar for the
probe analysis step (which has the full probe infrastructure). Or friend
runs probes locally — `submit_probe_deterministic.sh` uses any SLURM
cluster, just needs GPU and our repo synced.

Both work. Preferred: friend runs training + probe both on their side,
pushes npz and analysis JSONs, we integrate into paper.
