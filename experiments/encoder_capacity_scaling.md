# Experiment: Encoder-capacity scaling law

## Intention

Our core finding is **visual encoder capacity inversely determines LSTM
spatial encoding** — blind and matched-compute (1×1 encoder) develop
cognitive-map-like GPS codes in their recurrent state, while full-RGB
encoders do not. We have two points in the bottleneck regime (blind,
matched-48) and three saturated points at full capacity (uniform,
foveated, foveated-learned). To upgrade this from a monotonic ordering
to a **scaling law**, we need intermediate points along the encoder-
capacity axis.

Current 5-condition data (5-fold CV, deterministic probe):

| Condition | Input res | ResNet18 feat map | GPS R² | Compass R² |
|-----------|-----------|-------------------|--------|-----------|
| Blind | — | — | **0.95 ± 0.02** | 0.81 ± 0.08 |
| Matched-48 | 48×48 | 1×1 × 256 | **0.78 ± 0.10** | 0.64 ± 0.10 |
| Uniform | 256×256 | 8×8 × 256 | -0.31 ± 0.86 | 0.36 ± 0.23 |
| Foveated | 256×256 blurred | 8×8 × 256 | +0.06 ± 0.88 | 0.07 ± 0.69 |
| Fov-learned | 256×256 + gaze | 8×8 × 256 | -2.43 ± 3.98 | -1.34 ± 3.14 |

## Plan

Sweep input resolution through ResNet18 with everything else held
fixed. ResNet18 has total stride 32, so spatial feature map is
`max(1, res/32) × max(1, res/32)`.

### Sweep points needed

| Res | Config | Status | Who does what |
|-----|--------|--------|---------------|
| 32×32  | `matched32_gibson`  | **not trained** | **train + probe here** |
| 48×48  | `matched_gibson`    | trained, det probe done | — |
| 64×64  | `matched64_gibson`  | **not trained** | **train + probe here** |
| 96×96  | `matched96_gibson`  | **not trained** | **train + probe here** |
| 128×128 | `matched128_gibson` | trained on Izar (ckpt.49), det probe **missing** | **det probe only** |
| 192×192 | `matched192_gibson` | **not trained** | **train + probe here** |
| 256×256 | `uniform_gibson`    | trained, det probe done | — |

So this experiment = **4 new trainings** (32, 64, 96, 192) + **1 det
re-probe** of the Izar-trained matched128 checkpoint. All configs
already committed in `habitat_configs/`:
- `ddppo_pointnav_matched32_gibson.yaml`
- `ddppo_pointnav_matched64_gibson.yaml`
- `ddppo_pointnav_matched96_gibson.yaml`
- `ddppo_pointnav_matched192_gibson.yaml`

### Submit (on any SLURM cluster with our repo synced)

```bash
for res in 32 64 96 192; do
    sbatch --gres=gpu:1 --time=24:00:00 \
        scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_matched${res}_gibson
done
```

**Per-run time**: ~10–15h on H100/H200 for 250M frames. 4 runs in
parallel → all done in ~15h.

### Probe the trained checkpoints

```bash
# 4 new sweep points
for res in 32 64 96 192; do
    sbatch scripts/cluster/submit_probe_deterministic.sh \
        pointnav/ddppo_pointnav_matched${res}_gibson \
        /path/to/checkpoints/matched${res}_gibson/ckpt.49.pth 500
done

# Plus the Izar-trained matched128 checkpoint (needs det re-probe).
# This ckpt is at /scratch/izar/wxu/habitat_checkpoints/matched128_gibson/ckpt.49.pth
# on the Izar cluster; sync it to friend's cluster first, or run this
# probe on Izar directly.
sbatch scripts/cluster/submit_probe_deterministic.sh \
    pointnav/ddppo_pointnav_matched128_gibson \
    /path/to/matched128_gibson/ckpt.49.pth 500
```

Probe time: ~2–4h each on H100. 5 parallel → ~4h.

Output: `probing_data/matched${res}_gibson_det.npz`
        `probing_results/matched${res}_gibson_det_analysis.json`

## Expected outcome

GPS $R^2$ monotonically decreases with resolution. Prediction:

```
matched32:   R² ≈ 0.90 (very strong bottleneck, near-blind regime)
matched48:   R² = 0.78 (observed)
matched64:   R² ≈ 0.50–0.70
matched96:   R² ≈ 0.20–0.50
matched128:  R² ≈ 0.0–0.3
matched192:  R² ≈ 0 (saturated pass-through)
uniform256:  R² ≈ 0 (observed)
```

If observed, this gives a clean scaling curve — 7 points along the
encoder-capacity axis — which turns a monotonic ordering into a
testable prediction that reviewers can verify and contest.

## Integration

Friend's outputs (npz + analysis JSONs) sync into the same
`/probing_data/` and `/probing_results/` folders as Izar's. The hero
figure (scripts/paper_figures/make_bottleneck_figure.py) extends
naturally to include the sweep points as additional bars or a scatter
plot with resolution on the x-axis.

Paper appendix will report the scaling curve as quantitative support
for the bottleneck hypothesis.
