# Scale x Sensor Pilot 8 Results

## Setup

Pilot 8 reuses Memory-Maze trajectories from:

`/scratch/wxu/dh-spatial/outputs/wmprobe_scale_sensor/pilot6/data`

Frozen visual encoders:

- small: `dinov2_vits14` (22M parameters)
- large: `dinov2_vitb14` (87M parameters)

Sensors:

- `foveated`
- `uniform`

The world-model run trained a recurrent feature-prediction LSTM per cell, then
tested whether position could be read from the recurrent state.

## Trajectory-Held-Out Probe

The original held-out-trajectory probe did not support the primary claim. All
conditions were near or below zero eval R2, so this readout is not a reliable
basis for a positive slide result.

Key comparison:

- `dinov2_vits14/foveated`: linear R2 = -0.1728
- `dinov2_vitb14/uniform`: linear R2 = -0.0793

Interpretation: do not use this as evidence for the broader implication.

## Frame-Level Spatial Readout

We then asked a simpler diagnostic question: within the same trajectory pool,
is position linearly readable from the recurrent state? This uses random
held-out frames rather than held-out trajectories. It is weaker than full
generalization, but it tests whether the memory format carries spatial
information at all.

Across five random frame splits:

| seed | DINOv2-S/foveated | DINOv2-B/uniform | delta |
|---:|---:|---:|---:|
| 0 | 0.5190 | 0.4739 | 0.0452 |
| 1 | 0.5207 | 0.4767 | 0.0439 |
| 2 | 0.5220 | 0.4725 | 0.0495 |
| 3 | 0.5266 | 0.4819 | 0.0447 |
| 4 | 0.5234 | 0.4797 | 0.0437 |

Mean:

- `dinov2_vits14/foveated`: 0.5223
- `dinov2_vitb14/uniform`: 0.4769
- delta: +0.0454 +/- 0.0024

## Capacity/Prediction Burden

The foveated input was also easier for the recurrent model to predict:

| encoder | sensor | final feature-prediction loss, last 50 |
|---|---:|---:|
| dinov2_vits14 | foveated | 0.2996 |
| dinov2_vits14 | uniform | 0.5283 |
| dinov2_vitb14 | foveated | 0.2066 |
| dinov2_vitb14 | uniform | 0.4352 |

## Slide-Safe Claim

Use cautiously:

> Pilot evidence: a task-aligned visual constraint preserved spatial
> readability while reducing recurrent prediction burden; in a within-pool
> frame-level readout, small foveated matched or exceeded large uniform.

Do not claim:

> Foveated small models outperform large uniform models on held-out navigation.

That stronger claim still requires a policy-level Habitat 2x2 experiment.

## Route Robustness Follow-Up

A second pilot used a supervised recurrent localizer instead of the
feature-prediction world model. This is the better place to include `blind`,
because predicting `agent_pos` forces the blind model to use the non-visual
motion route rather than letting it ignore blank visual input.

Inputs per step:

- frozen visual feature
- action
- heading
- start position

Target:

- current `agent_pos`

Evaluation:

- held-out trajectories
- window: steps 250-500
- 3 training seeds

| encoder | route / sensor | mean R2 | sd R2 | mean RMSE |
|---|---:|---:|---:|---:|
| dinov2_vits14 | blind / integration | 0.6478 | 0.0248 | 1.7026 |
| dinov2_vits14 | foveated / visual | 0.5315 | 0.0558 | 1.9864 |
| dinov2_vits14 | uniform / visual | 0.5541 | 0.0477 | 1.9342 |
| dinov2_vitb14 | blind / integration | 0.6192 | 0.0126 | 1.7728 |
| dinov2_vitb14 | foveated / visual | 0.5744 | 0.0199 | 1.8808 |
| dinov2_vitb14 | uniform / visual | 0.5534 | 0.0183 | 1.9268 |

Interpretation:

- This supports the route story: a motion-integrated blind route and sighted
  visual routes both carry position on held-out trajectories.
- It does not support the strongest scale claim: `dinov2_vits14/foveated`
  does not beat `dinov2_vitb14/uniform` in this held-out localizer.
- The safe empirical message is therefore route-level, not winner-takes-all:
  sensor structure changes the route by which spatial position becomes
  available to memory.
