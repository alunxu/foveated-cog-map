# Foveation & Gaze Design for the PoC Run

Date frozen: 2026-04-09.
Scope: the first end-to-end 4-condition Habitat Gibson study that goes
into the CS503 report. This document pins every dial *before* we burn
machine-weeks so the four runs are comparable.

Non-goal: biologically calibrated foveation, log-polar cortical
magnification, or foveation hyperparameter sweeps. Those belong in a
follow-up, not the PoC.

## 1. The experimental logic (why we need these four)

| Condition | RGB input | Purpose in the table |
|---|---|---|
| **Blind** | — | Baseline: what can path integration alone solve? Replicates Wijmans et al. 2023. |
| **Uniform** | 256×256 full-res | Upper bound on vision-driven performance at this architecture. |
| **Foveated** | 256×256, blurred periphery | The condition the paper is about: same image size as uniform, same ResNet18, but perceptual information concentrated at gaze center. |
| **Matched-compute** | 64×64 uniform | Controls for *total information*. If foveated ≥ matched-compute, the spatial *distribution* of information matters; if foveated ≈ matched-compute, only the budget matters. |

H1 (perceptual bottleneck): Uniform > Foveated ≥ Matched-compute > Blind
(in success and SPL).

H2 (maps still emerge under foveation): a linear probe on the LSTM
hidden state predicts global agent pose, more strongly in foveated than
in matched-compute, *despite* the peripheral information loss.

H3 (gaze-memory coupling): *only* tested in the appendix ablation with
learned gaze. Not in the main table.

## 2. Decisions

### 2.1 Foveation transform

Currently implemented in `src/habitat/torch_foveation.py` as a per-pixel
multi-scale Gaussian blend:

```
sigma(ecc) = sigma_max · f(ecc)        ecc = |p − gaze| normalized
foveated(p) = interp(blur_stack, level(sigma(p)))
```

**Frozen PoC defaults** (YAML + policy constructor):

| Parameter | Value | Rationale |
|---|---:|---|
| `image_size` | 256 | Matches the uniform RGB sensor size; keeps the ResNet18 encoder identical across all three sighted conditions. |
| `fovea_radius` | 16 px | ~6% of width. In primate vision the high-acuity fovea is ~1-2° out of ~180° FoV (~0.5-1%); 6% is a compromise that keeps enough signal for PPO to learn but is clearly narrower than uniform. |
| `blur_sigma_max` | **8.0** (was 6.0) | Sigma 6 left the periphery still readable in a quick visual-inspection sanity check — foveated ≈ uniform would kill the experimental contrast. Sigma 8 pushes the cutoff frequency to roughly `f_c ≈ 0.35/σ ≈ 0.044 cyc/px` at the image edge, i.e. features smaller than ~23 px are unresolvable. |
| `falloff` | `quadratic` | Sharp near the fovea, soft far out. A quadratic profile (`ecc²`) is a crude but smooth approximation to the cortical magnification ~`1/ecc` mapping. Linear is too aggressive near the center; exponential is too aggressive far out. |
| `n_levels` | 5 | The blend is smooth enough; 7 and 9 showed no qualitative difference in the sanity check. |

### 2.2 Gaze model — **fixed center**

The cleanest scientific comparison to uniform is one where the only
thing that changed is the *perceptual bandwidth profile*. Learning a
gaze controller simultaneously adds a second learning problem (where to
look) that PPO gets no direct reward signal for, and empirically (our
own MiniGrid run) the gaze distribution stayed close to uniform-random
for hundreds of millions of steps.

→ **Main table uses `FoveatedWijmansPolicy` with gaze hardcoded to
(0.5, 0.5)**. This is already the behavior in `foveated_policy.py`.

The learned-gaze variant (`FoveatedLearnedGazePolicy`) remains in the
repo and is earmarked for an appendix ablation: "does emergent gaze
change the picture?" — run only if time permits and only *after* the
main four conditions are in.

### 2.3 Matched-compute calibration

Matched-compute is a uniform-but-low-resolution control. Its job is to
match the *total perceptual bandwidth* of the foveated agent. A rough
pixel-budget count with the frozen foveation parameters:

- Sharp foveal disk (zero blur): π·r² ≈ π·16² ≈ 804 pixels at full res.
- Peripheral annulus: (256² − 804) ≈ 64 732 pixels, each carrying
  information at an effective resolution of ~1/σ (= 1/8) of the
  original. Effective pixel contribution ≈ 64 732 / 64 ≈ 1 012 pixels.
- **Total effective pixel budget ≈ 1 816**. A uniform image with the
  same budget is √1 816 ≈ **43×43 pixels**.

The matched-compute config currently ships at **64×64** — a bit
generous (64² = 4 096 ≈ 2.3× the foveated budget). For the PoC we
tighten this to **48×48** (48² = 2 304, 1.27× budget — still a hair
generous but playing safe on the control side: matched-compute has a
slight advantage, so if foveated beats it the result is unambiguous).

Action: edit `ddppo_pointnav_matched_gibson.yaml` RGB sensor from
64×64 → 48×48.

### 2.4 Training budget

| Condition | `total_num_steps` | Reason |
|---|---:|---|
| Blind | 5.0e8 | Wijmans curve needs ~500 M to saturate near 0.95. Already running. |
| Uniform | 2.5e8 | Sighted converges much faster; 2.5e8 is already plenty (at 23 M / 10 h we're at 73 %). Already running. |
| **Foveated** | **2.5e8** | Same architecture class as uniform so same horizon. |
| **Matched-compute** | **2.5e8** | Same. |

2.5e8 ≈ 110-130 wall-clock hours per 2×V100 run (extrapolating from the
uniform job's current fps). Realistically one of them will finish on
the SLURM 72 h wall time cap and need one resume; that's fine, the
checkpoint path supports it.

## 3. Code changes this decision implies

1. `habitat_configs/ddppo_pointnav_foveated_gibson.yaml`: bump
   `blur_sigma_max` comment to 8.0. The actual value lives in
   `src/habitat/foveated_policy.py` — change the default from 6.0 → 8.0
   there and make sure it propagates into the `FoveatedWijmansNet` and
   `FoveatedResNetEncoder` constructors. The config file's current
   note is already correct: Hydra structured-config validation rejects
   unknown fields, so the actual hyperparameters must live in Python.

2. `habitat_configs/ddppo_pointnav_matched_gibson.yaml`: RGB sensor
   `height`/`width` 64 → 48.

3. No code change needed for the gaze model — `FoveatedWijmansPolicy`
   already renders with a fixed center gaze.

## 4. Run order & scheduling

The `cs-503` QOS on Izar caps us at 2 concurrent jobs. Current state:

- 2826964 `habitat_blind_gibson` — running, ~10 h in
- 2826965 `habitat_uniform_gibson` — running, ~10 h in

Plan:

1. Let blind + uniform keep running. Both are on track; no reason to
   cancel (see [project_status.md](../memory/project_status.md)).
2. When either finishes (or hits the 72 h wall cap and stops), submit
   the next condition from the queue — order: **foveated first**,
   **matched-compute second** (foveated is the higher-variance /
   higher-risk condition, we want early signal on it).
3. If neither finishes in 48 h, switch the foveated + matched-compute
   submissions to the `normal` QOS instead (longer queue, no
   concurrent-job cap) and accept the queue delay.

The README current-status section and `memory/project_status.md` are
the source of truth for what's running when.

## 5. What this intentionally does not do

- **No learned gaze in the main table.** See 2.2. Appendix ablation only.
- **No biologically calibrated foveation.** Quadratic falloff is cheap,
  defensible, and lets the reader re-derive the effective pixel budget
  in a paragraph. A cortical-magnification model is a follow-up paper.
- **No multi-seed runs for the PoC.** Each condition is one seed.
  Multi-seed comes after we know the main trend holds.
- **No foveation hyperparameter sweep.** One setting per the table
  above. Sweeps are expensive and the PoC must show the *qualitative*
  story first.
