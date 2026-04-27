# HC cluster experiment plan — friend's 14 trainings (with rationale)

**Author**: wxu (paper lead).  **For**: friend running on hc cluster (4× H100 + 1× H200).
**Deadline**: 2026-05-06 (NeurIPS submission).  **Today**: 2026-04-27.
**Net working days**: ~9.

This document is the single source of truth for what to run, why, and how to
ship results back.  It is verbose on purpose: when a training fails or the
prediction comes out unexpected, the *why-paragraph* tells you whether to
panic, retry, or just log it.

---

## TL;DR — what's at stake

The paper claims a **substitution mechanism (H1)**: when the visual encoder
produces enough information per step, the LSTM stops integrating GPS, and
the linear top-layer GPS code disappears.  Rich-encoder agents (uniform,
foveated) lose linear GPS at $\mathbf{h}_2$; bottleneck agents (blind,
coarse) preserve it.  So far this rests on **single-seed** results across
**5 fixed conditions**.  Reviewers will ask:

1. **Is this seed-1 noise?** → multi-seed replication (3 conditions).
2. **Is encoder bandwidth really the axis?** → scaling sweep at
   $K \in \{32, 64, 96, 128, 192\}$ input resolution.
3. **What if foveation isn't blur but spatial subsampling?** → log-polar
   falsifiable test.
4. **What about dynamic gaze (H3)?** → stochastic gaze policy.
5. **What about the foveated condition's NaN-corrupted ckpt.36?** →
   foveated_v2 clean re-run.

The 14 trainings below answer those five questions.  Tier 1 (5 trainings)
covers the must-haves; Tier 2 (5) fills out the scaling sweep + falsifiable
test; Tier 3 (4) is foveation completeness if time allows.

ETA per H100 / H200 training: **2–3 days** (vs ~5 on V100).

---

## What's already done — do NOT relaunch

**On Izar (V100, our side)** — these are converged or in progress; friend
should not duplicate them.

| Status | Run | Frames | What it gives the paper |
|---|---|---|---|
| ✅ Converged | `blind_gibson` (seed=1) | 342M | Fig 2 H1 baseline, R²(GPS\|h₂)=0.95 |
| ✅ Converged | `matched_gibson` (seed=1, "coarse" 48×48 → 1×1) | 250M | Fig 2 H1 anchor, R²=0.78 |
| ✅ Converged | `uniform_gibson` (seed=1) | 250M | Fig 2 H1 anchor, R²≈0 |
| ✅ Converged | `foveated_gibson` (fix, seed=1) | 174M | Fig 2 H1 anchor, R²≈0 |
| ✅ Converged | `foveated_learned_gibson` (seed=1) | 250M | Fig 2 H1 anchor, learned-gaze |
| ✅ Converged | `matched128_gibson` (seed=1) | 250M | Scaling sweep K=128 anchor (already probed) |
| 🔄 Running | `uniform_gibson_seed=2` | 140M / 250M | Multi-seed for uniform (~6 days remaining) |
| 🔄 Running | `foveated_gibson_seed=2` | 130M / 250M | Multi-seed for foveated (fix) (~6-9 days) |

> ⚠️ **K=128 is done**.  Tier 1 covers K=32 and K=64; Tier 2 covers K=96 and
> K=192.  K=48 is the "coarse" condition (already done as `matched_gibson`).
> So the full sweep is K ∈ {32, **48**=coarse, 64, 96, **128**, 192, **256**=uniform}
> with **bold = on Izar**, the rest = friend.

---

## Tier 1 — Day 1 (5 trainings, must-have)

### Training 1: Stochastic gaze (`foveated_stochastic_gibson`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_foveated_stochastic_gibson` |
| GPU | H200 (highest priority — most novel architecture) |
| Wall | ~2-3 days @ 250M frames |
| Submit | `sbatch <flags> scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_foveated_stochastic_gibson` |

**Motivation**:  Our H3 hypothesis is that *gaze dynamics* — not just static
gaze location — should affect whether substitution occurs.  Currently the
paper has only the *static* foveated and foveated-shifted (gaze locked at
the image center or at (0.49, 0.62)).  We need a *dynamic* gaze policy whose
gaze actually moves at rollout time.

The deterministic learned-gaze MLP we tried earlier collapses to a
near-constant gaze under PPO (Appendix on H3 documents two failed pilots).
The fix is a stochastic policy: gaze is sampled from a bounded Gaussian
$\mathcal{N}(\mu, \sigma)$ with $\sigma \in [0.05, 0.30]$ per environment,
giving it a permanent exploration floor.  See
`src/habitat/foveated_stochastic_policy.py`.

**Specific success criterion (write before training finishes!)**:

* **Per-env $\sigma > 0.05$ at convergence** → policy actually keeps gaze
  diverse → H3 is testable.
* **SPL ≥ 0.7 by 250M frames** → the gaze noise didn't ruin the navigation.
* If both true, then the §4.6 H3 question becomes a real comparison:
  static-foveated vs.\ stochastic-foveated R²(GPS\|h₂) and SPL drops.

**What it would mean**:

* Stochastic R² high (~0.5+) like coarse → gaze *dynamics* prevent
  substitution → "the encoder is unreliable per-step, so LSTM keeps GPS".
* Stochastic R² low (~0) like uniform → gaze dynamics *don't* save GPS →
  substitution is about average encoder bandwidth, not per-step variability.

Either outcome is publishable.  This is the highest-information experiment
on the list.

**Watch for**: if $\sigma$ collapses to 0.05 (its lower bound) and stays
there, the policy has effectively reverted to deterministic gaze — flag to
wxu before continuing.  This was a known failure mode in earlier pilots.

---

### Training 2: Scaling sweep K=64 (`matched64_gibson`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_matched64_gibson` |
| GPU | H100 |
| Wall | ~2 days @ 250M frames |
| Submit | `sbatch <flags> scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_matched64_gibson` |

**Motivation**:  The H1 substitution mechanism predicts that **encoder
spatial-output dimensionality** is the trigger axis.  A $64{\times}64$ RGB
input fed through ResNet-18 produces a $\sim 2{\times}2$ feature map.  This
sits between coarse's $1{\times}1$ (full GPS preservation) and uniform's
$8{\times}8$ (full substitution).

**Specific prediction (paper §App E)**:  R²(GPS\|h₂) somewhere in
$[\sim 0.4, 0.6]$ — partial preservation.  SPL between coarse's 0.84 and
uniform's 0.79 (likely ~0.82).

**What it tests**:  Whether the H1 axis is *smooth* (R² declines
monotonically as K rises) or *threshold-like* (R² stays high until some
critical K then drops).  This is the headline finding of the scaling sweep
appendix and the falsifiability core of H1.

**What it would mean if R² jumps directly from 0.78 (K=48) → ~0 (K=64)**:
the substitution mechanism is closer to a phase transition than a
gradient — paper framing would shift.

**What it would mean if R² stays near 0.78**: the relevant axis isn't
encoder spatial output but something else (input pixel count? channel
information?) — would force a re-think.

---

### Training 3: Scaling sweep K=32 (`matched32_gibson`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_matched32_gibson` |
| GPU | H100 |
| Wall | ~2 days @ 250M frames |
| Submit | `sbatch <flags> scripts/cluster/submit_train.sh pointnav/ddppo_pointnav_matched32_gibson` |

**Motivation**:  K=32 produces a $1{\times}1$ encoder feature map (same as
coarse's K=48), but with *even lower input resolution*.  This is the
**bandwidth lower bound** of the encoder spatial-output axis.  K=32 + K=48
(coarse) jointly check whether the $1{\times}1$ collapse is the active
ingredient or whether further input-pixel reduction matters.

**Specific prediction**:  R²(GPS\|h₂) ≥ coarse's 0.78 (because the encoder
is at least as bandwidth-starved as coarse).  SPL likely a bit lower than
coarse's 0.84 (the agent has fewer pixels to even check for collisions).

**What it would mean**:

* R² ≈ 0.78–0.95 → substitution depends on encoder spatial-output, not on
  input resolution per se → confirms the mechanism's "rate axis" framing.
* R² much lower than coarse's 0.78 → low input resolution by itself is enough
  to reduce GPS retention even when the encoder bottleneck is the same →
  weakens the "encoder-output is the trigger" claim, would force re-framing.

This is the **lower-end anchor** for the scaling sweep figure.

---

### Training 4: Multi-seed blind seed=2 (`blind_gibson seed=2`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_blind_gibson` (with `seed=2` override) |
| GPU | H100 |
| Wall | ~2-3 days @ 342M frames (blind takes more frames to converge) |
| Submit | `sbatch <flags> scripts/cluster/submit_train_seeded.sh pointnav/ddppo_pointnav_blind_gibson 2` |

**Motivation**:  Every cross-condition R² number in the paper is **single
seed**.  Reviewers will pounce on this — Wijmans 2023 reported R²=0.95 for
blind at seed=1, we replicate seed=1=0.95, but a full peer-reviewed claim
needs at least **mean ± std over 2 seeds** so reviewers know the number isn't
a 1-in-5 lucky run.  Blind is the cleanest baseline (no visual confound), so
it's the easiest seed=2 sanity check.

**Specific prediction**:  R²(GPS\|h₂) at seed=2 within [0.85, 0.95].
SPL at seed=2 within [0.45, 0.55].  If both fall in those ranges → seed
robustness confirmed → main paper §4.1 number gets `±std` upgrade.

**What it would mean**:

* R² in [0.85, 0.95] → seed-1 wasn't a lucky outlier → cleanly publishable.
* R² < 0.7 → seed-1 was unrepresentative → trigger panic, possibly run seed=3.
* R² inside expected range but SPL much lower → blind seed-2 didn't fully
  converge — extend training before probing.

---

### Training 5: Multi-seed matched seed=2 (`matched_gibson seed=2`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_matched_gibson` (with `seed=2` override) |
| GPU | H100 |
| Wall | ~2 days @ 250M frames |
| Submit | `sbatch <flags> scripts/cluster/submit_train_seeded.sh pointnav/ddppo_pointnav_matched_gibson 2` |

**Motivation**:  Same as blind seed=2, but for the *coarse* (matched 48×48
→ 1×1) condition.  Coarse is the most theoretically interesting baseline:
it has visual input but no usable encoder spatial output, so R²(GPS\|h₂) is
high *despite* having vision.  If coarse's R²=0.78 doesn't replicate at
seed=2, the entire "encoder spatial-output is the trigger" mechanism story
weakens.

**Specific prediction**:  R²(GPS\|h₂) at seed=2 within [0.65, 0.85].
SPL at seed=2 within [0.80, 0.90].

Together with blind seed=2, this gives us multi-seed numbers for **both
ends of the bottleneck regime**.  With foveated_learned seed=2 added in
Tier 2, we'll have 3-of-5 conditions with seed-2 — sufficient for an "all
H1 numbers replicate to within ±0.10 across seeds" caveat upgrade.

---

## Tier 2 — Day 3-4 (5 trainings, when Tier 1 frees up GPUs)

### Training 6: Scaling sweep K=96 (`matched96_gibson`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_matched96_gibson` |
| GPU | H100 |
| Wall | ~2 days |

**Motivation**:  Mid-sweep filling.  K=96 produces $\sim 3{\times}3$
encoder output.  Together with K=64 (~$2{\times}2$) and K=128 (~$4{\times}4$)
this gives a 3-point trace through the regime where R² should be smoothly
declining if the H1 axis is gradient-like.

**Specific prediction**:  R²(GPS\|h₂) in $[\sim 0.2, 0.4]$.

**What it tests**:  smoothness of the K → R² curve.  If K=64 → 0.6, K=96 →
0.3, K=128 → 0.1 — the curve is clean.  If K=64 → 0.6, K=96 → 0.5, K=128 →
0.1 — there's a knee, suggests a phase transition, paper figure caption
adjusts accordingly.

---

### Training 7: Scaling sweep K=192 (`matched192_gibson`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_matched192_gibson` |
| GPU | H100 |
| Wall | ~2 days |

**Motivation**:  High-end of the scaling sweep (close to uniform's K=256).
K=192 → ~$6{\times}6$ encoder output.  Tests whether R² has *bottomed out*
near uniform's $\approx 0$ by K=192, or whether it's still falling into
uniform.

**Specific prediction**:  R²(GPS\|h₂) in $[0, 0.15]$ — basically at-floor.

**What it tests**:  upper anchor of the scaling sweep figure.  Confirms
that the **R² → 0 regime** isn't just a uniform-specific quirk (different
visual style) but a continuous extrapolation of the matched-K series.

---

### Training 8: Multi-seed foveated_learned seed=2 (`foveated_learned_gibson seed=2`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_foveated_learned_gibson` (with `seed=2`) |
| GPU | H100 |
| Wall | ~2-3 days @ 250M frames |
| Submit | `sbatch <flags> scripts/cluster/submit_train_seeded.sh pointnav/ddppo_pointnav_foveated_learned_gibson 2` |

**Motivation**:  Foveated_learned is the **H3 anchor** in the paper — gaze
is predicted by an MLP head trained end-to-end.  Seed=1 gave R²=0.67 ±0.18
(early-train), declining in a manner specific to learned-gaze.  Multi-seed
needed to confirm the *decay-rate ordering* (uniform fastest, foveated
slower, foveated_learned in between) — currently the part of the paper
"most exposed to seed variability" by our own admission.

**Specific prediction**:  R²(GPS\|h₂) at seed=2 within [0.55, 0.75]
early-training, declining to chance later.  Decay timing should be similar
to seed=1 (peak around 50M frames, decay by 100-150M).

**What it would mean**:

* Seed=2 decay roughly matches seed=1 → "rich-encoder substitution
  timescale" claim survives multi-seed → keep §4.4 narrative.
* Seed=2 decay much faster or slower → substitution timescale is
  noisy across seeds → soften the "uniform fastest, foveated_learned
  middle" claim, possibly drop the per-condition decay-rate ordering.

This is the multi-seed condition with the **biggest paper-impact risk** —
explicitly flag results to wxu.

---

### Training 9: Foveated log-polar (`foveated_logpolar_gibson`) — F3 falsifiable

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_foveated_logpolar_gibson` |
| GPU | H100 |
| Wall | ~2 days |

**Motivation**:  This is the **falsifiable core of H1**.  Gaussian-blur
foveation (our standard `foveated_gibson`) preserves the encoder's
$8{\times}8$ spatial output even though peripheral content is blurred.
Log-polar foveation is *spatial subsampling* — non-uniform retinal grid
that drops the encoder spatial output to $\sim 2{\times}2$.  Per the H1
mechanism (encoder spatial-output → memory recruitment), log-polar should
behave *between coarse and uniform*, not like uniform.

**Specific prediction (written before result available!)**:
$$
R^2(GPS|h_2) \in [0.30, 0.65]
$$

**Falsifiable outcome**:

* R² ≥ 0.30 → mechanism survives → §App E remains the falsifiable
  prediction it's framed as.
* R² < 0.30 (matches uniform) → **mechanism is wrong** → encoder spatial
  output is *not* the trigger → would force a paper rewrite.

Wxu wrote the prediction down before training started so reviewers can see
this is a real falsifiable test, not post-hoc.

---

### Training 10: Foveated v2 clean re-run (`foveated_v2_gibson`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_foveated_v2_gibson` |
| GPU | H100 |
| Wall | ~2-3 days @ 250M frames |

**Motivation**:  Our seed=1 `foveated_gibson` ckpt.36 (174M frames) was
hit by a silent NaN-gradient corruption mode in DD-PPO that we discovered
late in the project (Appendix on training stability).  We patched the bug
and need a *clean* foveated re-run to verify that the H1 numbers
(R²(GPS\|h₂)≈0, SPL=0.75, MP3D shift) don't shift meaningfully in a
NaN-free environment.  This addresses §5.5(ii) limitations.

**Specific prediction**:  R²(GPS\|h₂) within [-0.1, +0.1] (still at floor).
SPL within [0.72, 0.80].

**What it would mean**:

* Numbers within ±0.05 of ckpt.36 → NaN bug didn't materially affect H1
  conclusions → safe to keep our results.
* Numbers shift by >0.1 → NaN bug was load-bearing → re-run on Izar with
  the clean ckpt becomes the canonical foveated number, paper figures get
  updated.

---

## Tier 3 — Day 6-7 (4 trainings, foveation completeness; skip if running tight)

### Training 11: Foveated σ=20 (`foveated_strong_gibson`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_foveated_strong_gibson` |
| GPU | H100 |
| Wall | ~2 days |

**Motivation**:  F4 *foveation strength* sweep, **high-blur endpoint**.
At σ_max = 20 the periphery is so blurred that the encoder's $8{\times}8$
output is effectively a center-only spotlight.  Tests whether stronger
blur produces uniform-like substitution (encoder dominates) or coarse-like
preservation (encoder so weak it can't substitute).

**Specific prediction**:  R²(GPS\|h₂) in $[0.3, 0.6]$ — partial preservation,
between coarse and standard foveated.

**Why it matters**:  Foveation under our model is currently single-
strength (σ=8).  A 4-point strength sweep (σ ∈ {2, 4, 8, 12, 20} with σ=4
explicitly skipped, see below) gives a continuous knob into the H1
substitution dynamics within the foveation family.

---

### Training 12: Foveated σ=2 (`foveated_sigma2_gibson`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_foveated_sigma2_gibson` |
| GPU | H100 |
| Wall | ~2 days |

**Motivation**:  F1 *foveation strength* sweep, **low-blur endpoint**.
σ=2 is barely-foveated (peripheral degradation is mild).  Predicted to
behave like uniform — substitution as strong as the encoder allows.

**Specific prediction**:  R²(GPS\|h₂) in $[-0.1, +0.2]$ — at-floor like
uniform.

**Why it matters**:  Confirms that the foveation effect we observe at
σ=8 isn't an artifact of *some* peripheral blur — when blur is light, the
condition reverts to uniform-like substitution.

---

### Training 13: Foveated σ=12 (`foveated_sigma12_gibson`)

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_foveated_sigma12_gibson` |
| GPU | H100 |
| Wall | ~2 days |

**Motivation**:  F1c mid-strength.  Fills the σ ∈ {2, 8, 12, 20} sweep
between standard (σ=8) and strong (σ=20).  Useful for the F1c monotonicity
plot in App E.

**Specific prediction**:  R²(GPS\|h₂) in $[0.1, 0.4]$.

---

### Training 14: Foveated shifted (`foveated_shifted_gibson`) — H3 static control

| | |
|---|---|
| Config | `pointnav/ddppo_pointnav_foveated_shifted_gibson` |
| GPU | H100 |
| Wall | ~2 days |

**Motivation**:  Static gaze hardcoded at $(0.49, 0.62)$ rather than image
center $(0.5, 0.5)$.  This is the **H3 static control** — same architecture
as foveated (fix), only the gaze location changes.  It tests whether the
behaviour difference between foveated (fix) and uniform is *because* gaze
is at center, or *because* gaze is fixed.

**Specific prediction**:  R²(GPS\|h₂) within ±0.15 of standard foveated
(fix)'s ≈0; SPL within ±0.05 of foveated's 0.75.

**Why it matters**:  Pairs with stochastic gaze (Tier 1) for the H3
section — static-shifted vs.\ static-center vs.\ stochastic = three points
on the gaze-mobility axis.  Without this, we can't claim "gaze location"
and "gaze dynamics" are separable axes.

---

## Skip-list — do NOT run

| Config | Why skip |
|---|---|
| `foveated_sigma4_gibson` | Too close to σ=2 / σ=8; marginal information value. |
| `foveated_normaliser_gibson` | F2 normalizer ablation; already in App E from Izar runs. |

If Tier 3 finishes ahead of schedule and you have spare GPUs, ping wxu
before launching anything not on this list.

---

## Pre-flight checklist (before launching anything)

```bash
cd /path/to/cs503-project

# 1. Pull latest repo state
git pull origin main
# Latest commit at submission prep ≥ 9684d3e (or ask wxu for the head).

# 2. Verify the 12 configs exist
ls habitat_configs/ddppo_pointnav_{foveated_stochastic,matched32,matched64,matched96,matched128,matched192,foveated_logpolar,foveated_v2,foveated_strong,foveated_sigma2,foveated_sigma12,foveated_shifted}_gibson.yaml
# Should list 12 files.

# 3. Verify the policies are registered
python -c "
import sys; sys.path.insert(0, '/path/to/cs503-project')
import src.habitat
from habitat_baselines.common.baseline_registry import baseline_registry
for name in [
    'FoveatedStochasticGazePolicy',
    'FoveatedSigma2WijmansPolicy',
    'FoveatedSigma12WijmansPolicy',
    'FoveatedStrongWijmansPolicy',
    'FoveatedLogPolarWijmansPolicy',
    'FoveatedShiftedGazePolicy',
]:
    cls = baseline_registry.get_policy(name)
    print(f'{name}: {cls is not None}')
"
# All should print True.

# 4. Smoke test the new stochastic gaze policy (under 1 min)
python scripts/cluster/smoke_policy.py FoveatedStochasticGazePolicy
# Should print "policy forward pass ok" (or equivalent).

# 5. Dataset check (run this even if you ran the dataset setup yesterday)
ls $HABITAT_DATA/datasets/pointnav/gibson/v1/train_extra_large/content/ | wc -l   # Expect 411
ls $HABITAT_DATA/datasets/pointnav/mp3d/v1/train/content/ | wc -l                   # Expect 61
ls $HABITAT_DATA/datasets/pointnav/mp3d_gibson/v1/train/content/ | wc -l            # Expect 472
# If any number is wrong, see docs/DATASET_SETUP.md.
```

If any of (1)–(5) fails, **do not launch anything** — ping wxu first.

---

## How to ship trained checkpoints back to Izar

The friend's hc cluster has no shared filesystem with EPFL's Izar.  Shipping
is via **rsync over SSH using a deploy key** (one-time setup below).

### One-time setup (~5 minutes)

```bash
# On hc cluster, create a fresh deploy keypair
ssh-keygen -t ed25519 -f ~/.ssh/id_izar_wxu_deploy -N ''

# Send the public key (~80 chars one line) to wxu via Slack / email.
cat ~/.ssh/id_izar_wxu_deploy.pub
```

wxu adds it to Izar's `~/.ssh/authorized_keys`; friend confirms with:

```bash
ssh -i ~/.ssh/id_izar_wxu_deploy wxu@izar.epfl.ch 'echo connected; date'
```

### Per-run shipping

After each training hits 250M frames (or 8-day partial — whichever first):

```bash
bash scripts/cluster/ship_to_izar.sh <RUN_NAME>
# e.g.: bash scripts/cluster/ship_to_izar.sh foveated_stochastic_gibson
```

This rsyncs `latest.pth` + `ckpt.{10,20,30,40,49}.pth` + `tb/` (training
curves) into `/scratch/izar/wxu/habitat_checkpoints/<RUN_NAME>/`.

wxu's `probe_hc_arrival.sh` cron on Izar detects the new files within 30
minutes and auto-submits the probing pipeline.  No further action needed
from friend after running the ship script.

### Fallback if SSH outbound is blocked

(rare on academic clusters but possible)

```bash
# rclone with a shared Google Drive folder
rclone copy <ckpt-dir>/ shared:cs503_paper/<RUN_NAME>/
# wxu then pulls: rclone copy shared:cs503_paper/ /scratch/izar/wxu/habitat_checkpoints/
```

3.5 GB total transfer for all 14 runs — comfortable for any method.

---

## Daily check-in flow

| Frequency | Friend does | wxu does |
|---|---|---|
| Every morning | `squeue -u $USER` snapshot → Slack | Confirm crons clean on Izar |
| When training hits 250M | run `ship_to_izar.sh <run>` | wait for `probe_hc_arrival.sh` to detect |
| When ship script printed "ok" | confirm to wxu via Slack | run probing on landed ckpts |
| If anything weird | flag wxu BEFORE acting | diagnose |

The probing pipeline on Izar takes 20–40 min per run.  R² numbers + figs
update on the wxu side; friend doesn't need to look at them unless wxu
flags an unexpected result.

---

## Failure signatures — when to escalate

| Symptom | Likely cause | Action |
|---|---|---|
| `nan_sanitised > 0` in TB metrics | Numerical instability (rare on H100/H200; we patched the worst case). | Continue training; the fix absorbs it. |
| Stochastic gaze: σ → 0.05 (its lower bound) and pinned | PPO is suppressing exploration. | Continue 50M more frames; if still pinned, ping wxu. |
| Stochastic gaze: SPL < 0.5 by 50M frames | Too much gaze noise hurting navigation. | Continue but flag — may need to reduce σ_max. |
| Scaling sweep K=N converges to coarse-1×1 R² when expected to be lower | Encoder collapsing earlier than predicted. | Continue, this would actually *support* the substitution mechanism (good!) — flag for paper. |
| OUT_OF_MEMORY on H100 | Batch size / num_envs mismatch. | Reduce `num_environments` from 4 → 2 in sbatch wrapper, ping wxu. |
| Job killed by walltime | Just resubmit from latest ckpt. | The training picks up from `latest.pth`. |
| Probe-pipeline output (on Izar) stalls for 24h after ship | wxu's cron may have died. | Ping wxu. |

---

## Rough daily schedule

```
Day 1 (today)    Pre-flight checks. Launch Tier 1 (5 trainings).
Day 2-3          Tier 1 training; first ckpts ship to Izar at ~125M frames.
Day 3-4          Tier 1 finishes. Ship final ckpts. Launch Tier 2 (5).
Day 5-6          Tier 2 training. wxu probes Tier 1 results.
Day 6-7          Tier 2 finishes. Launch Tier 3 (4) if on schedule.
Day 8            Tier 3 finishes. Final ship. wxu probes everything.
Day 9 (2026-05-06) Final paper integration. Submit.
```

Slip allowance: each tier has 1 day of slack.  If Tier 1 takes 4 days
instead of 3, drop Tier 3 entirely and run only Tier 1+2.

Submission target: clean NeurIPS submission with **multi-seed (3 conditions)
+ scaling sweep (5 K-points) + stochastic gaze H3 + falsifiable log-polar**.
That's the minimum credible set per our review-risk analysis.

---

## Quick links

- Dataset setup: [docs/DATASET_SETUP.md](DATASET_SETUP.md)
- Paper TeX: [docs/NeurIPS_2026/neurips_2026.tex](NeurIPS_2026/neurips_2026.tex)
- Sleep log (wxu's autonomous overnight progress): [docs/NeurIPS_2026/SLEEP_LOG.md](NeurIPS_2026/SLEEP_LOG.md)

---

*— wxu, 2026-04-27.  If anything in this doc seems wrong / missing, ping
me on Slack before improvising.*
