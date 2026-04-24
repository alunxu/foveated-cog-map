# NeurIPS 2026 Audit Findings

This file collects audit findings across categories. Each agent appends its own
section; headings are unique to avoid collisions.

---

## Category E — Custom policies

Files audited:
- `src/habitat/wijmans_policy.py`
- `src/habitat/foveated_policy.py`
- `src/habitat/foveated_learned_policy.py`
- `src/habitat/foveated_shifted_policy.py`
- `src/habitat/torch_foveation.py`

### E1. Foveation eccentricity saturates at corners — MINOR (claim-level mismatch)

**File/line**: `src/habitat/torch_foveation.py:58`, `:131-132`

**Observation**: `self._max_dist = (2 ** 0.5) * image_size / 2` is the center-to-
corner distance (≈ 45 px for a 64×64 foveated buffer). When gaze is at an edge
or corner, per-pixel distances to the far corner reach `sqrt(2) * image_size`
(≈ 90 px for 64×64) — about 2× larger than `_max_dist`. The eccentricity
`(dist - fovea_radius) / (_max_dist - fovea_radius)` is then clamped to `[0,1]`
at line 132, so the far-corner periphery **saturates at maximum blur** rather
than scaling smoothly.

**Why it matters**: for the foveated-learned final gaze `(0.49, 0.62)` the
offset from center is small (≈ 8 px in the 64×64 buffer), so almost all
pixels remain in the well-defined eccentricity range and the saturation
regime is barely entered. The claim in §H3 ("gaze 30 px below image centre")
is computed relative to the 128×128 raw image — after `avg_pool2d(2)` this is
≈ 15 px in the 64×64 buffer, still well within the `_max_dist` support.
**Results not invalidated**, but if the paper shows a saturation/edge study
the description should note the clamp behaviour.

**Severity**: **MINOR**. Proposed fix: document in Appendix the `_max_dist`
choice and the resulting saturation at image corners; no code change
necessary unless new experiments place gaze near image edges.

### E2. Foveation uses `_max_dist` relative to image center, not current gaze — MINOR (documentation only)

**File/line**: `src/habitat/torch_foveation.py:58`

**Observation**: `_max_dist` is a scalar pre-computed at construction time,
using `image_size / 2` (center). It is gaze-independent, so the normalisation
of eccentricity is not tight for off-center gaze. For gaze `(0.5, 0.5)` a
pixel at the corner has ecc → 1 (fully blurred). For gaze `(0.49, 0.62)`
a pixel at the corner has raw distance slightly >`_max_dist`, clamped to 1.

**Why it matters**: the effective *blur magnitude* (not just the map shape)
depends on gaze position, through the clamp. This is a subtle confound: the
shifted-gaze condition and the fixed-center condition do not receive the
exact same distribution of (ecc → sigma) transformations. It is
small in practice because the gaze shift is ~15 pixels, but for the H3
gaze-location claim it is a systematic confound.

**Severity**: **MINOR**. Fix is a one-line code change (`_max_dist` becomes
per-image dynamic: `sqrt((W-1)^2 + (H-1)^2)`) or — better — document the
design choice in the paper. Proposed paper sentence for §H3 foveated-shifted:
"The Gaussian-blur envelope is normalised to the image-centre-to-corner
distance, so a shifted gaze at (0.49, 0.62) places slightly more of the
image in the saturated-blur regime than a centred gaze does. The magnitude
of this asymmetry is small (clamped eccentricity is ~1% higher at the
farthest corner)."

Doesn't invalidate existing foveated or foveated-shifted results (same
constant `_max_dist` across fixed/shifted/learned variants).

### E3. `foveated_shifted_policy.py` has unused `gaze_hidden` + outdated gaze_decoder doc — MINOR

**File/line**: `src/habitat/foveated_shifted_policy.py:155`, `:214-224`

**Observation**: `FoveatedShiftedGazeNet` inherits from `WijmansPointNavNet`
(not from `FoveatedWijmansNet`), duplicating the entire foveation setup. The
class-level docstring at line 131-138 still says "adds a small MLP that
decodes a 2-D gaze position from the previous LSTM hidden state" — this is
wrong; the policy uses a hardcoded `(0.49, 0.62)`, no MLP. The
`gaze_hidden=64` kwarg is accepted and threaded through but unused.

**Why it matters**: documentation debt. Someone modifying this policy could
be confused about whether a gaze decoder is active. The commented-out
gaze_decoder block is copy-pasted from `foveated_policy.py` and should be
deleted.

**Severity**: **MINOR**. Verified at line 256-260: gaze IS the fixed
`[[0.49, 0.62]]` as claimed. All other hyperparameters
(`fovea_radius=16`, `blur_sigma_max=8.0`, `falloff="quadratic"`,
`hidden_size=512`, `num_recurrent_layers=3`, `backbone="resnet18"`,
`resnet_baseplanes=32`) match `foveated_policy.py`. The shifted policy
is **functionally identical** to foveated-fixed except for the gaze
location.

**Fix**: (a) rewrite docstring; (b) drop unused `gaze_hidden` kwarg or
add a comment; (c) optionally refactor to inherit from `FoveatedWijmansNet`
to avoid duplicated foveation setup. Does not invalidate results.

### E4. Gaze-decoder collapse mechanism — OK (architecture matches claim)

**File/line**: `src/habitat/foveated_learned_policy.py:116-121`, `:156-190`

**Verification**: the gaze decoder is exactly as claimed:
```
nn.Sequential(
    nn.Linear(hidden_size=512, gaze_hidden=64),
    nn.ReLU(),
    nn.Linear(gaze_hidden=64, 2),
    nn.Sigmoid(),
)
```
Mapping: previous LSTM hidden state → 2-D gaze in `[0, 1]²`. Input to the
decoder: `rnn_hidden_states.detach().mean(dim=1)` — per-env average across
the 6 (3 layers × (h,c)) state channels, shape `(num_envs, hidden_size)`
(line 157-159). Detach is justified in comment to avoid in-place autograd
conflict with the LSTM state_encoder.

**No architectural bug forcing collapse**: a ReLU MLP of this size is
trivially rich; sigmoid output is bounded but not saturated at init
(init weights are near zero → pre-sigmoid ≈ 0 → gaze ≈ 0.5). The observed
collapse to `(0.49, 0.62)` is *learning dynamics*, not a dead-ReLU or
vanishing-gradient artefact: the gradient path from PPO loss backward
through LSTM → foveation → gaze_decoder is long, diluted, and easily
dominated by other parameter updates. This matches the `gaze_diversity_loss`
module's diagnosis (line 7-13).

**Severity**: **OK**. The paper's claim "gaze decoder converged to near-constant
(0.49, 0.62) with std < 2.5e-4" is consistent with the implementation; the
decoder is architecturally sound.

### E5. Slow-gaze update semantics — OK with caveat

**File/line**: `src/habitat/foveated_learned_policy.py:142-190`

**Verification**: gaze is computed ONCE from the initial hidden state of
each rollout segment (via `rnn_hidden_states.detach().mean(dim=1)`) and
broadcast across all timesteps via `gaze_per_env.repeat(num_steps, 1)`
(line 190). During `act()` (per-step inference) `total_batch == num_envs`
so no broadcast; during `evaluate_actions()` (PPO update) it's T-major
tiling. This matches the "slow-gaze, updates every N steps" claim — N
equals `num_steps` per rollout, which is 128 for uniform and 256 for blind
(from YAML configs). The paper mentions 256 in §H3.

**Caveat (MINOR)**: the paper says "once per 256-step rollout segment" but
this is only exactly 256 for the blind config; the foveated/uniform configs
use `num_steps: 128`. Check the actual learned-gaze YAML config if the paper
cites 256 — if the foveated-learned config uses 128 not 256, this should be
corrected in §3 and §H3.

**Severity**: **MINOR-to-OK**. Mechanism is correct; only a numerical
detail in the paper might need updating.

### E6. Blind policy `normalize_visual_inputs` bypass — OK

**File/line**: `src/habitat/wijmans_policy.py:301`, `:521`

**Verification**: at line 521, `normalize_visual_inputs="rgb" in
observation_space.spaces` is False for blind. `ResNetEncoder.__init__`
therefore does NOT instantiate `RunningMeanAndVar(0)`. The assertion
`assert self._n_input_channels > 0` (inside habitat-baselines) is only
triggered when `normalize_visual_inputs=True` AND `_n_input_channels=0`.
Our code sidesteps it.

**The README patch (`normalize_visual_inputs and self._n_input_channels > 0`)
is a defensive upstream-patch recipe; our code bypasses the bug
differently by never passing True for blind.** This bypass is **correct**.
Not all users need the patch — only those who construct the encoder with
both flags True simultaneously.

**Severity**: **OK**. Recommend clarifying README §7.3: "The assertion is
only a problem if you build a sighted config with `normalize_visual_inputs:
True` and then set `force_blind_policy: True`; our provided configs do not
trigger this path."

### E7. NaN sanitisation on whole-batch failure — OK (design choice)

**File/line**: `src/habitat/wijmans_policy.py:156-176`

**Verification**: `_safe_before_step` iterates all parameters, detects
non-finite grad elements, and uses `torch.nan_to_num_` to replace them with
zeros. Then `clip_grad_norm_` (called inside `_original_before_step`) sees
zero for those slots, so `optimizer.step()` writes a zero update for every
non-finite element.

**Important subtlety**: the patch does NOT skip `optimizer.step()`. If ALL
grads in a batch are non-finite, all grads become zero → the step is a
no-op. If only SOME are non-finite, the non-finite ones become zero but
others still update. This is the documented "zero those elements" behaviour;
the question asked whether step is *skipped*. It is NOT — the step runs but
with zeros in the bad slots, which is equivalent to skipping for those
params.

Is there a risk of Adam/optimizer momentum absorbing the zero into its
running state and diverging? In practice, Adam with zero grad → zero
update but momentum/variance terms do update (denominator becomes smaller
if grad is zero). Over many consecutive NaN batches this could bias Adam
state. Evidence from the live run: `NAN_SANITISATION_STATS["total_events"]`
is tracked, the paper's checkpoint loading shows no sustained NaN
corruption, and the sanitisation is cited as rare.

**Severity**: **OK / MINOR**. The design doc is correct that "the step is
a no-op for that mini-batch — the safe behaviour" **when all grads are
non-finite**. For partial cases, the step continues with a zeroed slice.
This matches the intended behaviour. Suggested minor improvement: log the
fraction of non-finite grads per batch (currently we log only binary
event-or-not), so the paper can report the total count of
actually-corrected updates vs. spurious trigger. Not results-invalidating.

### E8. `RunningMeanAndVar` disabled for foveated — OK (gradient path note)

**File/line**: `src/habitat/foveated_policy.py:183-197`, `:197`

**Verification**: in `FoveatedResNetEncoder`, `normalize_visual_inputs=False`
is FORCED regardless of the passed kwarg (line 197). Comment at line 183-191
explains: the in-place `_count` buffer update in `RunningMeanAndVar` conflicts
with autograd when there's a parallel gradient path through the gaze decoder.
Disabling it yields an identity module.

**Note**: uniform and matched use `RunningMeanAndVar` (standard path), so
foveated visual features are on a **different normalisation regime** than
uniform. This is a known confound in H1/H2 comparisons: foveated vs uniform
differ in (a) spatial blur (the intended manipulation) AND (b) input
normalisation (side-effect). The paper's "pass-through" finding for both
foveated and uniform suggests this did not change the qualitative outcome.

**Severity**: **OK**, but recommend a one-line methods note: "The foveated
conditions use an identity-module substitute for `RunningMeanAndVar` to
avoid an autograd-vs-in-place conflict along the gaze-decoder gradient
path. Uniform and matched use the standard running-norm." This explanation
is already buried in the code comment; it should surface in the paper.

### E9. Logit clamp `_LOGIT_CLAMP = 10.0` — OK

**File/line**: `src/habitat/wijmans_policy.py:86`, `:105-111`

**Verification**: logit clamp is 10.0, matching the common range for
categorical policies. `nan_to_num` before clamp handles the "NaN.clamp() is
still NaN" edge case correctly. Single per-policy install via
`_wrap_action_distribution_with_clamp`.

**Severity**: **OK**.

---

## Category G — Behavioural interventions

Files audited:
- `scripts/eval/shortcut.py`
- `scripts/eval/transplant.py`
- `scripts/eval/debug_eval.py`

### G1. Shortcut `per-bar n` mismatch with paper prose — MAJOR (claim wording)

**File/line**: `scripts/eval/shortcut.py:230-237`,
`scripts/paper_figures/make_h3_content_figure.py:68-71`

**Issue**: the paper caption (Fig. `h3_content`, §H3) says:
> "20 Gibson scenes × 10 paired episodes" (= 200 SPLs per bar)

But the aggregation is **scene-averaged mean-of-means**, not a flat 200-
sample average:
- For each (scene, condition), `per_episode_spl` is a length-10 list.
- `all_results[i]["mean_spl"] = np.mean(per_episode_spl)` — per-scene mean.
- `reset_mean_spl = np.mean([r["mean_spl"] for r in all_results])` — mean
  over 20 scene-means, **not** mean over 200 SPLs.

Figure code reads `d["reset_mean_spl"]` and `d["persistent_mean_spl"]`
(same mean-of-means).

These two quantities differ when scenes have different per-scene means
(they do — scene 1 may be easy, scene 20 hard). The bar height is the
20-scene average; the claimed n=200 is true for the underlying pool but
the bar does not equal `np.mean` of those 200.

**Why it matters**: a reviewer who computes error bars expecting n=200 IID
samples will get a different σ than a reviewer using scene-level means
(n=20 with within-scene averaging). Since the paper reports percentage-
drop values (52%, 41%, 21%, 18%, 15%) but no error bars in the shortcut
bar chart, the claim is unfalsified. But any future error-bar addition
must use the correct aggregation level.

**Severity**: **MAJOR** for caption wording; **OK** for directional claim.
Proposed fix: either (a) change caption to "20 scenes, 10 episodes per
scene; bars show mean of per-scene means" or (b) recompute bar as flat
200-sample mean (requires changing aggregator to flatten
`per_episode_spl` across scenes). Option (a) is simpler and more
statistically defensible (avoids pseudo-replication since episodes in
same scene are not IID).

### G2. Shortcut "paired episodes" wording — MINOR (paper vs. code)

**File/line**: `scripts/eval/shortcut.py:167-221`, paper §H3 "paired episodes"

**Observation**: the paper says "second of two paired same-scene/different-
goal episodes" — implying n=200 = 20 scenes × 10 second-episode-pairs.
But code evaluates ALL 10 episodes in each condition (reset + persistent),
computes per-scene mean across all 10, and aggregates. The "second episode"
framing is not literally implemented — every episode-position is
aggregated.

**Why it matters**: the persistent condition's 1st episode is IDENTICAL to
the reset condition's 1st episode (both start with zeroed state; the first
episode is the first, with no prior context to carry). The meaningful
comparison is episodes 2..N within persistent. Currently episode 1 is
pooled into the mean, which **slightly understates** the persistent-vs-
reset gap because the first pair contributes zero gap by construction.

**Severity**: **MINOR**. Effect size: 1/10 of the pair is zero-gap ⇒
bar heights are biased towards equality by ~10%. Doesn't flip any
ordering but shrinks the "persistent advantage/disadvantage" magnitude.
Fix: optionally skip episode 0 in aggregation, or report "2..10" subset;
OR remove "second of two" phrasing in paper and keep current aggregation.

### G3. Shortcut uses `env._episode_iterator = iter(eps)` — OK but fragile

**File/line**: `scripts/eval/shortcut.py:178`

**Observation**: Habitat's private `_episode_iterator` attribute is set
directly. This is a known but non-public API pattern. The same `iter(eps)`
is reassigned for each condition (lines 175, 178), so reset & persistent
each start from `eps[0]`. Works; brittle to Habitat upgrades.

**Severity**: **OK**. Flag in paper supplementary if reproducibility is
important: "We pin habitat-lab commit X".

### G4. Transplant uses `env._current_episode = ep` to pin episode — MINOR to MAJOR (needs manual verification)

**File/line**: `scripts/eval/transplant.py:329-332`, `:344-346`, `:387-389`

**Observation**: the transplant script does:
```python
env._current_episode = ep
env._episode_over = False
obs = env.reset()
```
three times per `ei` iteration (once per condition). The intent is "run
the SAME episode three times" — baseline, self-transplant, cross-transplant.

**Risk**: in Habitat-lab ≥ 0.2.x, `env.reset()` pulls the NEXT episode from
`env._episode_iterator`, not from `env._current_episode`. Setting
`_current_episode` alone may have no effect — the iterator advances
anyway, and each `env.reset()` loads a DIFFERENT episode.

If this bug is live, the three conditions (baseline, self-transplant, cross-
transplant) for iteration `ei` are three DIFFERENT episodes, not the same
one. Baseline = episode 3*ei, self = 3*ei+1, cross = 3*ei+2 (approximately).
The "self-transplant should equal baseline" sanity check then becomes a
check that self-transplant's random episode happens to SPL comparably to
baseline's random episode — which is an average-over-large-N convergence
rather than a per-episode identity.

**Counter-evidence (why this may be working despite the concern)**: the
results show `self_transplant ≈ baseline` within ~0.03 SPL and
`cross_transplant` drops consistently by 0.19–0.21. If episodes were
drifting, self vs baseline might look noisier; but over n=150 they'd
converge to the same mean. So **the aggregate mean would be unaffected
even if episodes drift**. The BUG, if present, only breaks *per-episode
paired-difference* analyses, not the aggregate SPL deltas.

**Severity**: **MAJOR for paired-episode interpretation; MINOR for
reported aggregate SPL numbers**. Proposed verification:
1. Add a one-line assertion after each `env.reset()`:
   `assert env.current_episode.episode_id == ep.episode_id`.
2. If the assertion fails, replace the `_current_episode` pattern with
   the same iterator-reset pattern used in `shortcut.py:178`:
   `env._episode_iterator = iter([ep])` before each condition.

**What would be invalidated if the bug is real**: the per-episode columns
in `results_baseline`, `results_self`, `results_cross` would NOT be
aligned — any analysis that claims "on episode N, self/cross differed by X"
would be wrong. The aggregate transplant SPL drops (0.19–0.21) would
survive because averaging over 150 random episodes gives statistically
similar means.

### G5. Transplant cross-condition uses donor's obs but recipient's policy — OK with caveat

**File/line**: `scripts/eval/transplant.py:280-299`

**Observation**: the shared env is built from `donor_config.habitat`. If
the donor is SIGHTED (uniform/foveated/matched), the env has RGB sensors.
If the donor is BLIND, the env has NO RGB. Blind-donor + sighted-recipient
→ the recipient policy's forward pass is missing RGB obs key.

**What happens in practice**: `WijmansPointNavPolicy.forward` uses
`observation_space` filtering at construction time (line 295-301 in
`wijmans_policy.py`): the visual_encoder is built with only visual keys
in the observation space. If the env doesn't supply RGB, the recipient's
visual_encoder was still constructed from recip_config, which DOES include
RGB. At inference, `observations[RGBSensor.cls_uuid]` would `KeyError`.

**Severity**: **MAJOR for blind-donor + sighted-recipient pairs**.
But the paper's actual pairs (verified from commit 3ce936c: "launch 3 key
pairs") are: `foveated→uniform`, `foveated-learned→foveated`,
`foveated→blind` (blind as recipient, not donor). In these pairs the
donor's config is ALWAYS sighted or equally rich, so the env has enough
sensors for the recipient. **Results not invalidated** for the reported
pairs.

**Proposed fix for robustness**: build env from the UNION of donor+recipient
obs spaces (e.g., always use the sighted config's env, even if donor is
blind). Add a runtime assert that the env's observation space is a
superset of both configs'. Document in the script header.

### G6. Transplant `midpoint=30` default — OK (matches paper)

**File/line**: `scripts/cluster/submit_transplant.sh:30`,
`scripts/eval/transplant.py:239-242`

**Verification**: the submit wrapper sets `MIDPOINT_STEP=30` by default
(line 30). The Python CLI default is `--midpoint-frac 0.5 * max_steps=500
→ 250`, but the SLURM wrapper overrides to 30. The paper §H2 figure
caption says "midpoint sweep {0,15,30,60,90}" — so 30 is one point in the
sweep, not the single default.

Commit `6f9dda5` ("Transplant: reduce default midpoint to 30 steps (was
250)") confirms the change was intentional. Recent commit log suggests
the 150-episode submissions have been running with midpoint=30.

**No off-by-one**: `_run_first_half(env, obs, policy, ..., n_steps=midpoint)`
iterates `for step in range(midpoint)` — executes exactly `midpoint` steps
(0, 1, ..., midpoint-1). Donor takes actions for steps 0..29 inclusive
(30 actions). Recipient takes over at step 30. That matches the paper's
claim of "donor drives first 30 steps".

**Severity**: **OK**.

### G7. Transplant self-transplant path carries `prev_a, mask` — OK (after fix)

**File/line**: `scripts/eval/transplant.py:370-375`, `:428-439`

**Verification**: self-transplant uses `rnn_h.clone()` (identity), then
continues with `prev_a, mask` from the first half. Cross-transplant uses
`donor_rnn_h.clone()` + `donor_prev_a, donor_mask`. Recent commits
`0b689d4` ("Transplant: carry over donor's prev_action + mask at midpoint
(fix NameError)") show this was recently corrected — earlier version had
a NameError because it referenced `prev_a` that wasn't defined in the
cross-transplant branch.

**Potential subtle issue**: the cross-transplant continues with
`donor_prev_a` as the recipient's `prev_action`. This is CORRECT — the
recipient at step 30 should believe "the previous action was action X" to
maintain LSTM state consistency, and action X was the donor's last action.
If the recipient had used its own `prev_a=None/zero`, the LSTM would
incorrectly read "start-of-episode". Current code is right.

**Severity**: **OK**.

### G8. Transplant episode count: 150 per pair — OK (with Habitat dataset caveat)

**File/line**: `scripts/cluster/submit_transplant.sh:29`,
`scripts/eval/transplant.py:317-318`

**Verification**: submit wrapper requests 150; Python caps at
`min(args.episodes, len(all_episodes))`. Gibson train has thousands of
episodes so 150 is reachable.

**Caveat**: `_run_first_half` can terminate early if the donor calls STOP
or finishes the episode within 30 steps. Such episodes are flagged as
failures in the cross-transplant branch (line 414-426) but still counted.
If the donor is very efficient (mean episode ≈ 70–100 steps), many
episodes would end before the 30-step midpoint — especially for
short-path episodes. Inspection of the `_print_progress` shows running
counts but no separation of "full transplant" vs "donor ended early"
counts.

**Severity**: **MINOR**. Fix: add a per-episode flag `transplant_applied`
in the output JSON; report the fraction of episodes where the full
transplant protocol actually executed. Doesn't invalidate the reported
aggregate ∆SPL if the early-termination rate is similar across conditions;
could bias comparisons if one donor-recipient pair has systematically
shorter first halves (e.g., blind donor = less confident → more STOP
before 30, vs foveated-learned donor = very long trajectories → always
reaches 30).

### G9. Both scripts use `deterministic=True` after the `collect.py` fix — OK

**File/line**: `shortcut.py:90`, `transplant.py:81`

**Verification**: both scripts set `deterministic=True` in policy.act()
calls. `collect.py` now defaults to True as well (line 153, after fix in
commit `c81352e`). All three evaluation paths are consistent.

**Recent `collect.py` changes do NOT affect shortcut/transplant flow**:
shortcut.py and transplant.py import only `load_habitat_config`,
`load_policy`, `compute_spl`, `heading_from_quaternion` from
`src.utils.habitat_env`. The `collect.py` changes (new `--deterministic`
flag, masked-sensor injection, gaze hook, occupancy collection) all live
in `scripts/probing/collect.py` and do not touch shared utilities.

**Severity**: **OK**.

### G10. SPL calculation uses `compute_spl(success, path_length, geodesic)` — MINOR (potential off-by-one)

**File/line**: `scripts/eval/shortcut.py:114`, `transplant.py:121`, `:187`

**Observation**: SPL is `success * geodesic / max(path_length, geodesic)`.
Path length is accumulated via `np.linalg.norm(cur_pos - prev_pos)` in
a loop. Two risks:
- If `path_length < geodesic` (agent teleports impossibly), `max()`
  keeps geodesic in the denominator → SPL = success (i.e., 1 if successful).
  Expected behaviour.
- In transplant.py `_run_second_half`, `path_length_so_far` from first half
  is passed through — consistent accumulation.

**Edge case**: cross-transplant's first half is driven by the DONOR,
whose path_length counts as "the shared agent's" path. When recipient
takes over, its path is appended. SPL's numerator uses the episode's
TRUE geodesic (from the episode definition), not donor's geodesic.
This is what SPL-via-transplant should be — the donor+recipient together
form a single navigation attempt, evaluated as one trajectory.

**Severity**: **OK**. Confirmed consistent.

---

## Summary

**No BLOCKERs for Category E (policies).**

**One MAJOR for Category G**: **G4** — the `env._current_episode = ep;
env.reset()` pattern in `transplant.py` may not pin the episode if
Habitat's `env.reset()` ignores `_current_episode` and advances the
iterator. This affects per-episode paired-difference interpretability
but NOT the aggregate SPL delta (0.19–0.21) reported in the paper. Needs
a single-line assertion to verify; easy fix if it's broken.

**One MAJOR for Category G (wording)**: **G1** — shortcut figure's per-bar
`n=200` is a mean-of-means over 20 scenes, not a flat 200-sample mean.
Paper caption should be clarified; no results-level impact.

**MINOR findings** (E1, E2, E3, E5, E8, G2, G5, G8) are doc/wording
issues or secondary confounds that do not invalidate the H1/H2/H3
narrative.

**OK findings** (E4, E6, E7, E9, G3, G6, G7, G9, G10) are verifications
that the code behaves as the paper claims.

### What claims would change if G4 (episode pinning) is broken

- §H2 "cross-transplant drops SPL by $0.19$--$0.21$": **UNAFFECTED** —
  aggregate over 150 random episodes.
- §H2 "self-transplant $\approx$ baseline (protocol noise $\sim 0.03$
  SPL)": **UNAFFECTED** for the same reason — averaging gives the
  correct protocol-noise estimate.
- Transplant-sweep plot (`fig/transplant_sweep.pdf`) showing midpoint-
  dependent SPL: **UNAFFECTED for means**, but **bootstrap CIs computed
  from per-episode pairs would be too narrow** since they'd assume
  pairing that doesn't exist.
- Claim "foveated-learned→foveated is the most incompatible pair at
  EVERY midpoint" (Fig. caption, L214): could shift ordering noise-
  wise at small midpoints if episode-drift adds variance. Aggregate
  means should still order consistently.

### What claims would change if G1 (shortcut N aggregation) is clarified

- §H3 figure 4b percentages ($52\%, 41\%, 21\%, 18\%, 15\%$): current
  values ARE mean-of-means. If recomputed as flat-pool means, numbers
  may shift by a few percentage points but the ordering should be robust.
- "n=200 SPLs per bar": replace with "bar height = mean of 20 scene-
  means; each scene-mean averages 10 per-episode SPLs".

---

## Category B — Probe methodology & Category C — Target definitions & Category D — Hidden-state extraction

Files audited:
- `src/utils/probing.py`
- `scripts/probing/analyze.py`
- `scripts/probing/extended_lag_probe.py`
- `scripts/probing/goal_vector_probe.py`
- `scripts/probing/masked_heading_probe.py`
- `scripts/probing/collect.py`
- `scripts/probing/confidence_probe.py` (spot check)
- Supporting: `src/habitat/wijmans_policy.py`, `src/habitat/wijmans_sensors.py`,
  `src/habitat/foveated_learned_policy.py`, `src/utils/habitat_env.py`,
  `habitat_configs/ddppo_pointnav_blind_gibson.yaml`

**No BLOCKER-class bugs were found.** The `deterministic=False` bug we
previously fixed was the biggest problem; remaining issues are either
defensive-programming nits, cosmetic (sign/naming conventions), or
MAJOR-but-known-to-us caveats that should be disclosed in the paper.

### B1. `probe_1a_per_scene_position` temporal split is step-level, not episode-level — MAJOR

**File/line**: `scripts/probing/analyze.py:118-132`

**Observation**: Within each scene, the code splits the concatenated steps
by index:
```python
h_s, p_s, t_s = H[mask], P[mask], theta[mask]
split = int(len(h_s) * train_frac)
H_tr, H_te = prepare_features(h_s[:split], h_s[split:], pca_dim)
```
This is a **step-split within scene**, with no awareness of episode
boundaries. Because `collect.py` appends steps serially (episode-by-
episode), the first 80% of steps in a scene are mostly the first 80%
of episodes — so the leakage is small in expectation. But if an
episode straddles the 80% boundary, its first few steps end up in
train and its last steps in test. Consecutive steps in the same
episode are **highly correlated** (one `TURN_LEFT/RIGHT` of ~10°
apart, or a `MOVE_FORWARD` of 0.25 m), so the probe can memorize
per-episode position and interpolate across the split.

This is **exactly the same class of confound** as the `deterministic=False`
bug: a probe that looks like it's measuring "generalization across
unseen trajectories" is actually doing "within-trajectory temporal
interpolation".

**Why it matters**: the per-scene position R² is reported as a paper
figure of merit. If those numbers have been inflated by within-
episode leakage, the per-scene/global gap (or the cross-condition
ordering) could be noise. The GLOBAL probe in `probe_1b_global_gps_compass`
uses `episode_split` so it's fine — but the per-scene probe uses a
different code path.

**Proposed fix** (1 of 2):
- **Preferred**: within each scene, gather unique episode IDs, shuffle
  with a seed, take the first 80% for train and the rest for test —
  same `episode_split` logic but scoped to that scene.
- **Minimum**: exclude the straddling episode entirely (drop any episode
  whose indices span both sides of `split`).

**Data invalidation**: the per-scene position R² values reported in
`analyze.py`'s `1a_per_scene_position` field in every `_analysis.json`
file are potentially inflated. Rerunning Phase-1 analysis on existing
.npz files takes ~minutes and does not require re-collection.

**Severity**: **MAJOR**. Per-scene position R² appears in paper Table 1
and the discussion of "scene-specific vs. scene-general memory".
Rerun before freezing numbers.

---

### B2. `probe_1ef_control_and_selectivity` shuffles globally, not within-episode — MINOR

**File/line**: `scripts/probing/analyze.py:217-232`

**Observation**: the Hewitt-Liang control uses `rng.shuffle(gps_tr)`,
a **global** permutation across all training episodes. The user's
audit prompt asks whether this should be within-episode. Standard
H&L uses global shuffles (tests whether the probe has enough capacity
to memorize arbitrary random labels), so this is technically
defensible. **However**, because our episode-level split is adversarial
for any agent that encodes episode identity in H, a global shuffle
will still let the probe achieve slightly-above-chance control R² (it
can match the per-episode mean of the shuffled labels). A within-
episode shuffle would be a stricter control.

**Why it matters**: only impacts the `gps_selectivity` /
`compass_selectivity` metric. Empirically we've been reporting
selectivity ≈ 0 for control probes, so this is robust in practice.
Reviewers may nit-pick the choice.

**Proposed fix**: add a second control that shuffles within-episode
and report both selectivity numbers in Appendix. No existing claim is
invalidated.

**Severity**: **MINOR**.

---

### B3. R² clipping to `[-10, 1]` can create misleading sentinel values — MINOR

**File/line**: `src/utils/probing.py:28`

**Observation**: `fit_probe` clamps R² to [-10, 1]. If the probe
genuinely fails (e.g., target variance near zero on the test set, or
Ridge is forced to extrapolate out-of-distribution), the unclipped
R² can be very large and negative (e.g., −10⁴). After clipping, every
such failure reports **−10** exactly.

Downstream code in `analyze.py:2c_path_history` uses the R² value
directly in summary statistics and plotting. If two different
failure modes (degenerate target, missing features) both report −10,
they become indistinguishable. Specifically, `probe_2c_path_history`
reports a per-lag `reliable` flag keyed on `test_target_var > 0.01`,
but does not flag `r2 ≤ −10` cases as "clipped" — a downstream
reader can't tell whether that lag was catastrophically bad or
genuinely near the floor.

**Why it matters**: low-stakes for paper claims (we report GPS
R²≈0.95 / 0.78, well inside the valid range). Matters for
extended-lag / truncated-episode panels, where some lags hit
degenerate-target regimes and are rendered as "R²=−10" bars.

**Proposed fix**: keep the clip, but also add an `r2_raw` field and
a `clipped` boolean to `fit_probe`'s return dict so downstream can
tell. No data invalidation; reports only read cleaner.

**Severity**: **MINOR**.

---

### B4. `prepare_features` scaler is fit on train, transformed on test — OK

**File/line**: `src/utils/probing.py:108-110`

**Observation verified**: `StandardScaler().fit_transform(H_tr)` and
`scaler.transform(H_te)` are called correctly. No scaler ever sees
combined train+test data. PCA (when `pca_dim > 0`) likewise uses
`fit_transform` on train, `transform` on test. No leakage.

Same pattern is repeated correctly in `probe_2b_cross_heading_generalization`
(analyze.py:319-321).

**Severity**: **OK**.

---

### B5. `episode_split` disjointness and `train_frac=0.8` consistency — OK (with one exception)

**File/line**: `src/utils/probing.py:135-153`

**Observation verified**: `episode_split` partitions `unique_eps` into
first `0.8*N_ep` for train and the rest for test; the returned masks
are disjoint by construction (`train_mask` and `~train_mask`). All
four conditions (blind/uniform/foveated/matched) are probed with
`seed=42` and `train_frac=0.8` by default in `analyze.py`.

**Exception**: `scripts/probing/extended_lag_probe.py:35` and
`scripts/probing/goal_vector_probe.py:60` and
`scripts/probing/masked_heading_probe.py:29` and
`scripts/probing/confidence_probe.py:35` each re-implement their own
`split_by_episode` helper with **different defaults**:
- `extended_lag_probe.py`: `test_frac=0.2`, `seed=0`, `np.random.default_rng`
- `goal_vector_probe.py`: `test_frac=0.2`, `seed=0`
- `masked_heading_probe.py`: `test_frac=0.2`, `seed=0`
- `confidence_probe.py`: `train_frac=0.8`, `seed=0`

All of these effectively use the same 80/20 ratio, but the **seed
differs** (`0` vs `42`). The different seeds mean each probe sees a
different episode partition. For comparisons across probes (e.g.,
extended-lag R² vs analyze.py's 1b R² at lag=0), we're not comparing
on identical test sets.

**Why it matters**: small — with 500 episodes, a seed-0 vs seed-42
split gives ~identical aggregate R². But it's a code-hygiene issue
that reviewers may flag if they look at the repo.

**Proposed fix**: route every probe script through
`src.utils.probing.episode_split` and a single shared seed. Purely
editorial, no result changes.

**Severity**: **OK/MINOR** (cosmetic).

---

### B6. `fit_probe_cv` episode-level CV does not leak timesteps across folds — OK

**File/line**: `src/utils/probing.py:33-94`

**Observation verified**: The function pre-shuffles `unique_eps`, maps
each step to an episode-position index, then uses `KFold(shuffle=False)`
over `unique_eps`. The train/test masks are built by checking
membership in the fold-assigned episode-position sets — so every
timestep of a given episode ends up in exactly one fold. No timestep
leakage.

Small nit: `KFold(n_splits, shuffle=False)` then splits into
contiguous blocks of (already-shuffled) episodes. With 500 episodes
and 5 folds, each fold has ~100 episodes. Fold-to-fold variance is
therefore largely a measure of episode-partition variance, not of
probe instability. That's the intended semantics. Good.

**Severity**: **OK**.

---

### B7. `probe_2c_path_history` / `build_lag_pairs` respects episode boundaries — OK

**File/line**: `scripts/probing/analyze.py:388-402` and
`scripts/probing/extended_lag_probe.py:45-56`

**Observation verified**: both lag-k builders loop over episodes and
pair `H[t]` with `position[t-k]` **only when `t-k` is in the same
episode** (in `analyze.py`, via `step_in_ep[local_i] >= k` and a
within-episode search; in `extended_lag_probe.py` via
`idx[i-lag]` where `idx = np.where(ep_ids == e)[0]` returns only
this-episode step-indices in temporal order because `collect.py`
appends monotonically per episode).

One invariant the `extended_lag_probe.py` version relies on that
isn't obviously documented: `np.where(ep_ids == e)[0]` returning
indices in *temporal* order. This is true only because
`collect.py` appends serially per episode. If the .npz is ever
re-shuffled in post-processing, the lag probe would silently
produce garbage. Adding an explicit
`idx = idx[np.argsort(step_in_ep[idx])]` sort by step would be
defensive.

**Severity**: **OK** (with a one-line defensive-programming
suggestion).

---

### C1. GPS and compass targets ARE episodic (start-of-episode frame) — OK

**File/line**: `habitat_configs/ddppo_pointnav_*.yaml` under
`lab_sensors: [gps_sensor, compass_sensor]`.

**Observation verified**: `gps_sensor` and `compass_sensor` are the
default Habitat names for `EpisodicGPSSensor` and
`EpisodicCompassSensor` respectively. `src/habitat/wijmans_policy.py`
imports these classes by their exact symbol names and uses their
`cls_uuid` ("episodic_gps" and "episodic_compass") to read the
observation dict. Every training and probing config in the repo uses
these sensors — never `PointGoalWithGPSAndCompassSensor` (which
combines them into a goal-in-start-frame vector).

So `obs["gps"]` is 2-D (start-frame x, z) and `obs["compass"]` is
1-D radians (start-frame yaw), both reset to (0, 0) and 0 at
step 0 of every episode. Downstream probes treat them as episodic,
consistent with the paper's "episodic GPS R²" nomenclature.

**Severity**: **OK**.

---

### C2. `angular_mae` inverts sin/cos correctly with wrap-around — OK

**File/line**: `src/utils/probing.py:119-132`

**Observation verified**: the function uses
`arctan2(pred[:,0], pred[:,1])` — i.e., first column treated as sin,
second as cos. Caller sites in `analyze.py` produce targets with
`np.hstack([np.sin(comp), np.cos(comp)])` (or `np.stack([sin, cos],
axis=1)`) — sin at col 0, cos at col 1. The convention is consistent.
Wrap-around is handled by
`arctan2(sin(pred - true), cos(pred - true))` which maps any
angular difference to the principal range (−π, π]. The MAE is then
taken in degrees. Correct.

**Severity**: **OK**.

---

### C3. `goal_vector_probe.goal_direction_mae_deg` does NOT wrap angular difference — MINOR

**File/line**: `scripts/probing/goal_vector_probe.py:112-114`

**Observation**: the MAE is computed as
```python
mae_dir_deg = float(np.rad2deg(
    np.abs(np.arctan2(dir_pred[:, 0], dir_pred[:, 1]) - direction[test_mask]).mean()
))
```
There is no `arctan2(sin(d), cos(d))` wrap — so if the prediction is
near +π and the target is near −π, the reported error is ≈ 2π
(360°) rather than ≈ 0. The `r2_dir` (reported alongside) on the
2-D sin/cos encoding is unaffected, but the degree-number in
`goal_direction_mae_deg` can be inflated at the ±π boundary.

Same bug in `scripts/probing/masked_heading_probe.py:90`:
```python
mae_rad = float(np.abs(np.arctan2(pred_c[:, 0], pred_c[:, 1]) - ep_compass[te]).mean())
```
No wrap. Same ±π-boundary overestimate.

**Why it matters**: low — these MAE numbers are rarely cited
in paper. `r2_dir` is the reported primary metric. But if the paper
ever quotes a "goal-direction MAE in degrees" from
`goal_vector_probe.py`, the number is worse than it should be for
uniform-yawed distributions near ±π.

**Proposed fix**: wrap the difference:
```python
diff = np.arctan2(np.sin(pred - true), np.cos(pred - true))
mae = np.rad2deg(np.abs(diff).mean())
```
This is the exact pattern already in `src/utils/probing.angular_mae`.
Both scripts should call that shared helper.

**Data invalidation**: no — only affects a secondary MAE number if it's
ever cited. R²s are correct.

**Severity**: **MINOR**.

---

### C4. `ego_goal_vector` rotation sign — convention-only, R² invariant — OK (with caveat)

**File/line**: `scripts/probing/goal_vector_probe.py:41-57`

**Observation**: the code rotates world deltas by `-heading` with the
mapping `forward = cos_h*(-dz) - sin_h*dx`,
`lateral = sin_h*(-dz) + cos_h*dx`. Substituting various headings
(0, π/2, π) shows that `forward` is correct at θ∈{0, π} but may
pick up a sign flip at θ=π/2 relative to the right-hand-rule
"forward-of-agent" definition, depending on how one defines Habitat's
heading offset (the `heading_from_quaternion` helper returns π for
identity quaternion, not 0, so the whole pipeline has a π-offset
convention).

**Why it matters**: **only for interpretation** of the per-dim
`r2_forward` and `r2_lateral` fields. Ridge regression is invariant
under sign flips of the target (it will learn the right weights
regardless), so **combined** `goal_vector_r2` and
`goal_direction_r2` are correct. But if the paper discusses
"forward-component R² vs lateral-component R²" the labels may be
swapped (what you call "forward" might be "backward" or
partially-forward-partially-lateral).

**Proposed fix**: add a sanity-check block that, on a few random
(position, goal, heading) triples from one episode, prints the
expected and computed (forward, lateral) values. If labels are
swapped, just rename the dict keys in the output JSON. No
re-training or re-collection needed.

**Severity**: **OK/MINOR** (interpretation-only).

---

### C5. `masked_heading_probe.py` reconstructs episodic compass from world-frame headings — OK

**File/line**: `scripts/probing/masked_heading_probe.py:67-82`

**Observation**: this script explicitly addresses a known problem with
the masked-compass ablation: if `obs["compass"]` is zeroed during
collection, regressing `H → compass` gives a trivially perfect R².
The script instead re-derives the episodic compass from
`headings_full` (world-frame) by subtracting each episode's
step-0 heading and wrapping. It also re-derives episodic GPS the
same way. This is the **correct** decoupling and it matches how the
sensor would have behaved absent masking.

No ground-truth labels are read from `obs[...]` in this path, so
masking does not contaminate the probe target.

**Severity**: **OK**.

---

### D1. LSTM state layout `(h0, c0, h1, c1, h2, c2)` and `0::2 / 1::2` slicing is correct — OK

**File/line**: `scripts/probing/collect.py:286-297`

**Observation verified**: The comment block at L286-288 matches the
interleaved layout documented at `foveated_learned_policy.py:143-148`
("Habitat-baselines' rollout buffer stores the LSTM hidden state with
shape (num_envs, num_recurrent_layers * 2, hidden_size) - env-major
... (h0, c0, h1, c1, ...) ordering"). For `num_recurrent_layers=3`
(our config value, per the YAML), `policy.net.num_recurrent_layers`
returns `2 * 3 = 6` (via `state_encoder.num_recurrent_layers`,
which in habitat-baselines is `num_layers * 2` for LSTM). The
collected `new_rnn_hidden` has shape `(1, 6, 512)`, and:
- `new_rnn_hidden[0, 0::2]` picks indices [0, 2, 4] → h0, h1, h2
- `new_rnn_hidden[0, 1::2]` picks indices [1, 3, 5] → c0, c1, c2
- `h_all[-1]` = h2 = top (deepest) layer

This matches the paper's "top-LSTM-layer hidden state" definition.

**Severity**: **OK**.

---

### D2. `top_h` = top layer (deepest) by last-index convention — OK

**File/line**: `scripts/probing/collect.py:293`

**Observation verified**: In habitat-baselines' LSTMStateEncoder, the
first pair (indices 0, 1) is the **first** LSTM layer (closest to
input), and the last pair (indices 4, 5 for 3-layer) is the
**last / top / deepest** layer. `h_all[-1]` extracts layer 2 (the
last), consistent with PyTorch's `nn.LSTM` which stacks layers as
`(num_layers, batch, hidden)` with layer 0 = first. ✓

**Severity**: **OK**.

---

### D3. `rnn_hidden_states` contains only LSTM h/c, nothing else — OK

**File/line**: `scripts/probing/collect.py:289`

**Observation verified**: `action_data.rnn_hidden_states` in
habitat-baselines is specifically the recurrent state tensor for the
policy's RNN backbone. It does not contain auxiliary state (gaze
decoder outputs, value-head state, etc.). In
`foveated_learned_policy.py`, the gaze decoder reads a **detached
copy** of `rnn_hidden_states` (as input) but does not write its
output back into it — gaze is stored in `all_gaze_positions` via
a separate forward hook (collect.py:229-236). ✓

**Severity**: **OK**.

---

### D4. Hidden-state / pose temporal alignment — OK

**File/line**: `scripts/probing/collect.py:262-344`

**Observation verified**: the loop records:
- `h_t = new_rnn_hidden` = LSTM state AFTER processing observation `obs_t`
- `pos_t` = agent state AT TIME t (from `env.sim.get_agent_state()`
  called BEFORE `env.step(action_t)`)
- `gps_t`, `compass_t` = sensor readouts AT TIME t (from `obs` before
  it's overwritten by `env.step`)

So `h_t` encodes what the agent has observed up to and including
step t, and the ground-truth pose and sensor readouts are aligned
to step t. This is the canonical alignment. The probe `H → gps_t`
is therefore "does the hidden state at step t encode the sensor
readout that was just given to the policy at step t?". That's a
valid question — the agent could choose to discard GPS input from
its hidden state, and blind/matched show that sighted agents with
enough visual info do.

**Severity**: **OK**.

---

### D5. Lag-k via `step_in_episode`: relies on `step_in_episode` being injective per episode — OK

**File/line**: `scripts/probing/analyze.py:398-402`

**Observation verified**: `local_target = np.where(ep_steps == target_step)[0]`
assumes `step_in_episode` values are unique within each episode.
`collect.py` sets `all_step_in_episode.append(step)` where `step` is
monotonically incremented inside the per-episode loop, so this is
guaranteed by construction. If a future refactor ever merges two
episodes under the same `episode_id` (not currently possible),
the probe would silently pick an arbitrary one.

**Severity**: **OK** (defensive-programming aside).

---

### Summary of proposed code changes

The one change we'd recommend before the paper is finalized is
**B1** — fix `probe_1a_per_scene_position`'s temporal split to be
episode-aware. This can be done in analyze.py alone and re-run
against existing .npz files (no re-collection).

All other findings are either:
- convention/interpretation only (C3, C4, D-series are OK),
- cosmetic/editorial (B5 seed unification, B3 clipping sentinel),
- paper-caveat-worthy but not result-breaking (B2).

### Summary of claim impact

- **Paper's core finding (blind GPS R²=0.95, matched R²=0.78, RGB at
  chance)**: comes from `probe_1b_global_gps_compass`, which uses
  `episode_split` (no per-scene step-split leakage). **UNAFFECTED** by
  B1.
- **Per-scene position R² numbers (if quoted anywhere)**: potentially
  inflated by B1. Re-run Phase-1 after the fix.
- **Goal-vector and masked-heading probe R²s**: correct.
- **Extended-lag decay curves**: correct lag pairing, episode-
  bounded, uses ego-frame GPS target (intentionally — comment in
  file explains why).
- **Hidden-state extraction**: no layout bug, no off-by-one, no
  auxiliary state contamination.

No BLOCKER found. B1 is the only MAJOR; all others are MINOR or OK.

---
