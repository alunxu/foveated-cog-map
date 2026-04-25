# Experiment: foveation–normaliser invariance check (F2)

## Intention

The default `FoveatedWijmansPolicy` disables `normalize_visual_inputs`
(the `RunningMeanAndVar` running input normaliser) because the
in-place buffer mutation in that module conflicts with autograd along
the gaze decoder's gradient path.  Foveated-fixed has no gaze
decoder, so the conflict does not actually arise — but we historically
left the flag at False to keep all foveated variants consistent.

This is a confound when comparing fov-fix (no normaliser) to uniform /
matched-compute (normaliser on) on representational metrics. If the
normaliser hides spatial information from the LSTM, our paper's
"rich-encoder pass-through" finding for fov-fix could be partly an
implementation artefact rather than a property of foveation.

F2 trains a fov-fix variant with the normaliser ENABLED and otherwise
identical settings, to test invariance.

## What's being trained where

Single retrain on **friend's H100 cluster**.

| Run | Config | Cluster |
|-----|--------|---------|
| `foveated_normaliser_gibson` | `pointnav/ddppo_pointnav_foveated_normaliser_gibson` | **friend** |

~12-18h on H100/H200 for 250M frames.

## Prerequisite

The new policy is in `src/habitat/foveated_normalised_policy.py`.
Registered in `src/habitat/__init__.py`.

```bash
cd ~/cs503-project && git pull
grep "FoveatedNormalisedWijmansPolicy" src/habitat/__init__.py
# expect 1 line
python3 tests/test_torch_foveation.py
# expect "All tests passed."
```

## Submit

```bash
sbatch --gres=gpu:1 --time=24:00:00 \
    scripts/cluster/submit_train.sh \
    pointnav/ddppo_pointnav_foveated_normaliser_gibson
```

## Probe

```bash
sbatch scripts/cluster/submit_probe_deterministic.sh \
    pointnav/ddppo_pointnav_foveated_normaliser_gibson \
    data/checkpoints/foveated_normaliser_gibson/ckpt.49.pth 500
```

Output:
- `probing_data/foveated_normaliser_gibson_det.npz`
- `probing_results/foveated_normaliser_gibson_det_analysis.json`

## Expected outcome

Compare top-layer GPS R² to the existing `foveated_gibson` (no
normaliser):

- **GPS R² ≈ 0** (matches current fov-fix at R²=0.06): the
  normaliser is not a confound. The paper's "rich-encoder
  pass-through" finding for fov-fix is robust. Drop this confound from
  the limitations list.

- **GPS R² substantially > 0** (e.g., closer to matched-compute's
  0.78): the normaliser is hiding spatial information. We must
  re-run all rich-encoder vs bottleneck comparisons with the
  normaliser flipped consistently across conditions — this is a
  fundamental confound.

## Integration with paper

- If invariant: fold the result into a one-line "we verified the
  normaliser disabling is not a confound" sentence in §3.2 or §5.5
  limitations.
- If a confound: reframe the H1 finding under the consistent
  normaliser setting. Likely affects the rich-encoder comparison
  numbers but should leave bottleneck conditions (blind, matched)
  unchanged.

## Cluster cost summary

1 job × 12-18h on H100/H200 = ~12-18 GPU-hours.
