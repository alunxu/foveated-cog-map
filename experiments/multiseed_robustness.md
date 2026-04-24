# Experiment: Multi-seed robustness (error bars on main claim)

## Intention

Our H1 result (blind / matched $\to$ LSTM spatial encoding; uniform /
foveated / fov-learned $\to$ pass-through) currently rests on a single
training seed per condition. Reviewer concern: is the monotonic
ordering reproducible, or an artefact of one optimisation trajectory?

We need 2–3 additional seeds per condition to quote the pattern with
statistical error bars.

## Plan

10 new training runs: 5 conditions × 2 additional seeds (the existing
single-seed numbers serve as seed 0).

| Condition | Seeds still needed |
|-----------|-------------------|
| Blind | 2 |
| Uniform | 2 (seed 2 already training on Izar; need seed 3) |
| Foveated | 2 (seed 2 already training; need seed 3) |
| Foveated-learned | 2 (seed 1, 3 already training) |
| Matched-compute | 2 |

## Submit

```bash
for seed in 1 2; do
    for cfg in blind uniform foveated foveated_learned matched; do
        sbatch --gres=gpu:1 --time=24:00:00 \
            scripts/cluster/submit_train_seeded.sh \
            pointnav/ddppo_pointnav_${cfg}_gibson ${seed}
    done
done
```

Each run ~10–15h on H100/H200 for 250M frames. With 10-way
parallelism → all done in ~15h. If fewer GPUs available, serialise.

## Probe

Each seeded checkpoint goes through the standard det-probe pipeline:

```bash
for seed in 1 2; do
    for cfg in blind uniform foveated foveated_learned matched; do
        sbatch scripts/cluster/submit_probe_deterministic.sh \
            pointnav/ddppo_pointnav_${cfg}_gibson \
            /path/to/${cfg}_gibson_seed${seed}/ckpt.49.pth 500
    done
done
```

Output: `probing_data/<cond>_gibson_seed<k>_det.npz` +
        `probing_results/<cond>_gibson_seed<k>_det_analysis.json`

## Expected outcome

The hero figure's 5 bars become bars with wider error bars that
combine (a) 5-fold CV and (b) cross-seed variability. Prediction: the
ordering survives (blind $\gg$ matched $\gg$ others) with overlap
only within the "pass-through" trio.

## Integration

Analysis combines seeds within condition (mean of means, std of means)
and reports the ordering as statistically significant. Updates Table 1
and the hero bar chart.
