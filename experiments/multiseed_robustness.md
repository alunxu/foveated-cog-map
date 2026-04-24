# Experiment: Multi-seed robustness (error bars on main claim)

## Intention

Our H1 result (blind / matched $\to$ LSTM spatial encoding; uniform /
foveated / fov-learned $\to$ pass-through) currently rests on a single
training seed per condition. Reviewer concern: is the monotonic
ordering reproducible, or an artefact of one optimisation trajectory?

We need 2–3 additional seeds per condition to quote the pattern with
statistical error bars.

## Plan

Target: 3 seeds per condition (seed 0 already done; need 2 more each).
**Some seeds are already training on Izar** — this experiment covers
only the remaining gap.

| Condition | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Gap to fill here |
|-----------|--------|--------|--------|--------|------------------|
| Blind | done | — | — | — | **s1 + s2** |
| Uniform | done | — | Izar ~24h elapsed | — | **s3** |
| Foveated | done | — | Izar ~23h elapsed | — | **s3** |
| Foveated-learned | done | Izar ~22h | — | Izar ~23h | — (covered) |
| Matched-compute | done | — | — | — | **s1 + s2** |

**Gap = 6 new trainings** (blind × 2, uniform × 1, foveated × 1,
matched × 2).

## Submit

```bash
# Gap fills — skip the seeds Izar is already running.
sbatch scripts/cluster/submit_train_seeded.sh pointnav/ddppo_pointnav_blind_gibson 1
sbatch scripts/cluster/submit_train_seeded.sh pointnav/ddppo_pointnav_blind_gibson 2
sbatch scripts/cluster/submit_train_seeded.sh pointnav/ddppo_pointnav_uniform_gibson 3
sbatch scripts/cluster/submit_train_seeded.sh pointnav/ddppo_pointnav_foveated_gibson 3
sbatch scripts/cluster/submit_train_seeded.sh pointnav/ddppo_pointnav_matched_gibson 1
sbatch scripts/cluster/submit_train_seeded.sh pointnav/ddppo_pointnav_matched_gibson 2
```

Each run ~10–15h on H100/H200 for 250M frames. With 6-way
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
