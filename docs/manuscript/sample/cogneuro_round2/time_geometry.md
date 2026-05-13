# Cueva neural geometry of time

## Reference paper

Cueva, C. J., Saez, A., Marcos, E., Genovesio, A., Jazayeri, M., Romo, R., Salzman, C. D., Shadlen, M. N. & Fusi, S. (2020). *Low-dimensional dynamics for working memory and time encoding.* **PNAS** 117, 23021-23032.

Replications:
- Bhalla (2019) *Front. Neural Circuits* — temporal codes in hippocampus.
- Eichenbaum (2014) **Nat Rev Neurosci** — time cells review.
- Tsao et al. (2018) *Nature* — entorhinal time-coding.
- Cueva, Wei (2018) *ICLR* — emergence of grid cells alongside time codes.

## Original cogneuro use

In Cueva 2020, neural population activity in monkey PFC during a working-memory task lies on a **low-dimensional curved manifold** parameterised by time-since-stimulus and the value of the held stimulus. The geometry separates:

- **Time-on-trajectory** (time since the start of the current trajectory segment).
- **Time-in-episode** (absolute time within the trial / episode).

These two time variables can collapse onto the same manifold direction (in pure time-cell systems) or live in orthogonal directions (in mixed time × content systems). The geometry is a strong signature of how time is multiplexed with content.

## DL analogue (our setting)

We have `step_in_episode` (= time-in-episode) and can compute "time-since-last-turn" or "time-on-current-leg" (= time-on-trajectory) from the action sequence. For each condition:

1. Project `h_t` onto a 3-D subspace via PCA.
2. Colour each point by `step_in_episode` and by `time_on_trajectory`.
3. Quantify the **angle** between the time-in-episode gradient direction and the time-on-trajectory gradient direction.

## Hypothesis (capacity-allocation prediction)

1. **Blind agents**: time-in-episode and time-on-trajectory are nearly aligned — the agent has only its self-motion + step counter, so both are encoded along the same integrator direction.
2. **Sighted agents**: time-on-trajectory and time-in-episode are more orthogonal — sighted agents can distinguish "time elapsed" from "distance moved" because vision provides external landmarks.
3. **Curvature of the time manifold**: blind has a near-straight 1-D trajectory in `h_t`-space (path-integrator); sighted has a more curved manifold (richer state).

## Pseudocode

```python
def time_geometry(H, step_in_episode, time_on_trajectory, n_components=3):
    pca = PCA(n_components=n_components).fit(H)
    Z = pca.transform(H)
    # gradient direction for each time variable
    grad_episode = fit_linear(Z, step_in_episode)  # 3-vec
    grad_traj = fit_linear(Z, time_on_trajectory)  # 3-vec
    angle = np.degrees(np.arccos(
        np.clip(np.dot(grad_episode, grad_traj) /
                (np.linalg.norm(grad_episode) * np.linalg.norm(grad_traj)), -1, 1)
    ))
    # curvature: deviation of trajectory from straight-line
    sorted_by_t = Z[np.argsort(step_in_episode)]
    chord = sorted_by_t[-1] - sorted_by_t[0]
    arclen = np.linalg.norm(np.diff(sorted_by_t, axis=0), axis=1).sum()
    curvature_proxy = arclen / np.linalg.norm(chord)
    return angle, curvature_proxy

# Per condition:
for cond in conditions:
    cache = load_cache(cond)
    t_traj = compute_time_on_trajectory(cache['action'], cache['episode_id'])
    angle, curvature = time_geometry(cache['h_t'], cache['step_in_episode'], t_traj)
    results[cond] = (angle, curvature)
```

## Runtime estimate

- PCA + linear-fit: ~5 min/condition.
- 5 conditions = **~30 min**.
- Plus per-condition figure: 1.5 hours total.

## Success criteria

**Strong signal**: angle increases monotonically with bandwidth; blind angle < 30°, uniform angle > 70°.

**Modest signal**: monotone trend but small range (30–60°). Supplementary.

**Failure mode**: time variables are too correlated to disentangle (which is in fact the prediction for blind — but if it's true for *all* conditions, the test is uninformative). Mitigation: pre-reg requires uniform/foveated angles to be > 50° as the discriminating bar.

## Fit to backbone

Time geometry is a **format axis with explicit time-multiplexing**. Round 1's intrinsic timescale gave us a scalar per condition; this gives us the *geometric structure* of how time is multiplexed with content, addressing the question: "is time multiplexed with content the same way across conditions?"

## Pre-registered prediction

> The angle between the time-in-episode gradient and time-on-trajectory gradient in the top-3 PCA components of `h_t` will be monotone increasing across {blind, coarse, foveated_logpolar, foveated, uniform}, with blind < 30° and uniform > 60°. The trajectory-curvature proxy (arclength / chord) will be monotone increasing along the same axis.

If predictions land, the format axis acquires a temporal-geometric refinement: not just *where* position lives but *how time is multiplexed* with it.
