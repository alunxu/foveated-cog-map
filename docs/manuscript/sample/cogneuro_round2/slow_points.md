# Sussillo–Barak slow-point analysis

## Reference paper

Sussillo, D. & Barak, O. (2013). *Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks.* **Neural Computation** 25(3): 626-649.

Replications / extensions:
- Mante, Sussillo, Shenoy, Newsome (2013) *Nature* — context-dependent computation in PFC.
- Driscoll, Shenoy, Sussillo (2024) *Nat Neurosci* — flexible multitask computation across populations.
- Maheswaranathan, Williams, Golub, Ganguli, Sussillo (2019) *NeurIPS* — universality of dynamics across RNN architectures.

## Original cogneuro use

Find approximate fixed points and slow points of trained RNNs by minimising
\(q(h) = \tfrac{1}{2}\|F(h, u_{\text{const}}) - h\|_2^2\) via gradient descent in `h`-space, then linearise the dynamics via the Jacobian \(J = \partial F/\partial h\) at each slow point. Eigenvalues of \(J\) reveal whether the slow point is a stable attractor, a line attractor, a plane attractor, or a saddle. The collection of slow points and their stable manifolds = a **dynamical-systems portrait** of what the RNN computes.

In the navigation literature this has been used to show that grid-cell-like networks form low-dimensional toroidal attractors (Cueva et al. 2018, 2020); in PFC it shows that decision integration runs along an unstable direction of a saddle (Mante et al. 2013).

## DL analogue (our setting)

We apply slow-point search **post-hoc to our frozen LSTM**, treating the constant-input dynamics as

\[ (h_{t+1}, c_{t+1}) = \text{LSTM}((h_t, c_t), u^{\star}) \]

with `u^*` set to the per-condition **mean encoder output** (so we are asking: "in the absence of new sensory information, where does the recurrent state want to go?"). The "no-input" portrait is exactly what one would compute for a hippocampal attractor model — a path-integration network's intrinsic dynamics in the absence of self-motion cues.

## Hypothesis (capacity-allocation prediction)

Capacity allocation predicts that **blind / coarse** agents — the ones that integrate self-motion over many steps — should have a **lower-dimensional, slower** intrinsic manifold (a soft line / plane attractor along the integrated-position direction) than **foveated / uniform** agents, which can re-anchor to vision and need not maintain a long-range integrator.

Concrete predictions (pre-registered):

1. **# of distinct slow points** (clustered with eps tuned by silhouette): **blind ≤ coarse ≤ foveated ≈ uniform** (sighted agents have richer multi-attractor structure; blind has a quasi-continuous integrator).
2. **Spectral radius of Jacobian at slow points**: **blind > coarse > sighted**. Eigenvalues nearer to 1 in modulus indicate slower integration — a path-integrator signature.
3. **Dimensionality of the slow manifold** (rank of stable subspace): **larger for sighted** (richer state-space, more functional modes); **smaller for blind** (one integrator direction dominates).
4. **Slow-manifold curvature in `h_t`-space**: should align with the linear-position direction in blind / coarse and with non-linear position directions in sighted (matching the format-shift result in §3.4).

If predictions 1–3 hold *and* 4 aligns with the format axis, this is the strongest single piece of evidence that capacity allocation is implemented as a *dynamical* tradeoff, not just a representational one.

## Pseudocode

```python
import torch
from torch.optim import Adam
from sklearn.cluster import DBSCAN

def find_slow_points(lstm, u_const, n_init=2048, n_iters=5000, lr=1e-2, q_thresh=1e-5):
    """Return slow points (h*, c*) and Jacobian at each."""
    # init from cached h_t — biases search toward visited states
    h0_idx = np.random.choice(H_cache.shape[0], n_init, replace=False)
    H = torch.tensor(H_cache[h0_idx], requires_grad=True)
    C = torch.tensor(C_cache[h0_idx], requires_grad=True)
    opt = Adam([H, C], lr=lr)
    for _ in range(n_iters):
        opt.zero_grad()
        H_next, C_next = lstm_step(lstm, H, C, u_const)
        q = 0.5 * ((H_next - H) ** 2 + (C_next - C) ** 2).sum(-1)
        q.mean().backward()
        opt.step()
    # filter: only points where q < threshold
    mask = q.detach().numpy() < q_thresh
    H_star, C_star = H[mask].detach(), C[mask].detach()
    # cluster
    labels = DBSCAN(eps=tune_via_silhouette(H_star)).fit_predict(H_star)
    centroids = [H_star[labels == k].mean(0) for k in set(labels) if k != -1]
    # Jacobian at each centroid (forward-mode autodiff)
    Js = [torch.autograd.functional.jacobian(
              lambda h: lstm_step(lstm, h.unsqueeze(0), c, u_const)[0].squeeze(0),
              cent) for cent, c in zip(centroids, ...)]
    return centroids, Js

# Run per condition:
for cond in ['blind', 'coarse', 'foveated', 'uniform', 'foveated_logpolar']:
    cents, Js = find_slow_points(load_lstm(cond), mean_encoder_out(cond))
    n_slow_points[cond] = len(cents)
    eigvals[cond] = [np.linalg.eigvals(J) for J in Js]
    slow_manifold_dim[cond] = estimate_rank(np.stack(cents))
```

## Runtime estimate

- 2048 init points × 5000 iters × Jacobian compute = **~30 min on a single 3090** per condition.
- 5 conditions × 30 min = **~2.5 GPU-hours**, plus 30 min for clustering / plotting / pre-reg writeup.
- **Critical path bottleneck**: requires LSTM weights loaded from RCP checkpoint — not pure-cache.

## Success criteria

**Strong signal**: predictions 1, 2, 3 all directionally correct; effect size > 1.5× between blind and uniform on at least 2 of 3.

**Modest signal**: 1 of 3 directionally correct, others non-significant — supplementary figure, no main-text claim.

**Failure mode**: slow-point count is comparable across conditions OR Jacobian spectrum is HP-sensitive (n_iters, lr, init distribution). In that case relegate to supplement and *do not* derive narrative.

## Fit to backbone

Slow points are the **dynamics axis** of capacity allocation. The format axis tells us *where* position information lives; slow-point structure tells us *how* the recurrent state moves through that subspace. If they agree, capacity allocation is real at both representational and dynamical levels — that's the "format shift is a dynamical tradeoff" claim.

## Pre-registered prediction

> Sighted (foveated, uniform) LSTMs will have ≥ 1.5× more distinct slow-point clusters than blind/coarse LSTMs, while blind LSTMs will have a higher mean Jacobian spectral radius (closer to 1), indicating a slower / line-attractor-like integration regime. The slow-manifold's principal direction will linearly align with the GPS-position decoder direction in blind/coarse but not in foveated/uniform.

If both directional predictions land, the paper has established a dynamical-systems signature of capacity allocation. If only the spectral-radius prediction lands, we still have a slow-integrator vs fast-updater story (weaker but defensible). If neither lands, the method enters the supplement with the honest report.
