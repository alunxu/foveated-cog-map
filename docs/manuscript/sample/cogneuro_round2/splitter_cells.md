# Splitter / journey-coding cells

## Reference paper

Wood, E. R., Dudchenko, P. A., Robitsek, R. J. & Eichenbaum, H. (2000). *Hippocampal neurons encode information about different types of memory episodes occurring in the same location.* **Neuron** 27, 623-633.

Replications / extensions:
- Frank, Brown, Wilson (2000) **Neuron** — prospective and retrospective coding in CA1 / EC.
- Ferbinteanu & Shapiro (2003) *Neuron* — prospective coding in dorsal hippocampus.
- Smith & Mizumori (2006) *Hippocampus* — task-dependent splitter cells.
- Eichenbaum (2014) **Nat Rev Neurosci** — review.
- Recent: Stoianov et al. (2018), Behrens et al. (2018) **Neuron** TEM — schema-level analogue.

## Original cogneuro use

In a T-maze (or similar) where the same physical location is visited en route to different goals, hippocampal place cells often show **journey-dependent firing**: cell `i` fires at location `(x, y)` only on left-turn trials, or only on trials originating from a particular start arm. These "splitter cells" or "journey cells" encode **trajectory identity** in addition to / instead of pure spatial position. Their existence is one of the cleanest demonstrations that hippocampal coding is not pure-position but *episode-conditioned position*.

## DL analogue (our setting)

For each unit `i` in `h_t`, ask: does its firing rate at a given `agent_pos` depend on the trajectory the agent took to get there?

Concretely:
1. Discretise `agent_pos` into spatial bins (e.g. 0.5 m grid).
2. For each bin `b` and each unit `i`, compute the firing rate as a function of trajectory features: prior 5-step direction, prior goal (`goal_vec`), prior episode-conditioned context.
3. Fit a 2-way ANOVA per (unit, bin): `h_i ~ traj_feature + bin + traj × bin`. A unit is a "splitter" if it has a significant `traj × bin` interaction.

Capacity allocation prediction: **blind / coarse** agents must integrate trajectory history to know where they are, so trajectory information should be more deeply embedded into per-unit firing → **higher splitter fraction**. **Foveated / uniform** agents can re-anchor via vision → **lower splitter fraction**, units more place-pure.

## Hypothesis

1. The fraction of units showing significant trajectory × position interaction (FDR-corrected p<0.01) is monotonically **decreasing** with encoder bandwidth: blind > coarse > sighted.
2. The strength of the interaction (η²-partial) is also monotonically decreasing.
3. Splitter units in the blind agent project preferentially onto the GPS-position decoder direction (i.e., trajectory information is *integrated into* the position code).
4. In sighted agents, splitter units (the few that exist) project onto encoder-derived directions (visual landmark identity).

## Pseudocode

```python
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

def splitter_fraction(H, pos, prev_action, episode_id, n_bins=10):
    pos_bin = np.digitize(pos[:, 0], np.linspace(pos[:, 0].min(), pos[:, 0].max(), n_bins))
    traj_feat = prev_action  # discrete; could also use rolling-mean direction
    n_units = H.shape[1]
    pvals = np.zeros(n_units)
    eta2 = np.zeros(n_units)
    for u in range(n_units):
        # 2-way ANOVA of h_u on bin × traj_feat
        groups = {}
        for b, t, val in zip(pos_bin, traj_feat, H[:, u]):
            groups.setdefault((b, t), []).append(val)
        # main + interaction via OLS
        F, p, eta = two_way_anova(H[:, u], pos_bin, traj_feat)
        pvals[u] = p['interaction']
        eta2[u] = eta['interaction']
    rej, _, _, _ = multipletests(pvals, alpha=0.01, method='fdr_bh')
    return rej.mean(), eta2.mean()

# Per condition:
for cond in conditions:
    cache = load_cache(cond)
    frac, eta = splitter_fraction(
        cache['h_t'], cache['agent_pos'],
        cache['action'], cache['episode_id']
    )
    results[cond] = (frac, eta)
```

## Runtime estimate

- 512 units × 5 conditions × ANOVA per unit: ~15 min/condition on CPU. Can vectorise with `pingouin` or `statsmodels` ols.
- Total: **~1.5 hours**.

## Success criteria

**Strong signal**: monotone trend in splitter-fraction across conditions, gap between blind and uniform > 2×.

**Modest signal**: monotone trend but small gap (1.5×); or a non-monotone trend driven by foveated_logpolar — supplement.

**Failure mode**: all conditions ~ same splitter fraction. Likely cause: trajectory feature too coarse (just prev-action). Mitigation: re-run with richer trajectory features (5-step rolling direction, goal_vec, distance_to_goal) before declaring null.

## Fit to backbone

Splitter cells are the canonical hippocampal way of asking "does this neuron encode pure position or position-given-trajectory?" The format axis says position is encoded differently across conditions; splitter analysis says **whether trajectory information is mixed into the position code at the unit level**. If blind has more splitters, it confirms format-differences arise because blind has to *bake trajectory into its place code*. This is the strongest unit-level cogneuro test we can do without retraining.

## Pre-registered prediction

> The fraction of units showing significant (FDR p<0.01) trajectory × position interaction in a 2-way ANOVA on cached `h_t` will be monotone decreasing across {blind, coarse, foveated_logpolar, foveated, uniform}; relative effect blind/uniform > 2×. The η²-partial of the interaction will follow the same ordering. Furthermore, projecting the top-50 splitter units onto the linear-GPS-decoder direction (from §3.4) will reveal a positive correlation between "splitter-strength" and "GPS-decoder-loading" in {blind, coarse} but not in {foveated, uniform}.

This is the single highest-leverage analysis of the round 2 set: cogneuro-canonical, low-risk, paper-ready in 2 hours, and directly answers the "what is the unit-level signature of capacity allocation" question.
