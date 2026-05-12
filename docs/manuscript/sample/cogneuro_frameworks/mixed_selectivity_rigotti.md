# Mixed-Selectivity Dimensionality

**Reference.** Rigotti, M., Barak, O., Warden, M.R., Wang, X.-J., Daw, N.D., Miller, E.K., Fusi, S. (2013). *The importance of mixed selectivity in complex cognitive tasks.* **Nature** 497, 585-590. https://doi.org/10.1038/nature12160

**One-line idea.** Single neurons in PFC encode *nonlinear mixtures* of multiple task variables; the dimensionality of the population code is what matters for read-out, not the purity of single-cell tuning. Crucially: dimensionality *collapses* on error trials and *predicts* behaviour.

## Original cogneuro use

Rigotti et al. recorded PFC during a sequence-memory task with multiple binary task variables. They (a) showed that single cells were "mixed selective" — tuned to nonlinear mixtures of variables, looking like noise to univariate tuning analyses; (b) used a "shattering dimensionality" measure: count how many of all 2^K possible binary partitions of trials can be linearly separated, normalised by the upper bound; (c) showed that this measure correlates with task accuracy and collapses on error trials.

This is closely related to (and a precursor of) Bernardi 2020 CCGP.

## DL analogue on h_t

For a set of K binary variables ({pos_left/right, head_NS/EW, goal_near/far, scene_oddeven, dist_to_goal_quartile_above_below_median}, K=5):

- Enumerate all 2^K - 2 = 30 non-trivial bipartitions of trials.
- For each, train a linear classifier and record CV accuracy.
- Shattering dimensionality (SD) = average accuracy across all bipartitions, or fraction with accuracy > 0.6.

## Hypothesis for our 5 conditions

This is *somewhat* redundant with PR (which we have) and CCGP (Pick #1). The unique value Rigotti adds: SD is *upper-bounded* by what a linear readout could possibly do for arbitrary downstream tasks. So it tells us how *flexible* the representation is — could it support new tasks?

Pre-registered: foveated > uniform > coarse > blind for shattering dim (richer sensors -> more separable random partitions).

## Implementation cost

- ~40 LOC sklearn.
- Add: `scripts/probing/extra/shattering_dim.py`.
- Runtime: ~5 min per condition (30 LDAs).
- Total: 1 h end-to-end.

## Pseudocode

```python
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def shattering_dim(h, vars_dict, threshold=0.6):
    """vars_dict: {name: (N,) binary array} for K variables."""
    K = len(vars_dict)
    names = list(vars_dict.keys())
    accs = []
    for combo in product([0,1], repeat=K):
        if all(c == combo[0] for c in combo): continue  # skip trivial
        labels = np.zeros(h.shape[0])
        for k, name in enumerate(names):
            if combo[k] == 1:
                labels = np.logical_xor(labels, vars_dict[name])
        if 0 < labels.sum() < len(labels):
            acc = cross_val_score(LogisticRegression(max_iter=1000), h, labels, cv=5).mean()
            accs.append(acc)
    return np.mean(accs), np.mean(np.array(accs) > threshold)
```

## What success / failure tells us

- **Success — strong ordering matching sensor info:** quantifies "richness" of representation in a Rigotti-aligned way. Modest add but well-cited.
- **Null — all conditions saturate:** drop.

## Risk

**Low.** No HP shopping. The threshold (0.6) is the only judgement and has standard precedent.

## Why this is rank #10 (not earlier)

This is on the boundary of "redundant with what we have" (PR + CCGP + linear probe accuracy). The unique add is the *flexibility* framing. Run it as a quick supplementary panel only if CCGP gives surprising / messy results that would benefit from this complementary lens.

## Fit to capacity-allocation backbone

**Boundary (R/S) on format axis.** Mostly redundant. The Rigotti citation is high-prestige if a reviewer asks "did you compute mixed selectivity?", but the analytical content overlaps with CCGP and PR. Run *only* if budget permits and CCGP needed reinforcement.
