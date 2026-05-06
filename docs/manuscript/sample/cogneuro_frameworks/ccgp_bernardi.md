# Cross-Condition Generalisation Performance (CCGP)

**Reference.** Bernardi, S., Benna, M.K., Rigotti, M., Munuera, J., Fusi, S., Salzman, C.D. (2020). *The Geometry of Abstraction in the Hippocampus and Prefrontal Cortex.* **Cell** 183(4), 954-967. https://doi.org/10.1016/j.cell.2020.09.031

**One-line idea.** A representation is "abstract" (factorised) iff a linear decoder of one variable, trained on a subset of conditions, generalises to held-out conditions of the other variable.

## Original cogneuro use

Bernardi et al. recorded HPC, DLPFC, ACC in monkeys doing a context-dependent value task. They asked whether each task variable (context, value, action) was encoded in an "abstract" format that supported zero-shot generalisation to novel combinations of the *other* variables. They built an *abstraction index* by training a linear classifier on a held-out dichotomy and measuring how its accuracy compared to a naive within-condition decoder. Crucially, the variables that mattered most for behaviour (context, value) were the ones with highest CCGP, and CCGP correlated with — but is not identical to — low-dimensional / disentangled coding.

The geometric intuition: if the hidden manifold is a parallelogram in {x: pos, y: goal} space (rather than a corner-cluster), a decoder of pos trained at goal=A will work at goal=B. CCGP detects parallelism. PR / dimensionality alone do not.

## DL analogue on h_t

Define two binary dichotomies of episodes:
- D1 = `scene_id` mod 2  (split scenes into A/B)
- D2 = `goal_quadrant` (split episodes by whether goal is to the agent's left vs right)

For variable V (e.g. `pos_x_bin`, `heading_octant`, `dist_to_goal_bin`):
- Train a linear decoder of V on episodes with (D1=0, D2=0) and (D1=0, D2=1)
- Test on (D1=1, D2=0) and (D1=1, D2=1) — generalises across D1
- Repeat for the other 3 holdouts; average -> CCGP_D1(V)
- Repeat for D2 -> CCGP_D2(V)
- Within-quadrant CV decoding accuracy of V -> WCV(V)
- **Abstraction index:** AI(V) = CCGP(V) / WCV(V) in [0,1]

## Hypothesis it tests for our 5 conditions

This is the missing format-axis test. Two pre-registered predictions:

1. **Goal-relative variables (dist_to_goal, goal-vec angle):** AI ordering = foveated > uniform > coarse > blind. The foveated agent must build a goal-frame to use peripheral context, and that frame is the abstract dimension.
2. **Allocentric pose variables (pos_x, heading):** AI flat or *inverted* — blind has the most abstract pose code because it's the only one driving behaviour, and sighted agents bind pose to scene-specific visual features (low CCGP across scenes).

A hit on prediction #2 (inverted ordering for pose) would be the strongest format-vs-magnitude dissociation in the paper.

## Implementation cost

- ~80 LOC, sklearn `LogisticRegression(C=1.0, multi_class='multinomial')`.
- Use existing splitter from `scripts/probing/extra/leave_one_scene_out.py`.
- Add: `scripts/probing/extra/ccgp_abstraction.py`.
- Runtime: ~10 min per condition on CPU (50 ep x 200 step, 512-d input).
- Total: ~1 h coding + 1 h running + 30 min figure = 2.5 h.

## Pseudocode

```python
def ccgp_one_dichotomy(h, V, D, n_splits=4):
    """h: (N, 512); V: (N,) labels to decode; D: (N,) binary dichotomy."""
    scores = []
    for hold in [0, 1]:
        train = D != hold
        test  = D == hold
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(h[train], V[train])
        scores.append(clf.score(h[test], V[test]))
    return np.mean(scores)

def abstraction_index(h, V, D):
    ccgp = ccgp_one_dichotomy(h, V, D)
    # within-condition cv (5-fold within D=0 and D=1, then average)
    wcv = np.mean([cross_val_score(LogisticRegression(), h[D==d], V[D==d], cv=5).mean()
                   for d in [0, 1]])
    return ccgp / max(wcv, 1e-6)
```

## What success / failure tells us

- **Success — predicted ordering for goal-relative variables:** strongest format-axis claim in paper. Replaces a paragraph of speculation with a panel.
- **Success — inverted ordering for pose:** publishable in its own right; reframes the magnitude/format tension.
- **Null (all conditions equal):** *no harm done*, simply report and weaken format-axis language. Still publishable as methodology.
- **Inverted from prediction:** rare — would mean we have a real surprise to write about. Not a "walk-back" because we pre-register both directions.

## Risk

**Low.** No HP shopping (C=1.0, default everything). The abstraction index is a ratio of accuracies, scale-invariant. Standard sklearn. The only failure mode is if dichotomies are too coarse (every episode in same scene cluster) — fix by stratifying on `scene_id` and `goal_distance_bin`.

## Fit to capacity-allocation backbone

**Sharpens (S) the format axis.** Bernardi gives us the numerical version of "capacity goes into the right *shape*, not just the right *amount*". Pairs cleanly with PR (which asks: how many dimensions?) — CCGP asks: are those dimensions disentangled?
