# Output-Potent / Output-Null Subspaces

**References.**
- Kaufman, M.T., Churchland, M.M., Ryu, S.I., Shenoy, K.V. (2014). *Cortical activity in the null space: permitting preparation without movement.* **Nature Neuroscience** 17, 440-448. https://doi.org/10.1038/nn.3643
- Saxena, S., Cunningham, J.P. (2019). *Towards the neural population doctrine.* **Curr. Opin. Neurobiol.** 55, 103-111.
- Elsayed, G.F., Lara, A.H., Kaufman, M.T., Churchland, M.M., Cunningham, J.P. (2016). *Reorganization between preparatory and movement population responses in motor cortex.* **Nature Communications** 7, 13239. https://doi.org/10.1038/ncomms13239

**One-line idea.** Decompose the hidden state into the subspace that drives the output (potent) and the subspace orthogonal to it (null). Activity in the null subspace can vary freely without affecting behaviour — and is where preparatory / cognitive computations live.

## Original cogneuro use

Kaufman et al. asked: how can M1 prepare a movement (during a delay period) without producing any actual movement? Answer: the preparatory activity sits in the null space of the output-projection matrix. Elsayed et al. extended this to show preparatory and movement activity occupy *orthogonal* subspaces in M1.

This is the canonical *consumption-axis* tool: it operationally separates "computation that drives behaviour now" from "computation that doesn't drive behaviour now".

## DL analogue on h_t

For each condition, decompose h_t into the policy-relevant subspace and its orthogonal complement:

1. Get the actor head's input-to-action linear weights W_actor (shape: [|A|, 512]).
2. The output-potent subspace = column-span of W_actor (effective dim ~|A|=4 in PointNav: forward, left, right, stop).
3. The output-null subspace = orthogonal complement (dim ~508).
4. Project h_t onto each: h_potent = P_potent h_t, h_null = P_null h_t.
5. Run the existing probes (pos, head, goal, dist) on h_potent and h_null *separately*.

The fraction of probe accuracy that comes from h_null is "private cognition" (consumed later, or not consumed at all) vs h_potent ("consumed now").

## Hypothesis for our 5 conditions

Pre-registered:

- **Goal probe in h_potent:** all conditions roughly equal (the policy uses goal directly).
- **Goal probe in h_null:** *blind > sighted* (blind agents must maintain goal in working memory across many no-action steps).
- **Position probe in h_null:** *blind > sighted* (blind dead-reckons position even when not acting on it immediately).

If this prediction holds, we have a *quantitative consumption-axis* claim: blind agents store *more state in the null space* — i.e., more "preparatory / not immediately consumed" cognition.

## Implementation cost

- ~150 LOC.
- Need to extract W_actor from the saved checkpoint (1 line per condition).
- Add: `scripts/probing/extra/null_potent_subspaces.py`.
- Runtime: ~10 min per condition (5 probes x 2 subspaces x 5 conds).
- Total: 3 h end-to-end.

## Pseudocode

```python
def get_potent_null_basis(actor_W):
    """actor_W: (|A|, D). Returns (P_potent, P_null)."""
    U, S, Vt = np.linalg.svd(actor_W, full_matrices=True)
    rank = (S > 1e-6).sum()
    V_potent = Vt[:rank].T   # (D, rank)
    V_null   = Vt[rank:].T   # (D, D-rank)
    P_potent = V_potent @ V_potent.T
    P_null   = V_null   @ V_null.T
    return P_potent, P_null

# project hidden states
h_pot = (P_potent @ h.T).T
h_nul = (P_null   @ h.T).T

# run probes on each
acc_pot = decode_position(h_pot, pos)
acc_nul = decode_position(h_nul, pos)
```

## What success / failure tells us

- **Success — null-space probes show the predicted ordering:** the most quantitative consumption-axis claim in the paper. Pairs *beautifully* with the memory-transplant result: transplant should fail when the source's null-space code is incompatible with the target's actor head.
- **Null — null and potent show the same accuracy:** would tell us the policy reads almost everything, which is itself an interesting result (no "stash" of unused cognition).

## Risk

**Medium.** Two issues:

1. **Actor-head ambiguity.** PPO has both a value head and a policy head. The "potent" subspace depends on which we use. Pre-register: use the *actor* (policy) head, not the critic. Discuss critic in supplementary if interesting.
2. **Subspace dimensionality is low (rank ~4).** With only 4 potent dimensions, the potent-subspace probe has very little capacity. Frame results as ratios (null acc / potent acc), and report potent-baseline accuracy alongside.

## Fit to capacity-allocation backbone

**Sharpens (S) consumption axis** more directly than any other method on the shortlist. The consumption axis is currently the weakest of the three in the paper because it lacks a clean operationalisation — output-null/potent decomposition gives that operationalisation.

## Recommended pairing

This *should* be Pick #4 if MFTMA HP-fails. Together with CCGP (format) and TGM (format/consumption), null/potent (consumption) gives one principled tool per axis, which is the cleanest possible story.
