# Risks: which methods could DSA-style fail us, and why

The user has explicitly asked: which of these methods have known failure modes that mirror what killed our DSA experiment (HP-sensitive, narrative-shopping risk)? This document flags risks **before** we commit budget.

## What killed DSA on this dataset

From `scripts/probing/dsa_attempt/`: the pairwise DSA distance was sensitive to (a) `n_delays`, (b) `rank`, (c) the warm-up protocol (excursion-warm vs deterministic), (d) whether episodes were truncated to 50 vs 100 steps. The protocol-confound check showed `blind_excursion-warmup vs blind_det` differed by ~0.15 in DSA distance — i.e., *the same condition appears different from itself* depending on protocol. With 5 conditions and DSA differences of order 0.1-0.3, this HP-sensitivity makes any narrative trivially shoppable.

The pre-mortem rule of thumb that emerges:
> If the analysis has more than 2 sweepable HPs that visibly change the ordering of conditions, *do not include it* unless we pre-register HPs and ordering-stability checks.

## Risk classification of the shortlist

### LOW risk (proceed)

- **CCGP (Pick #1)** — fixed C=1, fixed dichotomies. No HP shopping.
- **TGM (Pick #2)** — single ridge alpha=1, single T cutoff. The picture *is* the result.
- **dPCA (#4)** — fixed bin sizes, off-the-shelf marginalisations. No HP shopping.
- **Drift (#8)** — fixed n_blocks=4, Pearson correlation. No HP shopping.
- **Tangling Q (#9)** — single subsample size; report median + 95%ile.
- **Mixed selectivity (#10)** — fixed threshold, off-the-shelf accuracy.
- **Time cells (#11)** — three template scores, fixed thresholds.

### MEDIUM risk (proceed with pre-registration)

- **MFTMA (Pick #3 / #3 file)**. Known HP-sensitive in 3 dimensions: M (number of manifolds), P (points/manifold), kappa (margin). **Mitigation:** pre-register a 3x3 sweep over (M, P), require stability of ordering across all 9 cells. Drop entirely if unstable. Most likely to be a DSA-style casualty *if* the underlying signal is small. Worth attempting because the upside (formal magnitude/format dissociation) is the highest of the shortlist.

- **TCA (#5)**. HP = rank R. Mitigation: use elbow plot, require factor stability >= 0.7 across random restarts (the `tensortools` package gives us this for free). Lower HP-sensitivity than MFTMA in practice, but the unsupervised nature means narrative-shopping (focusing on whichever of R=3,5,8 looks best) is easy. **Pre-register R selection by reconstruction-error elbow.**

- **Timescales (#6)** — bimodality test risk. The single-tau fits are stable, but if we *additionally* claim "bimodal distribution in foveated", that adds an HP (the Hartigan dip threshold or however we test bimodality). Mitigation: report the distribution itself as the result, not a derived bimodality p-value.

- **Null/potent (#12)** — choice of head (actor vs critic). Pre-register actor only.

### HIGH risk (skip unless other methods land)

- **Sequenceness / TDLM (Pick #7)**. Both conceptual fit risk (continuous-acting LSTM lacks rest periods) AND HP-sensitivity (state discretisation K, lag range, transition-of-interest matrix). This is the closest in failure profile to DSA: forced biological analogy + sweepable HPs. **Do not start unless CCGP, TGM, MFTMA all land.** Cite as future work.

## General pre-registration template (for any method)

Before running the analysis on the full 5 conditions, write a 1-page note specifying:

1. **HP grid:** what HPs will be swept, what values.
2. **Stability criterion:** the ordering of conditions on the headline metric must be invariant across X% of the HP grid.
3. **Decision rule:** if the criterion is met, include the result. If not, drop entirely (no "discuss in supplementary").
4. **Expected outcome under null:** what does the result look like if all conditions are identical? (To avoid post-hoc reframing.)
5. **Expected outcome under alternative:** specific numerical ordering predicted.

The world_model_probe `PRE_REGISTRATION.md` is a working template.

## Headline-figure rule

Methods that go in the *main paper* must satisfy:
- Single-number summary that doesn't change under reasonable HP perturbation.
- Pre-registered ordering prediction.
- Existence of a clean null narrative ("we report no difference") that doesn't require reframing.

The shortlist that meets all three: **CCGP, TGM, dPCA, drift, tangling, timescales, time-cell motifs, null-potent**.

The shortlist that risks failing one: **MFTMA, TCA, sequenceness**.

## Recommended budget allocation

```
LOW-risk picks (~5 h total)
  CCGP           2.0 h    [Pick #1, headline]
  TGM            1.5 h    [Pick #2, headline]
  Tangling Q     1.0 h    [supplementary]
  Timescales     0.5 h    [supplementary; reuse code]
                 ----
                 5.0 h

MEDIUM-risk pick (~2 h with pre-reg)
  MFTMA          2.0 h    [Pick #3, headline if HP-stable]
                 ----
                 2.0 h

Buffer: 1 h for figures + writing.
TOTAL: 8 h.
```

**Skip entirely tonight:** sequenceness (high risk, conceptual mismatch), TCA (overlaps dPCA), Rigotti shattering (overlaps PR/CCGP), drift (low priority unless time permits).

If MFTMA HP-fails, swap in **null-potent (#12)** in its place — same ~3 h cost, lower risk profile, sharper consumption-axis claim. This swap should be *pre-decided* tonight (i.e. don't decide after seeing MFTMA results, that's narrative-shopping).
