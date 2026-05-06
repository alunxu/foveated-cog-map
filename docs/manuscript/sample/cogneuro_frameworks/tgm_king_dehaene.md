# Temporal Generalisation Matrix (TGM)

**Reference.** King, J.-R. & Dehaene, S. (2014). *Characterizing the dynamics of mental representations: the temporal generalization method.* **Trends in Cognitive Sciences** 18(4), 203-210. https://doi.org/10.1016/j.tics.2014.01.002

**One-line idea.** Train a decoder at time t, test it at time t' for every (t,t') in [0,T]^2. The shape of the resulting square matrix diagnoses whether the code is transient, stable, or a sustained chain.

## Original cogneuro use

In MEG/fMRI, the TGM was developed to distinguish three temporal architectures of cortical codes:

- **Diagonal-only** matrix -> the neural code is *transient* — the basis vector encoding a stimulus rotates at every time step (a "moving" representation, e.g. a chain of distinct populations).
- **Square block** in upper-left -> the code is *stable / sustained*: the same population maintains the representation throughout (working memory).
- **Off-diagonal arc / band** -> the code is *recurrent* — re-instantiation later (replay-like).
- **Cross-shaped** -> a stable code gated on/off by another transient code.

This is now the standard tool for asking whether MEG decoding reflects a fixed assembly or a sequence of assemblies. Used heavily by King, Dehaene, Cogitate consortium, and replicated in animal electrophysiology.

## DL analogue on h_t

For a variable V (we will use goal-relative position, i.e. `goal_vec` in agent frame):

```
For each (t_train, t_test) in [0..T-1]^2:
    Train: ridge_t = Ridge().fit(h[ep_train, t_train, :], V[ep_train, t_train, :])
    Score: M[t_train, t_test] = ridge_t.score(h[ep_test, t_test, :], V[ep_test, t_test, :])
```

Plot M as a TxT heatmap, one per condition.

## Hypothesis for our 5 conditions

This is a one-figure test of the **format / consumption** axis. Pre-registered predictions:

| Condition | Predicted TGM shape | Rationale |
|-----------|--------------------|-----------|
| blind | broad off-diagonal extension; near-square block | must hold goal in memory from t=0; same code maintained |
| coarse 1x1 | mostly diagonal; weak off-diag | re-derives goal each step from coarse signal; no need to maintain |
| foveated 4x4 | wide diagonal band + late off-diag | hybrid: peripheral context maintains, fovea re-derives |
| uniform 4x4 | similar to coarse but stronger | richer per-step signal, less need to maintain |
| foveated_logpolar | unique signature: extended off-diag from t=0 | the log-polar prior should compress goal across time, predicting strong block structure |

A clean visual difference between the blind (block) and coarse (diagonal) conditions would be a **single panel that proves format differs**, independent of magnitude.

## Implementation cost

- ~60 LOC numpy + sklearn.
- Add: `scripts/probing/extra/temporal_generalisation.py`.
- Runtime: ~5 min per condition (T=100, ~20 ridges per row, fast).
- Total: 1.5 h end-to-end.

## Pseudocode

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

def build_step_tensor(h, ep_id, step_id, T=100):
    """Returns h_tensor (E, T, D) with NaNs for missing steps."""
    eps = np.unique(ep_id)
    H = np.full((len(eps), T, h.shape[1]), np.nan)
    for i, e in enumerate(eps):
        m = ep_id == e
        s = step_id[m]
        H[i, s[s<T], :] = h[m][s<T]
    return H

def tgm(H, Y, T=100, n_folds=5):
    """H: (E, T, D); Y: (E, T, K). Returns TxT matrix of R^2."""
    E = H.shape[0]
    M = np.zeros((T, T))
    for t_tr in range(T):
        kf = KFold(n_folds, shuffle=True, random_state=0)
        # average over folds
        scores_row = np.zeros((n_folds, T))
        for f, (tr, te) in enumerate(kf.split(np.arange(E))):
            X_tr, y_tr = H[tr, t_tr], Y[tr, t_tr]
            ok = ~np.isnan(X_tr).any(axis=1)
            clf = Ridge(alpha=1.0).fit(X_tr[ok], y_tr[ok])
            for t_te in range(T):
                X_te, y_te = H[te, t_te], Y[te, t_te]
                ok2 = ~np.isnan(X_te).any(axis=1)
                if ok2.sum() < 3:
                    scores_row[f, t_te] = np.nan
                else:
                    scores_row[f, t_te] = clf.score(X_te[ok2], y_te[ok2])
        M[t_tr] = np.nanmean(scores_row, axis=0)
    return M
```

## What success / failure tells us

- **Success — distinct shapes per condition:** one figure proves *format differs* qualitatively. Pairs with CCGP for a 1-2 punch.
- **Success — blind shows pure square block:** strongest evidence yet that the blind agent has a categorically different memory architecture.
- **Null — all matrices look like wide diagonals:** modest negative; report it as "all 5 conditions converge on a sustained-but-evolving code". Still informative.
- **Off-diagonal arcs / replay-like signatures:** a *bonus* — would justify adding the sequenceness analysis (Pick #7).

## Risk

**Low.** Ridge with alpha=1.0 default, no HP shopping. The picture is the result, not a single number, so it's hard to over-interpret. The only sensitivity is T (truncation length); fix at T=100 (covers ~80% of episodes) and **note in caption**.

## Cross-condition extension (memory transplant x TGM)

Since we already do 5x5 transplant, compute "cross-TGM" too: train decoder on condition A at time t, test on condition B at time t'. This gives a 5x5 grid of TGMs — the diagonal blocks are the within-condition TGMs (Pick #2 main result), the off-diagonals tell us whether the temporal architecture itself transplants. Adds zero extra coding cost (same code, different inputs).

## Fit to capacity-allocation backbone

**Sharpens (S) format axis** AND **provides language for consumption axis.** TGM is the canonical tool for distinguishing "code maintained for use later" (consumption: late) from "code recomputed each step" (consumption: immediate). This is exactly the consumption-axis claim that currently lacks a clean operationalisation.
