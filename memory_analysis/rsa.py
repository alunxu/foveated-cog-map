"""
rsa.py
======
Representational Similarity Analysis (RSA) with noise-corrected RDMs.

Implements:
  1. WUC-corrected RDM  (Diedrichsen & Kriegeskorte 2017; Diedrichsen 2020)
  2. RDM comparison via Kendall's τ_a
  3. Eccentricity-split RDMs (for H1)
  4. Cross-agent RDM correlation matrix (for H2)

Theory summary
--------------
Standard RSA computes pairwise squared distances between condition means:
    d_raw(a, b) = ||ȳ_a - ȳ_b||²

This is upward-biased by measurement noise because:
    E[||ȳ_a - ȳ_b||²] = ||μ_a - μ_b||² + tr(Σ_a)/n_a + tr(Σ_b)/n_b

The WUC (Weighted Unbiased Crossvalidated) estimator removes the bias:
    d̂(a, b) = ||ȳ_a - ȳ_b||² - tr(Cov_a)/n_a - tr(Cov_b)/n_b

where tr(Cov_a) = Σ_k Var[X_k | condition a] (sum of per-dimension variances).
This is the "trace" or "total variance" of condition a.

With unequal bin sizes (which occur here for low vs. high eccentricity),
the WUC correction ensures the estimator is unbiased regardless of n_a ≠ n_b.
"""

import numpy as np
from typing import Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Core WUC-corrected distance
# ---------------------------------------------------------------------------

def wuc_distance(X_a: np.ndarray, X_b: np.ndarray) -> float:
    """Noise-corrected squared Euclidean distance between two conditions.

    Args:
        X_a: (n_a, D) activation matrix for condition a
        X_b: (n_b, D) activation matrix for condition b

    Returns:
        float — WUC-corrected distance d̂(a, b)
    """
    n_a, D = X_a.shape
    n_b, _ = X_b.shape

    mean_a = X_a.mean(axis=0)   # (D,)
    mean_b = X_b.mean(axis=0)   # (D,)

    # Squared Euclidean distance between means
    raw_dist = float(np.sum((mean_a - mean_b) ** 2))

    # Noise terms: tr(Cov_k) / n_k
    # tr(Cov_k) = sum of per-dimension variances = sum_d Var[X_d | k]
    # For n_a > 1: use unbiased estimator (divide by n-1)
    noise_a = 0.0
    if n_a > 1:
        trace_cov_a = float(np.sum(np.var(X_a, axis=0, ddof=1)))
        noise_a = trace_cov_a / n_a

    noise_b = 0.0
    if n_b > 1:
        trace_cov_b = float(np.sum(np.var(X_b, axis=0, ddof=1)))
        noise_b = trace_cov_b / n_b

    return raw_dist - noise_a - noise_b


# ---------------------------------------------------------------------------
# Full RDM construction
# ---------------------------------------------------------------------------

def build_rdm(
    hidden: np.ndarray,
    conditions: np.ndarray,
    correct_noise: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a Representational Dissimilarity Matrix (RDM).

    Args:
        hidden:     (N, D) activation matrix (all reps, all conditions)
        conditions: (N,)   condition index for each row in hidden
        correct_noise: if True, use WUC correction; if False, use raw distances

    Returns:
        rdm:        (C, C) symmetric RDM (NaN on diagonal)
        cond_ids:   (C,)   sorted unique condition indices
    """
    cond_ids = np.sort(np.unique(conditions))
    C = len(cond_ids)

    # Group activations by condition
    groups = {c: hidden[conditions == c] for c in cond_ids}
    for c, g in groups.items():
        if g.shape[0] == 0:
            raise ValueError(f"Condition {c} has no observations.")

    rdm = np.full((C, C), np.nan)

    for i, ci in enumerate(cond_ids):
        for j, cj in enumerate(cond_ids):
            if i == j:
                continue
            if not np.isnan(rdm[j, i]):
                rdm[i, j] = rdm[j, i]   # symmetric
                continue
            if correct_noise:
                rdm[i, j] = wuc_distance(groups[ci], groups[cj])
            else:
                mean_i = groups[ci].mean(axis=0)
                mean_j = groups[cj].mean(axis=0)
                rdm[i, j] = float(np.sum((mean_i - mean_j) ** 2))

    return rdm, cond_ids


# ---------------------------------------------------------------------------
# RDM comparison: Kendall's τ_a
# ---------------------------------------------------------------------------

def kendall_tau_a(rdm1: np.ndarray, rdm2: np.ndarray) -> float:
    """Compare two RDMs using Kendall's τ_a on upper-triangle entries.
    Theory : 
    Kendall's τ_a is preferred over Spearman ρ for RDM comparison because:
    (a) it handles ties correctly (ties are treated as 0, not ranked)
    (b) it is a proper distance metric on the space of orderings
    (c) it is the standard in neuroscience RSA (Nili et al. 2014)

    Args:
        rdm1, rdm2: (C, C) symmetric RDMs (NaN diagonal)

    Returns:
        float in [-1, 1] — τ_a similarity (1 = identical ordering)
    """
    C = rdm1.shape[0]
    assert rdm1.shape == rdm2.shape, "RDMs must have the same shape"

    # Extract upper triangle (excluding diagonal)
    idx = np.triu_indices(C, k=1)
    x = rdm1[idx]
    y = rdm2[idx]

    # Remove NaN pairs
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 2:
        return np.nan

    # Kendall's τ_a: concordant - discordant pairs / total pairs
    concordant = 0
    discordant = 0
    total = n * (n - 1) // 2

    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
            

    return (concordant - discordant) / total


def kendall_tau_a_fast(rdm1: np.ndarray, rdm2: np.ndarray) -> float:
    """Vectorised Kendall's τ_a — same result as above but O(n log n).

    Uses the merge-sort algorithm via scipy if available, falls back to the
    slow O(n²) version otherwise.
    """
    try:
        from scipy.stats import kendalltau
        C = rdm1.shape[0]
        idx = np.triu_indices(C, k=1)
        x = rdm1[idx]; y = rdm2[idx]
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        # kendalltau's tau_a: set variant='c' for Kendall's τ_a
        # (tau_a uses n*(n-1)/2 as denominator, ignoring ties)
        result = kendalltau(x, y, variant='c')
        return float(result.statistic)
    except ImportError:
        return kendall_tau_a(rdm1, rdm2)


# ---------------------------------------------------------------------------
# Eccentricity-split RDMs (for H1)
# ---------------------------------------------------------------------------

def split_by_eccentricity(
    hidden: np.ndarray,
    conditions: np.ndarray,
    eccentricity: np.ndarray,
    threshold: Optional[float] = None,
) -> Tuple[Dict, Dict]:
    """Split activations into low- and high-eccentricity subsets.

    Args:
        hidden:       (N, D) activation matrix
        conditions:   (N,)   condition index per row
        eccentricity: (N,)   eccentricity value per row (radians)
        threshold:    split point; defaults to median eccentricity

    Returns:
        low_data, high_data: dicts with keys 'hidden', 'conditions'
    """
    if threshold is None:
        threshold = float(np.median(eccentricity))

    low_mask  = eccentricity <= threshold
    high_mask = eccentricity >  threshold

    return (
        {"hidden": hidden[low_mask],  "conditions": conditions[low_mask]},
        {"hidden": hidden[high_mask], "conditions": conditions[high_mask]},
    )


def eccentricity_rdms(
    hidden: np.ndarray,
    conditions: np.ndarray,
    eccentricity: np.ndarray,
    threshold: Optional[float] = None,
    correct_noise: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build separate WUC-corrected RDMs for low- and high-eccentricity steps.

    Returns:
        rdm_low, rdm_high: (C, C) RDMs for low and high eccentricity.
    """
    low_data, high_data = split_by_eccentricity(
        hidden, conditions, eccentricity, threshold
    )

    rdm_low,  _ = build_rdm(low_data["hidden"],  low_data["conditions"],  correct_noise)
    rdm_high, _ = build_rdm(high_data["hidden"], high_data["conditions"], correct_noise)

    return rdm_low, rdm_high


# ---------------------------------------------------------------------------
# Cross-agent RDM comparison matrix (for H2)
# ---------------------------------------------------------------------------

def cross_agent_similarity(
    rdms: Dict[str, np.ndarray],
    compare_fn=None,
) -> np.ndarray:
    """Compute pairwise RDM similarity (Kendall's τ_a) across all agents.

    Args:
        rdms:       dict mapping agent_name → (C, C) RDM
        compare_fn: function(rdm1, rdm2) → similarity score

    Returns:
        sim_matrix: (n_agents, n_agents) similarity matrix
        agent_names: list of agent names (row/column labels)
    """
    if compare_fn is None:
        compare_fn = kendall_tau_a_fast

    names = sorted(rdms.keys())
    n = len(names)
    sim = np.full((n, n), np.nan)

    for i, ni in enumerate(names):
        sim[i, i] = 1.0
        for j, nj in enumerate(names):
            if j <= i:
                continue
            val = compare_fn(rdms[ni], rdms[nj])
            sim[i, j] = val
            sim[j, i] = val

    return sim, names


# ---------------------------------------------------------------------------
# Summary stats for a single RDM
# ---------------------------------------------------------------------------

def rdm_stats(rdm: np.ndarray) -> dict:
    """Compute descriptive statistics for a single RDM."""
    idx = np.triu_indices(rdm.shape[0], k=1)
    vals = rdm[idx]
    vals = vals[np.isfinite(vals)]
    return {
        "mean":  float(np.mean(vals)),
        "std":   float(np.std(vals)),
        "min":   float(np.min(vals)),
        "max":   float(np.max(vals)),
        "n_pairs": int(len(vals)),
    }


# ---------------------------------------------------------------------------
# Permutation test for RDM similarity significance
# ---------------------------------------------------------------------------

def permutation_test_rdm_similarity(
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    n_perms: int = 1000,
    seed: int = 42,
    compare_fn=None,
) -> Tuple[float, float]:
    """Test whether two RDMs are more similar than chance.

    Randomly permutes the row/column order of rdm2 and recomputes
    similarity n_perms times. Returns (observed_tau, p_value).
    """
    if compare_fn is None:
        compare_fn = kendall_tau_a_fast

    observed = compare_fn(rdm1, rdm2)
    C = rdm1.shape[0]
    rng = np.random.RandomState(seed)

    null = []
    for _ in range(n_perms):
        perm = rng.permutation(C)
        rdm2_perm = rdm2[np.ix_(perm, perm)]
        null.append(compare_fn(rdm1, rdm2_perm))

    null = np.array(null)
    p_value = float(np.mean(null >= observed))
    return float(observed), p_value