"""Hyperparameter sensitivity sweep for DSA.

Default in dsa_analysis was n_delays=25, rank=30 with 50-step trajectories,
which leaves ~25 effective DMD timepoints — possibly too few.

Test: vary (n_delays, rank) and check whether within/cross signal-noise
ratio improves.
"""
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from DSA import DSA

DATA = Path("/tmp/rcp_analysis_v3")
CONDITIONS = [
    ("blind", "blind_excursion.npz"),
    ("coarse", "matched_excursion.npz"),
    ("foveated", "foveated_excursion.npz"),
    ("foveated_logpolar", "foveated_logpolar_excursion.npz"),
]
N_PCS = 10


def load_warmup(path):
    d = np.load(path, allow_pickle=True)
    h = d["hidden_states"].astype(np.float32)
    eps = d["episode_ids"]
    seg = d["segments"]
    m = seg == 0
    return [h[m & (eps == ep)] for ep in np.unique(eps[m])]


def main():
    rng = np.random.default_rng(0)
    trials_all = {c: load_warmup(DATA / f) for c, f in CONDITIONS}
    pooled = np.concatenate([np.concatenate(t, axis=0) for t in trials_all.values()], axis=0)
    pca = PCA(n_components=N_PCS, random_state=0).fit(pooled)
    proj = {c: [pca.transform(t) for t in tr] for c, tr in trials_all.items()}
    T = 50

    arrs = {c: np.stack([t[:T] for t in tr]) for c, tr in proj.items()}

    print("(n_delays, rank): {within blind, blind-coarse, blind-foveated, coarse-foveated}")
    for n_delays, rank in [(5, 20), (10, 30), (15, 40), (25, 30), (10, 80)]:
        # within-blind (split)
        idx = np.arange(100); rng.shuffle(idx)
        X = arrs["blind"][idx[:50]]
        Y = arrs["blind"][idx[50:]]
        try:
            within_b = float(DSA(X, Y, n_delays=n_delays, rank=rank, score_method="angular", iters=1500, lr=5e-3, verbose=False).fit_score())
        except Exception as e:
            within_b = f"ERR:{e}"
        # blind-coarse
        try:
            bc = float(DSA(arrs["blind"], arrs["coarse"], n_delays=n_delays, rank=rank, score_method="angular", iters=1500, lr=5e-3, verbose=False).fit_score())
        except Exception as e:
            bc = f"ERR:{e}"
        # blind-foveated
        try:
            bf = float(DSA(arrs["blind"], arrs["foveated"], n_delays=n_delays, rank=rank, score_method="angular", iters=1500, lr=5e-3, verbose=False).fit_score())
        except Exception as e:
            bf = f"ERR:{e}"
        # coarse-foveated
        try:
            cf = float(DSA(arrs["coarse"], arrs["foveated"], n_delays=n_delays, rank=rank, score_method="angular", iters=1500, lr=5e-3, verbose=False).fit_score())
        except Exception as e:
            cf = f"ERR:{e}"
        if isinstance(within_b, str):
            print(f"  ({n_delays},{rank}): {within_b}")
        else:
            print(f"  ({n_delays},{rank}): within={within_b:.3f}, b-c={bc:.3f}, b-f={bf:.3f}, c-f={cf:.3f}")


if __name__ == "__main__":
    main()
