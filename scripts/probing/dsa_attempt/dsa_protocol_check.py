"""Check whether using excursion-warmup vs clean-det rollouts changes DSA result.

Run DSA between:
  blind_excursion (warmup segments, 50 steps × 100 episodes)
vs.
  blind_izar_det (clean deterministic rollouts, first 50 steps × 100 episodes)

Both should be the same agent (blind ckpt.25), just different rollout protocols.
If DSA(excursion-warmup, det-first50) ≈ within-condition floor, the protocol
difference doesn't drive the cross-condition results. If DSA is large, our
uniform-vs-others comparison is confounded.
"""
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from DSA import DSA

DATA = Path("/tmp/rcp_analysis_v3")
N_PCS = 10
N_DELAYS = 15
RANK = 40
T_PER = 50
N_TRIALS = 100


def load_warmup(path):
    d = np.load(path, allow_pickle=True)
    h = d["hidden_states"].astype(np.float32)
    eps = d["episode_ids"]
    seg = d["segments"]
    m = seg == 0
    h, eps = h[m], eps[m]
    return [h[eps == ep] for ep in np.unique(eps)][:N_TRIALS]


def load_first_T(path, T=T_PER, n=N_TRIALS):
    d = np.load(path, allow_pickle=True)
    h = d["hidden_states"].astype(np.float32)
    eps = d["episode_ids"]
    sip = d["step_in_episode"]
    trials = []
    for ep in np.unique(eps):
        m = (eps == ep) & (sip < T)
        if m.sum() < T:
            continue
        order = np.argsort(sip[m])
        trials.append(h[m][order][:T])
        if len(trials) >= n:
            break
    return trials


def main():
    blind_warmup = load_warmup(DATA / "blind_excursion.npz")
    blind_det = load_first_T(DATA / "blind_izar_det_RCP.npz")
    print(f"blind_warmup: {len(blind_warmup)} trials, len[0]={len(blind_warmup[0])}")
    print(f"blind_det: {len(blind_det)} trials, len[0]={len(blind_det[0])}")

    # Joint PCA on pooled
    pooled = np.concatenate(
        [np.concatenate(blind_warmup, axis=0),
         np.concatenate(blind_det, axis=0)],
        axis=0,
    )
    pca = PCA(n_components=N_PCS, random_state=0).fit(pooled)
    print(f"PCA top-{N_PCS} cum var = {pca.explained_variance_ratio_.sum():.3f}")

    X = np.stack([pca.transform(t)[:T_PER] for t in blind_warmup if len(t) >= T_PER])
    Y = np.stack([pca.transform(t)[:T_PER] for t in blind_det if len(t) >= T_PER])
    print(f"X={X.shape}, Y={Y.shape}")

    # Cross: warmup vs det
    score_xy = float(DSA(X, Y, n_delays=N_DELAYS, rank=RANK,
                          score_method="angular", iters=1500, lr=5e-3,
                          verbose=False).fit_score())
    # Within warmup: split-half
    rng = np.random.default_rng(0)
    idx = np.arange(len(X)); rng.shuffle(idx)
    half = len(idx) // 2
    x1, x2 = X[idx[:half]], X[idx[half:half * 2]]
    score_xx = float(DSA(x1, x2, n_delays=N_DELAYS, rank=RANK,
                          score_method="angular", iters=1500, lr=5e-3,
                          verbose=False).fit_score())
    # Within det: split-half
    idx2 = np.arange(len(Y)); rng.shuffle(idx2)
    h2 = len(idx2) // 2
    y1, y2 = Y[idx2[:h2]], Y[idx2[h2:h2 * 2]]
    score_yy = float(DSA(y1, y2, n_delays=N_DELAYS, rank=RANK,
                          score_method="angular", iters=1500, lr=5e-3,
                          verbose=False).fit_score())

    print(f"\n=== Result ===")
    print(f"  within blind_warmup (split-half): {score_xx:.4f}")
    print(f"  within blind_det (split-half):    {score_yy:.4f}")
    print(f"  cross warmup-vs-det:               {score_xy:.4f}")
    print()
    print("Interpretation:")
    print("  If score_xy ≈ score_xx ≈ score_yy → no protocol confound")
    print("  If score_xy >> score_xx, score_yy → protocol drives DSA difference")


if __name__ == "__main__":
    main()
