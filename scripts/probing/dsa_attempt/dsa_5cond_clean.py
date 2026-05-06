"""DSA pairwise distance — all 5 conditions, CONSISTENT deterministic-rollout protocol.

After protocol-confound check (/tmp/dsa_protocol_check.py) revealed that
blind_excursion-warmup vs blind_det differ by ~0.15 in DSA distance, we
re-run pairwise DSA using only the clean deterministic-rollout files for
all 5 conditions, taking first 50 steps per episode.

Data: /tmp/rcp_analysis_v3/{blind_izar,coarse,foveated,foveated_logpolar,uniform}_det_RCP.npz
HP: n_delays=15, rank=40, T_per_trial=50, n_trials=100 (per condition)
PCA: 10-d on pooled h_t.
Score: angular DSA distance (range [0, π]).
"""
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from DSA import DSA

DATA = Path("/tmp/rcp_analysis_v3")
FILES = {
    "blind": "blind_izar_det_RCP.npz",
    "coarse": "coarse_det_RCP.npz",
    "foveated": "foveated_det_RCP.npz",
    "foveated_logpolar": "foveated_logpolar_det_RCP.npz",
    "uniform": "uniform_det_RCP.npz",
}
N_PCS = 10
N_DELAYS = 15
RANK = 40
T_PER = 50
N_TRIALS = 100


def load_first_T(path: Path, T: int = T_PER, n: int = N_TRIALS):
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
    print("=== Loading all 5 deterministic-rollout files ===")
    trials_per_cond = {}
    for cond, fname in FILES.items():
        try:
            t = load_first_T(DATA / fname)
            print(f"  {cond}: {len(t)} trials × {len(t[0])} steps")
            trials_per_cond[cond] = t
        except Exception as e:
            print(f"  ERROR {cond}: {e}")

    # Joint PCA on pooled
    pooled = np.concatenate([np.concatenate(t, axis=0) for t in trials_per_cond.values()], axis=0)
    pca = PCA(n_components=N_PCS, random_state=0).fit(pooled)
    print(f"PCA top-{N_PCS} cumulative var = {pca.explained_variance_ratio_.sum():.3f}")

    cond_arrays = {}
    for cond, trials in trials_per_cond.items():
        proj = [pca.transform(t)[:T_PER] for t in trials if len(t) >= T_PER]
        arr = np.stack(proj[:N_TRIALS])
        cond_arrays[cond] = arr
        print(f"  {cond}: {arr.shape}")

    # Within-condition split-half (floor)
    rng = np.random.default_rng(0)
    print("\n=== Within-condition (floor) ===")
    floors = {}
    for cond in cond_arrays:
        idx = np.arange(len(cond_arrays[cond]))
        rng.shuffle(idx)
        half = len(idx) // 2
        X = cond_arrays[cond][idx[:half]]
        Y = cond_arrays[cond][idx[half:half * 2]]
        try:
            d = float(DSA(X, Y, n_delays=N_DELAYS, rank=RANK, score_method="angular",
                          iters=1500, lr=5e-3, verbose=False).fit_score())
            floors[cond] = d
            print(f"  {cond}: within DSA = {d:.4f}")
        except Exception as e:
            print(f"  {cond}: ERROR {e}")

    # Pairwise
    print("\n=== Pairwise DSA ===")
    cond_keys = list(cond_arrays.keys())
    n = len(cond_keys)
    dsa_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            X = cond_arrays[cond_keys[i]]
            Y = cond_arrays[cond_keys[j]]
            try:
                score = float(DSA(X, Y, n_delays=N_DELAYS, rank=RANK,
                                  score_method="angular", iters=1500, lr=5e-3,
                                  verbose=False).fit_score())
                dsa_dist[i, j] = dsa_dist[j, i] = score
                print(f"  {cond_keys[i]} vs {cond_keys[j]}: {score:.4f}")
            except Exception as e:
                print(f"  {cond_keys[i]} vs {cond_keys[j]}: ERROR {e}")
                dsa_dist[i, j] = dsa_dist[j, i] = np.nan

    # Print labelled matrix for reading
    print("\n=== Distance matrix ===")
    header = "      " + "  ".join(f"{c[:6]:>6s}" for c in cond_keys)
    print(header)
    for i, ci in enumerate(cond_keys):
        row = f"{ci[:6]:>6s} " + "  ".join(f"{dsa_dist[i, j]:>6.3f}" for j in range(n))
        print(row)

    out = {
        "conditions": cond_keys,
        "n_pcs": N_PCS,
        "n_delays": N_DELAYS,
        "rank": RANK,
        "T_per_trial": T_PER,
        "n_trials": N_TRIALS,
        "score_method": "angular",
        "within_condition_floor": floors,
        "dsa_distance_matrix": dsa_dist.tolist(),
    }
    OUT = Path("/tmp/dsa_5cond_clean_results.json")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
