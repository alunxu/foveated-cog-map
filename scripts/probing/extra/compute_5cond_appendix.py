"""Omnibus 5-condition compute for the spatial-format appendix figures
that previously used the 4-condition cache.

Outputs (under $RESULTS_OUT, default /tmp/extra_analyses_5cond):
  cka_5x5.json                 — pairwise linear CKA between top-layer h_2
  tsne_5cond.json              — t-SNE coords (5000 samples, perplexity 30)
                                 + per-cond DtG-coloured embeddings
  spatial_info_5cond.json      — per-unit spatial info distribution
                                 + sparse-vs-distributed GPS R^2 vs k
  position_axis_5cond.json     — Ridge-probe singular-direction power
                                 vs # PCs vs explained variance
  pc_cumulative_5cond.json     — Ridge probe R^2 vs # PCs + participation ratio

Reads NPZs from $NPZ_DIR (default /scratch/wxu/habitat_checkpoints_rcp/
probing_data_rcp), each containing keys: hidden_states, gps,
distance_to_goal (some), episode_ids, step_in_episode.

Designed to run on RCP (CPU is fine; sklearn + scipy + numpy only).
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np

NPZ_DIR = os.environ.get("NPZ_DIR",
    "/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp")
RESULTS_OUT = os.environ.get("RESULTS_OUT", "/tmp/extra_analyses_5cond")

CONDS = [
    ("blind",             "blind_izar_det.npz"),
    ("coarse",            "coarse_det.npz"),
    ("foveated_logpolar", "foveated_logpolar_det.npz"),
    ("foveated",          "foveated_det.npz"),
    ("uniform",           "uniform_det.npz"),
]
N_SUB = 30000          # subsample steps per cond for speed/memory
N_TSNE = 1000          # samples per cond for t-SNE
SEED = 0


def load_cond(npz_path):
    d = np.load(npz_path)
    h = d["hidden_states"].astype(np.float32)
    gps = d["gps"].astype(np.float32) if "gps" in d.files else None
    dtg = d["distance_to_goal"].astype(np.float32) if "distance_to_goal" in d.files else None
    n = len(h)
    if n > N_SUB:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(n, N_SUB, replace=False)
        h = h[idx]
        if gps is not None: gps = gps[idx]
        if dtg is not None: dtg = dtg[idx]
    return h, gps, dtg


def linear_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    XtX = X.T @ X; YtY = Y.T @ Y
    XtY = X.T @ Y
    num = (XtY * XtY).sum()
    den = np.sqrt((XtX * XtX).sum()) * np.sqrt((YtY * YtY).sum())
    return float(num / max(den, 1e-12))


def compute_cka(data):
    keys = list(data.keys())
    n = len(keys)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = linear_cka(data[keys[i]][0], data[keys[j]][0])
    return {"conds": keys, "matrix": M.tolist()}


def compute_tsne(data):
    from sklearn.manifold import TSNE
    keys = list(data.keys())
    Xs, ys, dtgs = [], [], []
    rng = np.random.default_rng(SEED)
    for k in keys:
        h, gps, dtg = data[k]
        n = len(h)
        idx = rng.choice(n, min(N_TSNE, n), replace=False)
        Xs.append(h[idx])
        ys.append(np.full(len(idx), keys.index(k)))
        dtgs.append(dtg[idx] if dtg is not None else np.zeros(len(idx)))
    X = np.concatenate(Xs); y = np.concatenate(ys); dtg = np.concatenate(dtgs)
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, init="pca")
    emb = tsne.fit_transform(X)
    return {
        "conds": keys,
        "embedding": emb.tolist(),
        "labels": y.tolist(),
        "dtg": dtg.tolist(),
    }


def compute_spatial_info(data, n_bins=20):
    """Per-unit spatial information (Skaggs 1996 in bits) using GPS bins.

    Skaggs SI is defined for non-negative firing rates; LSTM hidden
    states can be negative (tanh in [-1, 1]).  We use the paper-faithful
    rectification max(h, 0) (matching scripts/cluster/submit_skaggs_recompute.sh)
    rather than a shift, treating h2 like a place-cell firing rate.
    """
    out = {"conds": list(data.keys()), "per_cond": {}}
    for k, (h, gps, _) in data.items():
        if gps is None:
            continue
        h_rect = np.maximum(h, 0.0)                    # rectified (paper-faithful)
        xy = gps[:, :2] if gps.shape[1] >= 2 else gps
        x_edges = np.linspace(xy[:, 0].min(), xy[:, 0].max(), n_bins + 1)
        y_edges = np.linspace(xy[:, 1].min(), xy[:, 1].max(), n_bins + 1)
        x_idx = np.clip(np.digitize(xy[:, 0], x_edges) - 1, 0, n_bins - 1)
        y_idx = np.clip(np.digitize(xy[:, 1], y_edges) - 1, 0, n_bins - 1)
        bin_idx = x_idx * n_bins + y_idx
        n_total = len(h); n_units = h.shape[1]
        mean_a = h_rect.mean(0)                        # (n_units,)
        spatial_info = np.zeros(n_units)
        for b in range(n_bins * n_bins):
            m = bin_idx == b
            p_b = m.sum() / n_total
            if p_b < 1e-6:
                continue
            mean_a_b = h_rect[m].mean(0)               # (n_units,)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(mean_a > 1e-8, mean_a_b / np.maximum(mean_a, 1e-12), 0.0)
                term = ratio * np.log2(np.maximum(ratio, 1e-12))
            term = np.where(np.isfinite(term), term, 0.0)
            spatial_info += p_b * term
        # Zero-out units whose global mean is near 0 (Skaggs ill-defined).
        spatial_info = np.where(mean_a > 1e-8, spatial_info, 0.0)
        spatial_info = np.maximum(spatial_info, 0.0)   # numeric float-noise guard
        out["per_cond"][k] = spatial_info.tolist()
    return out


def compute_sparse_decoding(data, k_values=(1, 2, 5, 10, 20, 50, 100, 200, 512)):
    """Top-k spatial-info GPS decoding: probe R^2 using only top-k units."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    out = {"conds": list(data.keys()), "k_values": list(k_values), "per_cond": {}}
    for k, (h, gps, _) in data.items():
        if gps is None: continue
        # Same paper-faithful rectified Skaggs SI as compute_spatial_info()
        # for ranking units only (the probe input below is the original h).
        h_rect = np.maximum(h, 0.0)
        xy = gps[:, :2]
        n_bins = 20
        x_edges = np.linspace(xy[:, 0].min(), xy[:, 0].max(), n_bins + 1)
        y_edges = np.linspace(xy[:, 1].min(), xy[:, 1].max(), n_bins + 1)
        x_idx = np.clip(np.digitize(xy[:, 0], x_edges) - 1, 0, n_bins - 1)
        y_idx = np.clip(np.digitize(xy[:, 1], y_edges) - 1, 0, n_bins - 1)
        bin_idx = x_idx * n_bins + y_idx
        n_total = len(h); n_units = h.shape[1]
        mean_a = h_rect.mean(0)
        spatial_info = np.zeros(n_units)
        for b in range(n_bins * n_bins):
            m = bin_idx == b
            p_b = m.sum() / n_total
            if p_b < 1e-6: continue
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(mean_a > 1e-8, h_rect[m].mean(0) / np.maximum(mean_a, 1e-12), 0.0)
                term = ratio * np.log2(np.maximum(ratio, 1e-12))
            term = np.where(np.isfinite(term), term, 0.0)
            spatial_info += p_b * term
        spatial_info = np.where(mean_a > 1e-8, spatial_info, 0.0)
        spatial_info = np.maximum(spatial_info, 0.0)
        order = np.argsort(-spatial_info)
        # 5-fold CV
        rng = np.random.default_rng(SEED)
        idx = np.arange(len(h)); rng.shuffle(idx)
        folds = np.array_split(idx, 5)
        r2s_per_k = []
        for kk in k_values:
            top_units = order[:kk]
            r2_folds = []
            for fold in folds:
                tr = np.setdiff1d(idx, fold)
                X_tr = h[tr][:, top_units]; y_tr = gps[tr]
                X_te = h[fold][:, top_units]; y_te = gps[fold]
                ridge = Ridge(alpha=10.0).fit(X_tr, y_tr)
                r2_folds.append(r2_score(y_te, ridge.predict(X_te), multioutput="uniform_average"))
            r2s_per_k.append(float(np.mean(r2_folds)))
        out["per_cond"][k] = r2s_per_k
    return out


def compute_position_axis(data):
    """Ridge probe → singular directions of beta. Cumulative power vs # PCs."""
    from sklearn.linear_model import Ridge
    out = {"conds": list(data.keys()), "per_cond": {}}
    for k, (h, gps, _) in data.items():
        if gps is None: continue
        h_c = h - h.mean(0); gps_c = gps - gps.mean(0)
        # Compute PCs of h
        U, S, Vt = np.linalg.svd(h_c, full_matrices=False)
        # Variance distribution (cumulative)
        var = (S ** 2) / (S ** 2).sum()
        cum_var = np.cumsum(var)
        # Ridge beta
        ridge = Ridge(alpha=10.0).fit(h_c, gps_c)
        beta = ridge.coef_  # (2, 512)
        # Project beta onto PC basis: power per PC
        beta_pc = beta @ Vt.T  # (2, 512)
        beta_power = (beta_pc ** 2).sum(axis=0)
        cum_beta = np.cumsum(beta_power) / max(beta_power.sum(), 1e-12)
        out["per_cond"][k] = {
            "cum_var": cum_var.tolist(),
            "cum_beta_power": cum_beta.tolist(),
        }
    return out


def compute_pc_cumulative(data, n_pcs_max=50):
    """Linear readout R^2 vs # PCs + participation ratio."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    out = {"conds": list(data.keys()), "n_pcs": list(range(1, n_pcs_max + 1)),
           "per_cond": {}}
    for k, (h, gps, _) in data.items():
        if gps is None: continue
        h_c = h - h.mean(0); gps_c = gps - gps.mean(0)
        # PCA
        U, S, Vt = np.linalg.svd(h_c, full_matrices=False)
        # Participation ratio
        var = S ** 2
        pr = float((var.sum() ** 2) / (var ** 2).sum())
        # 5-fold CV per PC count
        rng = np.random.default_rng(SEED)
        idx = np.arange(len(h)); rng.shuffle(idx)
        folds = np.array_split(idx, 5)
        r2_per_pcs = []
        # Project all data onto PC basis once
        h_pc = h_c @ Vt.T  # (n, d)
        for n_pcs in range(1, n_pcs_max + 1):
            X = h_pc[:, :n_pcs]
            r2_folds = []
            for fold in folds:
                tr = np.setdiff1d(idx, fold)
                X_tr, y_tr = X[tr], gps_c[tr]; X_te, y_te = X[fold], gps_c[fold]
                ridge = Ridge(alpha=10.0).fit(X_tr, y_tr)
                r2_folds.append(r2_score(y_te, ridge.predict(X_te), multioutput="uniform_average"))
            r2_per_pcs.append(float(np.mean(r2_folds)))
        out["per_cond"][k] = {"r2_vs_pcs": r2_per_pcs, "participation_ratio": pr}
    return out


def main():
    Path(RESULTS_OUT).mkdir(exist_ok=True, parents=True)
    print(f"NPZ_DIR={NPZ_DIR}, RESULTS_OUT={RESULTS_OUT}")
    print("Loading 5 conditions...")
    data = {}
    for k, fn in CONDS:
        p = Path(NPZ_DIR) / fn
        if not p.exists():
            print(f"  MISSING {k}: {p}")
            continue
        h, gps, dtg = load_cond(p)
        data[k] = (h, gps, dtg)
        print(f"  {k}: h={h.shape}, gps={None if gps is None else gps.shape}")

    print("\n=== CKA 5×5 ===")
    out = compute_cka(data)
    Path(f"{RESULTS_OUT}/cka_5x5.json").write_text(json.dumps(out, indent=2))
    print("  saved cka_5x5.json")

    print("\n=== t-SNE (5000 samples, 1000/cond) ===")
    out = compute_tsne(data)
    Path(f"{RESULTS_OUT}/tsne_5cond.json").write_text(json.dumps(out))
    print("  saved tsne_5cond.json")

    print("\n=== Per-unit spatial info ===")
    out = compute_spatial_info(data)
    Path(f"{RESULTS_OUT}/spatial_info_5cond.json").write_text(json.dumps(out))
    print("  saved spatial_info_5cond.json")

    print("\n=== Sparse-vs-distributed decoding ===")
    out = compute_sparse_decoding(data)
    Path(f"{RESULTS_OUT}/sparse_decoding_5cond.json").write_text(json.dumps(out, indent=2))
    print("  saved sparse_decoding_5cond.json")

    print("\n=== Position-axis (cumulative beta power vs PC) ===")
    out = compute_position_axis(data)
    Path(f"{RESULTS_OUT}/position_axis_5cond.json").write_text(json.dumps(out))
    print("  saved position_axis_5cond.json")

    print("\n=== PC cumulative readout ===")
    out = compute_pc_cumulative(data)
    Path(f"{RESULTS_OUT}/pc_cumulative_5cond.json").write_text(json.dumps(out, indent=2))
    print("  saved pc_cumulative_5cond.json")

    print("\nALL DONE")


if __name__ == "__main__":
    main()
