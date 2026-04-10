"""
Shared probing utilities used by analyze_probes.py, analyze_probes_legacy.py,
and analyze_cross.py.

Consolidates Ridge probe fitting, feature preprocessing, angular error
computation, and episode-level train/test splitting.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler


def fit_probe(X_tr, X_te, Y_tr, Y_te, alpha=10.0):
    """Ridge regression probe: fit on train, evaluate on test.

    Returns:
        r2: float — R² score (clipped to [-10, 1] for numerical safety)
        mae: float — mean absolute error
        pred: ndarray — predictions on test set
    """
    reg = Ridge(alpha=alpha)
    reg.fit(X_tr, Y_tr)
    pred = reg.predict(X_te)
    r2 = float(np.clip(r2_score(Y_te, pred, multioutput="uniform_average"), -10, 1))
    mae = float(mean_absolute_error(Y_te, pred))
    return r2, mae, pred


def prepare_features(H_tr, H_te, pca_dim=0):
    """Standardize features and optionally reduce dimensionality via PCA.

    Args:
        H_tr: (N_train, D) training features
        H_te: (N_test, D) test features
        pca_dim: if > 0, reduce to this many dimensions

    Returns:
        H_tr, H_te: preprocessed arrays
    """
    scaler = StandardScaler()
    H_tr = scaler.fit_transform(H_tr)
    H_te = scaler.transform(H_te)
    if pca_dim > 0 and H_tr.shape[1] > pca_dim:
        n_comp = min(pca_dim, H_tr.shape[0], H_tr.shape[1])
        pca = PCA(n_components=n_comp)
        H_tr = pca.fit_transform(H_tr)
        H_te = pca.transform(H_te)
    return H_tr, H_te


def angular_mae(pred_sincos, true_headings):
    """Compute angular MAE (in degrees) from predicted sin/cos vs true headings.

    Args:
        pred_sincos: (N, 2) array of [sin(heading), cos(heading)] predictions
        true_headings: (N,) array of true headings in radians

    Returns:
        float — mean angular error in degrees
    """
    pred_angle = np.arctan2(pred_sincos[:, 0], pred_sincos[:, 1])
    diff = np.abs(np.arctan2(np.sin(pred_angle - true_headings),
                              np.cos(pred_angle - true_headings)))
    return float(np.degrees(np.mean(diff)))


def episode_split(ep_ids, train_frac=0.8, seed=42):
    """Split data indices by episode into train/test masks.

    Args:
        ep_ids: (N,) array of episode IDs per timestep
        train_frac: fraction of episodes for training
        seed: random seed for reproducibility

    Returns:
        train_mask: (N,) boolean array
        test_mask: (N,) boolean array
    """
    rng = np.random.RandomState(seed)
    unique_eps = np.unique(ep_ids)
    rng.shuffle(unique_eps)
    split = int(len(unique_eps) * train_frac)
    train_eps = set(unique_eps[:split].tolist())
    train_mask = np.array([e in train_eps for e in ep_ids])
    return train_mask, ~train_mask
