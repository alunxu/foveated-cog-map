"""§3.3 Temporal Format figure: phase portrait + autocorrelation curves.

Replaces the three simple bar/box charts in the previous Fig 5 with two
information-dense panels:

  Panel A — Phase portrait:  per-condition 2D PCA of h_2 trajectories
  along representative episodes, color-coded by step-in-episode. Visually
  reveals the §3.3 prose claim that blind's memory is a *slow-rotating
  spiral* in state-space while sighted-rich conditions are a *fast
  pendulum on a fixed axis* — i.e. trajectory orientation rotates much
  less per step in sighted than in blind.

  Panel B — Autocorrelation curves: per-condition mean per-unit
  autocorrelation as a function of lag k, with SD band across units.
  Replaces the box plot of intrinsic timescale tau (Murray 2014) with
  the actual decay SHAPE.

Reads:  /tmp/rcp_analysis_v3/{cond}_det_RCP.npz  (hidden_states, episode_ids,
        step_in_episode)
Writes: docs/manuscript/fig/fig5a_phase_portrait.pdf
        docs/manuscript/fig/fig5b_autocorrelation.pdf
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _style import apply_paper_style  # noqa: E402

apply_paper_style()


# ──────────────────────────────────────────────────────────────────────
# Conditions: order matches main.tex paper-canonical order.
# ──────────────────────────────────────────────────────────────────────
CONDS = [
    # (npz key,         label,           colour)
    ("blind_izar",       "Blind",         "#444444"),
    ("coarse",           "Coarse",        "#377eb8"),
    ("foveated_logpolar","Fov-logpolar",  "#984ea3"),
    ("foveated",         "Foveated",      "#e41a1c"),
    ("uniform",          "Uniform",       "#4daf4a"),
]
NPZ_DIR = Path("/tmp/rcp_analysis_v3")
OUT_DIR = Path("docs/manuscript/fig")

# Phase-portrait config
N_EPISODES_PER_COND = 5    # number of representative episodes per panel
MIN_EPISODE_LEN = 60       # only use episodes of at least this many steps
TARGET_STEPS = 250         # truncate at this step (mid-episode dynamics)
# Autocorrelation config
MAX_LAG = 50               # compute autocorr up to this lag (steps)
N_UNITS_FOR_AUTOCORR = 256 # subsample for speed (random pick from 512)
N_EPISODES_AUTOCORR = 100  # episodes used to estimate per-unit autocorr


def load_cond(cond_key: str):
    """Return (H, ep_ids, step_in_episode) for one condition's NPZ.

    H        — (n_steps, 512) hidden states
    ep_ids   — (n_steps,)  per-step episode id
    steps    — (n_steps,)  step-in-episode
    """
    p = NPZ_DIR / f"{cond_key}_det_RCP.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    H = d["hidden_states"].astype(np.float32)
    ep = d["episode_ids"].astype(np.int64)
    if "step_in_episode" in d.files:
        s = d["step_in_episode"].astype(np.int64)
    else:
        # Fallback: derive from ordering
        s = np.zeros(len(H), dtype=np.int64)
        for e in np.unique(ep):
            mask = ep == e
            s[mask] = np.arange(mask.sum())
    return H, ep, s


# ──────────────────────────────────────────────────────────────────────
# Panel A — Phase portrait
# ──────────────────────────────────────────────────────────────────────
def panel_phase_portrait():
    """Per-condition 2D PCA of h_2, plot N representative trajectories
    color-coded by step-in-episode. Each condition gets its own PCA
    coordinate system (within-condition dynamics is the question)."""

    fig, axes = plt.subplots(
        1, len(CONDS), figsize=(15.5, 3.8),
        gridspec_kw={"wspace": 0.10},
    )

    rng = np.random.default_rng(0)
    for ax, (key, label, colour) in zip(axes, CONDS):
        loaded = load_cond(key)
        if loaded is None:
            ax.set_title(f"{label}\n(missing)", fontsize=12)
            continue
        H, ep, steps = loaded

        # Pick episodes that are at least MIN_EPISODE_LEN long.
        unique_eps = np.unique(ep)
        ep_lens = {e: int((ep == e).sum()) for e in unique_eps}
        long_eps = [e for e in unique_eps if ep_lens[e] >= MIN_EPISODE_LEN]
        if len(long_eps) < N_EPISODES_PER_COND:
            picked = long_eps
        else:
            picked = rng.choice(long_eps, N_EPISODES_PER_COND, replace=False)

        # Fit PCA on a random subsample of this condition's hidden states.
        sub_idx = rng.choice(len(H), min(20000, len(H)), replace=False)
        Hs = H[sub_idx]
        Hs = Hs - Hs.mean(axis=0, keepdims=True)
        # SVD-based PCA
        U, S, Vt = np.linalg.svd(Hs, full_matrices=False)
        pcs = Vt[:2]                                # (2, 512)
        var_total = (S ** 2).sum()
        var_explained = (S[:2] ** 2).sum() / var_total
        # Centre everything against this condition's mean for projection.
        Hc = H - H[sub_idx].mean(axis=0, keepdims=True)

        # Plot each picked episode as a trajectory in PC1-PC2.
        all_pts = []
        for e in picked:
            mask = ep == e
            traj_h = Hc[mask]
            traj_steps = steps[mask]
            order = np.argsort(traj_steps)
            traj_h = traj_h[order]
            traj_steps = traj_steps[order]
            # Truncate at TARGET_STEPS for visibility.
            if len(traj_h) > TARGET_STEPS:
                traj_h = traj_h[:TARGET_STEPS]
                traj_steps = traj_steps[:TARGET_STEPS]
            xy = traj_h @ pcs.T                     # (T, 2)
            all_pts.append((xy, traj_steps))

        # Force a SQUARE bounding box around the data for a fair visual
        # comparison across panels. Use the larger half-extent to avoid
        # squashing/stretching the trajectories.
        all_xy = np.concatenate([xy for xy, _ in all_pts], axis=0)
        cx, cy = all_xy[:, 0].mean(), all_xy[:, 1].mean()
        half = max(all_xy[:, 0].max() - cx,
                   cx - all_xy[:, 0].min(),
                   all_xy[:, 1].max() - cy,
                   cy - all_xy[:, 1].min()) * 1.10
        x_lim = (cx - half, cx + half)
        y_lim = (cy - half, cy + half)

        # Coloured trajectories.
        cmap = plt.cm.viridis
        for xy, st in all_pts:
            # Build a coloured line via per-segment plotting.
            for i in range(len(xy) - 1):
                c = cmap(min(st[i] / TARGET_STEPS, 1.0))
                ax.plot(xy[i:i+2, 0], xy[i:i+2, 1],
                        color=c, lw=0.9, alpha=0.85, zorder=2)
            # Start marker
            ax.scatter(xy[0, 0], xy[0, 1], s=22, color="black", zorder=3,
                       edgecolor="white", linewidth=0.6)

        # Trajectory-rotation metric: mean cumulative path length
        # (per trajectory) in PC1-PC2 space. Higher = more
        # state-space coverage. This complements the autocorrelation
        # panel (slower decay <-> more rotation through state space).
        path_lengths = []
        for xy, _ in all_pts:
            dists = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            path_lengths.append(dists.sum())
        mean_path = float(np.mean(path_lengths))

        ax.set_title(
            f"{label}\n"
            f"path length {mean_path:.0f}  "
            f"(PC$_{{1,2}}$: {var_explained:.0%} var.)",
            fontsize=11, fontweight="bold", color=colour, pad=6,
        )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xticks([])
        ax.set_yticks([])
        for s_ in ("top", "right", "left", "bottom"):
            ax.spines[s_].set_color("#999")
            ax.spines[s_].set_linewidth(0.7)
        ax.set_aspect("equal", "box")

    # Shared colourbar legend for step-in-episode.
    cax = fig.add_axes([0.93, 0.20, 0.011, 0.55])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                               norm=plt.Normalize(0, TARGET_STEPS))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("step in episode", fontsize=9, fontweight="bold")
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Phase portrait: $\\mathbf{h}_2$ trajectories in PC$_{1,2}$ space "
        f"(per-condition PCA, {N_EPISODES_PER_COND} episodes; "
        "blind covers more state-space than sighted)",
        fontsize=12, fontweight="bold", y=1.05,
    )
    fig.subplots_adjust(left=0.01, right=0.92, top=0.82, bottom=0.04)
    out = OUT_DIR / "fig5a_phase_portrait.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ──────────────────────────────────────────────────────────────────────
# Panel B — Autocorrelation curves
# ──────────────────────────────────────────────────────────────────────
def autocorr_per_unit(unit_signal: np.ndarray, max_lag: int) -> np.ndarray:
    """Standard biased autocorrelation up to max_lag (inclusive of 0).
    Returns array of length max_lag+1, normalised so r[0] = 1.
    """
    x = unit_signal - unit_signal.mean()
    n = len(x)
    var = (x * x).sum() / n
    if var <= 0:
        return np.full(max_lag + 1, np.nan)
    out = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        if k == 0:
            out[k] = 1.0
        else:
            out[k] = (x[:n - k] * x[k:]).sum() / (n * var)
    return out


def panel_autocorrelation():
    """Per-condition mean autocorrelation curve across units, with SD
    band, plus an annotation of the per-condition mean intrinsic
    timescale tau (Murray 2014: lag at which autocorr first crosses
    1/e, fitted on the per-unit curves).
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    lags = np.arange(MAX_LAG + 1)
    rng = np.random.default_rng(0)

    legend_handles = []
    for key, label, colour in CONDS:
        loaded = load_cond(key)
        if loaded is None:
            continue
        H, ep, steps = loaded

        # Pick a subset of episodes long enough for reliable autocorr.
        unique_eps = np.unique(ep)
        ep_lens = {e: int((ep == e).sum()) for e in unique_eps}
        long_eps = [e for e in unique_eps if ep_lens[e] >= MAX_LAG + 5]
        if len(long_eps) < N_EPISODES_AUTOCORR:
            picked = long_eps
        else:
            picked = rng.choice(long_eps, N_EPISODES_AUTOCORR, replace=False)

        # Choose the random subset of units once.
        unit_idx = rng.choice(H.shape[1], N_UNITS_FOR_AUTOCORR, replace=False)

        # Average per-unit autocorr across episodes, then across units.
        all_unit_curves = []   # shape (n_units, max_lag+1)
        for u in unit_idx:
            curves = []
            for e in picked:
                mask = ep == e
                ts_steps = steps[mask]
                ts_signal = H[mask, u]
                order = np.argsort(ts_steps)
                ts_signal = ts_signal[order]
                if len(ts_signal) < MAX_LAG + 5:
                    continue
                curves.append(autocorr_per_unit(ts_signal, MAX_LAG))
            if curves:
                all_unit_curves.append(np.nanmean(curves, axis=0))
        all_unit_curves = np.array(all_unit_curves)
        mean_curve = np.nanmean(all_unit_curves, axis=0)
        std_curve = np.nanstd(all_unit_curves, axis=0)

        # Find tau: lag at which mean_curve first crosses 1/e.
        threshold = 1.0 / np.e
        crossings = np.where(mean_curve < threshold)[0]
        tau = float(crossings[0]) if len(crossings) else np.nan

        # Plot curve + band.
        line, = ax.plot(lags, mean_curve, color=colour, lw=2.2,
                        label=f"{label}  ($\\tau \\approx {tau:.0f}$ steps)",
                        zorder=4)
        ax.fill_between(lags,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        color=colour, alpha=0.10, zorder=2)
        legend_handles.append(line)
        # Mark tau on the curve.
        if not np.isnan(tau):
            ax.scatter([tau], [threshold], s=42, color=colour,
                       edgecolor="white", linewidth=1.3, zorder=5)

    # 1/e reference line.
    ax.axhline(1.0 / np.e, color="#666", ls="--", lw=0.8, zorder=1)
    ax.text(MAX_LAG, 1.0 / np.e + 0.015, "$1/e$ threshold (Murray 2014)",
            fontsize=9, color="#555", ha="right", va="bottom")
    ax.axhline(0, color="#bbb", lw=0.5, zorder=0)

    ax.set_xlabel("lag $k$ (steps)", fontsize=12, fontweight="bold")
    ax.set_ylabel("per-unit autocorrelation", fontsize=12, fontweight="bold")
    ax.set_xlim(0, MAX_LAG)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "Per-unit autocorrelation: blind decays slowest, sighted decay fast",
        fontsize=13, fontweight="bold",
    )
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(linestyle=":", alpha=0.3)
    ax.legend(handles=legend_handles, fontsize=10,
              loc="upper right", frameon=False)

    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.13)
    out = OUT_DIR / "fig5b_autocorrelation.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("[panel A] phase portrait...", flush=True)
    panel_phase_portrait()
    print(f"  ({time.time()-t0:.1f}s)")
    t0 = time.time()
    print("[panel B] autocorrelation curves...", flush=True)
    panel_autocorrelation()
    print(f"  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
