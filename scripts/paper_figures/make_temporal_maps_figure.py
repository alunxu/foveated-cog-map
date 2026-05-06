"""§3.3 Temporal Format figure (v2): Wijmans-style real-environment maps
+ compact autocorrelation panel.

Replaces the abstract PCA phase portrait with actual top-down environment
maps showing agent trajectories color-coded by step-in-episode (one
canonical episode per condition, all in the same Gibson scene). The
visual claim 'blind takes a winding path through the environment;
sighted takes a direct path' is grounded in physical space.

Layout: 5 panels in a row.
  Panels 1-4: top-down trajectory maps for blind / coarse / foveated /
  uniform on a single shared scene (8WUmhLawc2A), colour = step.
  Panel 5:    compact per-unit autocorrelation curves (5 conditions
  including foveated-logpolar), with 1/e threshold.

Reads:
  results/shortcut_results/{cond}_gibson_traj.npz  (positions, scenes)
  results/topdown_fig5/{scene}.png + .json         (Habitat top-down map)
  /tmp/rcp_analysis_v3/{cond}_det_RCP.npz          (h_2 for autocorr)

Writes: docs/manuscript/fig/fig5_temporal.pdf (replaces fig5a + fig5b)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from _style import apply_paper_style  # noqa: E402

apply_paper_style()


# ──────────────────────────────────────────────────────────────────────
# Trajectory-map panels (4 conditions, single shared scene).
# ──────────────────────────────────────────────────────────────────────
SCENE = "8WUmhLawc2A"        # has top-down PNG + 10 episodes per cond
EP_RANK = 0                  # 0 = first reset episode in this scene

TRAJ_CONDS = [
    # (filename_key, label,   colour)
    ("blind",         "Blind",          "#444444"),
    ("matched",       "Coarse",         "#377eb8"),
    ("foveated",      "Foveated",       "#e41a1c"),
    ("uniform",       "Uniform",        "#4daf4a"),
]

# Autocorrelation panel — uses the canonical post-retrain NPZs.
AUTOCORR_CONDS = [
    ("blind_izar",        "Blind",          "#444444"),
    ("coarse",            "Coarse",         "#377eb8"),
    ("foveated_logpolar", "Fov-LP",         "#984ea3"),
    ("foveated",          "Foveated",       "#e41a1c"),
    ("uniform",           "Uniform",        "#4daf4a"),
]
AUTOCORR_NPZ_DIR = Path("/tmp/rcp_analysis_v3")
TRAJ_DIR = Path("results/shortcut_results")
TOPDOWN_DIR = Path("results/topdown_fig5")
OUT = Path("docs/manuscript/fig/fig5_temporal.pdf")

MAX_LAG = 50
N_UNITS_AUTOCORR = 256
N_EP_AUTOCORR = 80


# ──────────────────────────────────────────────────────────────────────
# Trajectory-map helpers
# ──────────────────────────────────────────────────────────────────────
def load_topdown(scene: str):
    p_png = TOPDOWN_DIR / f"{scene}.png"
    p_json = TOPDOWN_DIR / f"{scene}.json"
    img = np.asarray(Image.open(p_png).convert("RGB"))
    meta = json.loads(p_json.read_text())
    return img, meta["world_lower_bound"], meta["world_upper_bound"]


def load_episode(cond_key: str, scene: str, rank: int):
    """Pick the rank-th 'reset' episode from this condition's traj NPZ
    in the given scene. Returns (positions Tx3, n_steps)."""
    d = np.load(TRAJ_DIR / f"{cond_key}_gibson_traj.npz", allow_pickle=True)
    mask = (d["scenes"] == scene) & (d["conditions"] == "reset")
    idx_list = np.where(mask)[0]
    if len(idx_list) <= rank:
        rank = 0
    idx = idx_list[rank]
    pos = np.array(d["positions"][idx])         # (T, 3): (x, y, z)
    steps = int(d["steps"][idx])
    start = np.array(d["starts"][idx])
    goal = np.array(d["goals"][idx])
    return pos, steps, start, goal


def panel_traj_map(ax, cond_key: str, label: str, colour: str,
                   tau_text: str | None = None):
    """Render the top-down PNG in world coordinates via `imshow(extent=...)`,
    then plot the trajectory directly in world (x, z) coords. Following
    make_shortcut_canonical_figure.py: `origin='lower'` places image[0,0]
    at world (lo[0], lo[2]), which matches Habitat's coord convention so
    trajectories follow corridors rather than cutting through walls."""
    img, lo, hi = load_topdown(SCENE)
    # imshow with world-coord extent + origin='lower'; alpha=0.55 keeps
    # trajectory lines visible against the navmesh.
    ax.imshow(img, extent=[lo[0], hi[0], lo[2], hi[2]],
              origin="lower", alpha=0.55, zorder=0,
              interpolation="bilinear")

    pos, n_steps, start, goal = load_episode(cond_key, SCENE, EP_RANK)
    # Habitat positions are (x, y, z); top-down uses the (x, z) plane.
    xz = pos[:, [0, 2]]

    # Plot trajectory in world coords, colour-coded by step.
    cmap = plt.cm.viridis
    n = len(xz)
    for i in range(n - 1):
        c = cmap(i / max(n - 1, 1))
        ax.plot(xz[i:i+2, 0], xz[i:i+2, 1], color=c,
                lw=2.4, alpha=0.95, zorder=3, solid_capstyle="round")

    # Start (green circle) and goal (yellow star), directly in world coords.
    s_xz = (start[0], start[2])
    g_xz = (goal[0], goal[2])
    ax.scatter(*s_xz, s=80, color="#0c7", zorder=5,
               edgecolor="white", linewidth=1.2)
    ax.scatter(*g_xz, s=180, color="#fc0", marker="*",
               zorder=5, edgecolor="black", linewidth=0.9)

    # Tight axis around the world bounds; equal aspect so paths follow
    # corridors visually.
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[2], hi[2])
    ax.set_aspect("equal", "box")

    # Title with quantitative annotation.
    title = f"{label}    {n_steps} steps"
    if tau_text:
        title += f"    {tau_text}"
    ax.set_title(title, fontsize=12, fontweight="bold", color=colour, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ("top", "right", "left", "bottom"):
        ax.spines[s_].set_color("#999")
        ax.spines[s_].set_linewidth(0.6)


# ──────────────────────────────────────────────────────────────────────
# Autocorrelation helper (compact panel).
# ──────────────────────────────────────────────────────────────────────
def autocorr_per_unit(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - x.mean()
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


def compute_per_cond_autocorr() -> dict:
    """Return {cond_key: (mean_curve, tau_value, colour, label)}."""
    rng = np.random.default_rng(0)
    out = {}
    for key, label, colour in AUTOCORR_CONDS:
        p = AUTOCORR_NPZ_DIR / f"{key}_det_RCP.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        H = d["hidden_states"].astype(np.float32)
        ep = d["episode_ids"].astype(np.int64)
        s = d["step_in_episode"].astype(np.int64) if \
            "step_in_episode" in d.files else np.arange(len(H))
        unique_eps = np.unique(ep)
        ep_lens = {e: int((ep == e).sum()) for e in unique_eps}
        long_eps = [e for e in unique_eps if ep_lens[e] >= MAX_LAG + 5]
        if len(long_eps) < N_EP_AUTOCORR:
            picked = long_eps
        else:
            picked = rng.choice(long_eps, N_EP_AUTOCORR, replace=False)
        unit_idx = rng.choice(H.shape[1], N_UNITS_AUTOCORR, replace=False)

        unit_curves = []
        for u in unit_idx:
            curves = []
            for e in picked:
                m = ep == e
                ts_steps = s[m]
                ts_signal = H[m, u]
                order = np.argsort(ts_steps)
                ts_signal = ts_signal[order]
                if len(ts_signal) < MAX_LAG + 5:
                    continue
                curves.append(autocorr_per_unit(ts_signal, MAX_LAG))
            if curves:
                unit_curves.append(np.nanmean(curves, axis=0))
        unit_curves = np.array(unit_curves)
        mean_curve = np.nanmean(unit_curves, axis=0)
        threshold = 1.0 / np.e
        crossings = np.where(mean_curve < threshold)[0]
        tau = float(crossings[0]) if len(crossings) else np.nan
        out[key] = (mean_curve, tau, colour, label)
    return out


def panel_autocorr_compact(ax, autocorr: dict):
    """Compact 1-panel autocorrelation summary (smaller than the full
    line plot in v1)."""
    lags = np.arange(MAX_LAG + 1)
    for key, _, _ in AUTOCORR_CONDS:
        if key not in autocorr:
            continue
        curve, tau, colour, label = autocorr[key]
        ax.plot(lags, curve, color=colour, lw=2.0, label=f"{label} ($\\tau{{=}}{tau:.0f}$)",
                zorder=4)
        if not np.isnan(tau):
            ax.scatter([tau], [1.0 / np.e], s=28, color=colour,
                       edgecolor="white", linewidth=1.0, zorder=5)
    ax.axhline(1.0 / np.e, ls="--", color="#666", lw=0.7, zorder=1)
    ax.text(MAX_LAG - 1, 1.0 / np.e + 0.03, "$1/e$", fontsize=9,
            color="#444", ha="right")
    ax.axhline(0, color="#bbb", lw=0.4, zorder=0)
    ax.set_xlim(0, MAX_LAG)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("lag $k$ (steps)", fontsize=10, fontweight="bold")
    ax.set_ylabel("per-unit autocorrelation", fontsize=10, fontweight="bold")
    ax.set_title("intrinsic timescale", fontsize=11, fontweight="bold", pad=4)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(linestyle=":", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False, handlelength=1.5)


# ──────────────────────────────────────────────────────────────────────
# Compose
# ──────────────────────────────────────────────────────────────────────
def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("[autocorr] computing per-cond curves...", flush=True)
    t0 = time.time()
    autocorr = compute_per_cond_autocorr()
    print(f"  ({time.time()-t0:.1f}s)")

    # Map traj-NPZ key -> autocorr key for tau lookup.
    AUTOCORR_FOR_TRAJ = {
        "blind":    "blind_izar",
        "matched":  "coarse",
        "foveated": "foveated",
        "uniform":  "uniform",
    }

    fig = plt.figure(figsize=(17.0, 4.0))
    gs = fig.add_gridspec(
        1, 5,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.10],
        wspace=0.05,
        left=0.01, right=0.99, top=0.91, bottom=0.13,
    )
    for i, (cond_key, label, colour) in enumerate(TRAJ_CONDS):
        ax = fig.add_subplot(gs[0, i])
        ac_key = AUTOCORR_FOR_TRAJ.get(cond_key)
        tau_text = None
        if ac_key in autocorr:
            tau = autocorr[ac_key][1]
            tau_text = f"$\\tau{{=}}{tau:.0f}$"
        panel_traj_map(ax, cond_key, label, colour, tau_text=tau_text)

    # Autocorr panel (5th column).
    ax_ac = fig.add_subplot(gs[0, 4])
    panel_autocorr_compact(ax_ac, autocorr)

    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
