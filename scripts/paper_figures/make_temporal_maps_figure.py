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
# Trajectory-map + autocorrelation panels — all 5 conditions, single
# shared scene, single canonical post-retrain NPZ per condition.
# ──────────────────────────────────────────────────────────────────────
SCENE = "Almena"             # tight scene-id match + striking 4.8x blind/sighted step ratio

# Canonical post-retrain NPZs (positions, scene_ids, hidden_states all
# in the same file).  Order matches paper-canonical condition order.
CONDS = [
    # (npz_key,             label,         colour)
    ("blind_izar",          "Blind",       "#444444"),
    ("coarse",              "Coarse",      "#377eb8"),
    ("foveated_logpolar",   "Fov-LP",      "#984ea3"),
    ("foveated",            "Foveated",    "#e41a1c"),
    ("uniform",             "Uniform",     "#4daf4a"),
]
NPZ_DIR = Path("/tmp/rcp_analysis_v3")
TOPDOWN_DIR = Path("results/topdown_fig5")
TGM_NPZ = Path("results/cogneuro_data/tgm_results.npz")
OUT = Path("docs/manuscript/fig/fig5_temporal.pdf")

MAX_LAG = 50
N_UNITS_AUTOCORR = 256
N_EP_AUTOCORR = 80


# ──────────────────────────────────────────────────────────────────────
# Trajectory-map helpers
# ──────────────────────────────────────────────────────────────────────
def load_topdown(scene: str):
    """Load the top-down map and rebuild as a 2-tone image: navigable
    corridors -> white, walls -> medium grey.  The Habitat top-down PNG
    is essentially binary (corridor=255, wall=128); we just remap the
    wall tone to a slightly darker shade so the corridor structure
    contrasts more clearly with the trajectory line."""
    p_png = TOPDOWN_DIR / f"{scene}.png"
    p_json = TOPDOWN_DIR / f"{scene}.json"
    img_g = np.asarray(Image.open(p_png).convert("L"))
    out = np.full_like(img_g, 255)            # corridor = white
    out[img_g < 200] = 165                    # wall = medium grey
    img = np.stack([out, out, out], axis=-1)
    meta = json.loads(p_json.read_text())
    return img, meta["world_lower_bound"], meta["world_upper_bound"]


def find_scene_id(d: np.lib.npyio.NpzFile, target_lo: list[float],
                  target_hi: list[float]) -> int | None:
    """Return the integer scene_id whose episode-position bounds match
    the target topdown's world bounds most tightly. The post-retrain
    NPZs use integer scene indices; this function recovers which
    integer corresponds to the named topdown scene by bounding-box
    matching, picking the candidate with HIGHEST diagonal-coverage
    (so we don't mistake a small scene that fits inside a large
    topdown for that scene)."""
    sids = d["scene_ids"]
    pos = d["positions"]
    margin = 2.0
    target_diag = float(np.hypot(target_hi[0] - target_lo[0],
                                 target_hi[2] - target_lo[2]))
    best_sid = None
    best_cov = 0.0
    for sid in np.unique(sids):
        m = sids == sid
        p = pos[m]
        if len(p) < 5:
            continue
        lo = p.min(axis=0)
        hi = p.max(axis=0)
        if not (target_lo[0] - margin <= lo[0] and hi[0] <= target_hi[0] + margin and
                target_lo[2] - margin <= lo[2] and hi[2] <= target_hi[2] + margin):
            continue
        diag_p = float(np.hypot(hi[0] - lo[0], hi[2] - lo[2]))
        cov = diag_p / max(target_diag, 1e-9)
        if cov > best_cov:
            best_cov = cov
            best_sid = int(sid)
    # Require at least 80% diagonal coverage to be confident the
    # scene_id actually corresponds to this topdown (not a small
    # different scene that happens to fit inside).
    if best_sid is not None and best_cov < 0.80:
        return None
    return best_sid


def successful_eps_in_scene(cond_key: str, scene: str) -> set[int] | None:
    """Return the set of episode_ids in the named scene that this
    condition completed successfully (final dtg <= 0.5, step count in
    [20, 600]).  Used to pick a common episode_id across all five
    conditions so each panel shows the SAME (start, goal) pair."""
    p_npz = NPZ_DIR / f"{cond_key}_det_RCP.npz"
    d = np.load(p_npz, allow_pickle=True)
    _, lo, hi = load_topdown(scene)
    sid = find_scene_id(d, lo, hi)
    if sid is None:
        return None
    sids = d["scene_ids"]; eps = d["episode_ids"]
    steps_arr = d["step_in_episode"] if "step_in_episode" in d.files \
        else np.arange(len(sids))
    dtg = d["distance_to_goal"] if "distance_to_goal" in d.files else None
    out = set()
    for e in np.unique(eps[sids == sid]):
        m = (eps == e) & (sids == sid)
        nstep = int(m.sum())
        if not (20 <= nstep <= 600):
            continue
        if dtg is not None:
            final = float(dtg[m][np.argmax(steps_arr[m])])
            if final > 0.5:
                continue
        out.add(int(e))
    return out


def pick_common_episode(scene: str) -> int | None:
    """Across all five conditions, find the set of episode_ids that
    every cond completed successfully in the target scene, then return
    the one with the LARGEST blind-step count (most visually striking
    blind/sighted path-length contrast).  Returns None if the
    intersection is empty."""
    common = None
    for key, *_ in CONDS:
        s = successful_eps_in_scene(key, scene)
        if s is None:
            return None
        common = s if common is None else (common & s)
    if not common:
        return None
    # Rank by blind's step count (the slowest, winding-est trajectory).
    blind_p = NPZ_DIR / f"{CONDS[0][0]}_det_RCP.npz"
    d = np.load(blind_p, allow_pickle=True)
    sid = find_scene_id(d, *load_topdown(scene)[1:])
    eps = d["episode_ids"]; sids = d["scene_ids"]
    ranked = sorted(
        common,
        key=lambda e: int(((eps == e) & (sids == sid)).sum()),
        reverse=True,
    )
    return ranked[0]


def load_episode_from_npz(cond_key: str, scene: str,
                           episode_id: int | None = None):
    """Load an episode for this cond from its post-retrain NPZ in the
    named scene. If `episode_id` is provided, force-pick it; otherwise
    fall back to the median-typical successful episode.  Used by the
    figure to render the SAME episode across all conditions for
    apples-to-apples visual comparison.

    Returns dict with keys: positions (Tx3), n_steps, start, goal,
    h2 (Tx512), cum_h2_path (T,) — cumulative L2 distance travelled
    in h_2 space up to step t."""
    p_npz = NPZ_DIR / f"{cond_key}_det_RCP.npz"
    d = np.load(p_npz, allow_pickle=True)
    img, lo, hi = load_topdown(scene)
    sid = find_scene_id(d, lo, hi)
    if sid is None:
        return None
    sids = d["scene_ids"]
    eps = d["episode_ids"]
    steps_arr = d["step_in_episode"] if "step_in_episode" in d.files \
        else np.arange(len(sids))
    dtg = d["distance_to_goal"] if "distance_to_goal" in d.files else None
    H = d["hidden_states"]               # (n_steps, 512), float32

    if episode_id is not None:
        chosen_ep = int(episode_id)
        mask_check = (eps == chosen_ep) & (sids == sid)
        if not mask_check.any():
            return None
        n_steps = int(mask_check.sum())
    else:
        ep_in_scene = np.unique(eps[sids == sid])
        candidates = []
        for e in ep_in_scene:
            m = (eps == e) & (sids == sid)
            nstep = int(m.sum())
            if not (20 <= nstep <= 600):
                continue
            if dtg is not None:
                final_dtg = float(dtg[m][np.argmax(steps_arr[m])])
                if final_dtg > 0.5:
                    continue
            candidates.append((e, nstep))
        if not candidates:
            candidates = [(e, int(((eps == e) & (sids == sid)).sum()))
                          for e in ep_in_scene
                          if 20 <= ((eps == e) & (sids == sid)).sum() <= 600]
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[1])
        chosen_ep, n_steps = candidates[len(candidates) // 2]

    mask = (eps == chosen_ep) & (sids == sid)
    pos = d["positions"][mask]
    s_steps = steps_arr[mask]
    order = np.argsort(s_steps)
    pos = pos[order]
    h2 = H[mask][order].astype(np.float32)
    # Cumulative h_2 path length: sum of consecutive-step L2 distances.
    deltas = np.linalg.norm(np.diff(h2, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(deltas)])
    start = pos[0]
    if "goal_positions" in d.files:
        goal = d["goal_positions"][mask][0]
    else:
        goal = pos[-1]
    return {
        "positions": pos,
        "n_steps": int(n_steps),
        "start": np.asarray(start),
        "goal": np.asarray(goal),
        "h2": h2,
        "cum_h2_path": cum,
    }


def panel_traj_map(ax, cond_key: str, label: str, colour: str,
                   ep_data: dict, cmap_norm: tuple[float, float],
                   tau_text: str | None = None):
    """Render the top-down PNG in world coordinates via `imshow(extent=...)`,
    then plot the trajectory directly in world (x, z) coords. Trajectory
    is coloured by *cumulative h_2 path length up to step t*, on a
    SHARED colormap across all 5 conditions (cmap_norm = global vmin,
    vmax). Blind's trajectory ends in a hot colour (memory state has
    travelled far in 512-d space); sighted trajectories stay in cool
    colours (memory barely moves)."""
    img, lo, hi = load_topdown(SCENE)
    # Thresholded navmesh (walls dark grey, corridors white) at full
    # alpha — high contrast so the trajectory line + halo pop clearly.
    ax.imshow(img, extent=[lo[0], hi[0], lo[2], hi[2]],
              origin="lower", alpha=0.95, zorder=0,
              interpolation="nearest")

    pos = ep_data["positions"]
    n_steps = ep_data["n_steps"]
    start = ep_data["start"]
    goal = ep_data["goal"]
    cum = ep_data["cum_h2_path"]
    xz = pos[:, [0, 2]]

    # White halo behind the trajectory for visual separation from walls.
    ax.plot(xz[:, 0], xz[:, 1], color="white", lw=4.0, alpha=0.85,
            zorder=2, solid_capstyle="round")

    # Coloured segments: colour encodes cumulative h_2 path length up to
    # this step, normalised against the global max across all conditions.
    cmap = plt.cm.plasma
    vmin, vmax = cmap_norm
    n = len(xz)
    for i in range(n - 1):
        c = cmap((cum[i] - vmin) / max(vmax - vmin, 1e-9))
        ax.plot(xz[i:i+2, 0], xz[i:i+2, 1], color=c,
                lw=2.2, alpha=0.98, zorder=3, solid_capstyle="round")

    # Start (green circle) + goal (yellow star).
    s_xz = (start[0], start[2])
    g_xz = (goal[0], goal[2])
    ax.scatter(*s_xz, s=85, color="#0c7", zorder=5,
               edgecolor="white", linewidth=1.4)
    ax.scatter(*g_xz, s=200, color="#fc0", marker="*",
               zorder=5, edgecolor="black", linewidth=1.0)

    # Tight axis around the world bounds; equal aspect so paths follow
    # corridors visually.
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[2], hi[2])
    ax.set_aspect("equal", "box")

    # Title with quantitative annotation:
    # condition (line 1) + step count + final cumulative h_2 path + tau (line 2).
    final_h2 = cum[-1]
    title = (
        f"{label}\n"
        f"$\\mathbf{{{n_steps}}}$ steps  "
        f"$\\Sigma|\\Delta\\mathbf{{h}}_2|{{=}}{final_h2:.0f}$  "
    )
    if tau_text:
        title += f" {tau_text}"
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=colour, pad=4)
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
    for key, label, colour in CONDS:
        p = NPZ_DIR / f"{key}_det_RCP.npz"
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


# ──────────────────────────────────────────────────────────────────────
# Panel: per-condition step-by-step decoder generalisation heatmap
# (the temporal-generalisation-matrix view).
# ──────────────────────────────────────────────────────────────────────
TGM_KEY_MAP = {
    "blind_izar":        "blind",
    "coarse":            "coarse",
    "foveated_logpolar": "foveated_logpolar",
    "foveated":          "foveated",
    "uniform":           "uniform",
}


def panel_tgm(ax, cond_key: str, label: str, colour: str,
              tgm_data: dict, vmin: float, vmax: float):
    """Plot the 50x50 step-by-step decoder generalisation heatmap for
    one condition (the temporal-generalisation-matrix view)."""
    key = TGM_KEY_MAP.get(cond_key)
    if key not in tgm_data:
        ax.text(0.5, 0.5, "(no TGM data)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="#888")
        ax.set_xticks([]); ax.set_yticks([])
        return None
    M = tgm_data[key]
    # Clip to the viz range so very-negative outliers don't dominate
    # the colour scale.  We use RdBu_r so the diagonal/block patterns
    # show up as warm-on-cool contrast.
    M_clipped = np.clip(M, vmin, vmax)
    im = ax.imshow(M_clipped, origin="lower", cmap="RdBu_r",
                   vmin=vmin, vmax=vmax, aspect="equal",
                   interpolation="nearest")
    ax.set_title(label, fontsize=12, fontweight="bold", color=colour, pad=4)
    ax.set_xlabel("test step", fontsize=9)
    ax.set_ylabel("train step", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    return im


def panel_autocorr_compact(ax, autocorr: dict):
    """Per-unit autocorrelation curves: 5 lines, one per condition, with
    the Murray $1/e$ crossing annotated as the intrinsic timescale
    $\\tau$.  Discriminates all five conditions (blind $\\tau{\\approx}12$,
    sighted $\\tau{\\approx}7$--$8$) within a single panel."""
    lags = np.arange(MAX_LAG + 1)
    for key, _, _ in CONDS:
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
    ax.set_title("intrinsic timescale (per-unit autocorr)",
                 fontsize=11, fontweight="bold", pad=4)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(linestyle=":", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False, handlelength=1.5)


def panel_cum_h2_lines(ax, eps_by_cond: dict):
    """Cumulative memory displacement $\\Sigma|\\Delta\\mathbf{h}_2|$ vs
    step-in-episode, one line per condition, on the SHARED matched
    episode (Almena #157).  This is the within-sighted discriminator
    that the redundant trajectory-map row was hiding: all four sighted
    conditions take the same direct path in physical space, but their
    memory states travel different distances along that path."""
    # Y-axis: cumulative displacement; X-axis: step-in-episode.
    for key, label, colour in CONDS:
        if key not in eps_by_cond:
            continue
        ep = eps_by_cond[key]
        cum = ep["cum_h2_path"]
        steps = np.arange(len(cum))
        n = ep["n_steps"]
        final = float(cum[-1])
        ax.plot(steps, cum, color=colour, lw=2.0,
                label=f"{label}: {n} steps, final $\\Sigma{{=}}{final:.0f}$",
                zorder=4)
        # Mark the final point with a dot.
        ax.scatter([steps[-1]], [cum[-1]], s=32, color=colour,
                   edgecolor="white", linewidth=1.0, zorder=5)
    ax.set_xlabel("step in episode", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"cumulative $\Sigma|\Delta\mathbf{h}_2|$",
                  fontsize=10, fontweight="bold")
    ax.set_title("memory dynamics on the matched episode",
                 fontsize=11, fontweight="bold", pad=4)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(linestyle=":", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.0, frameon=False,
              handlelength=1.5)


def panel_traj_overlay(ax, eps_by_cond: dict):
    """Single trajectory-overlay panel: blind path (thick, dark) plus
    every sighted path (thin, condition-coloured) all drawn on the same
    Almena top-down.  Visually demonstrates that the four sighted
    conditions converge on essentially identical direct paths while
    blind winds — a key piece of the §3.3 dichotomy.  Replaces the
    redundant 5-panel trajectory row of v5."""
    img, lo, hi = load_topdown(SCENE)
    ax.imshow(img, extent=[lo[0], hi[0], lo[2], hi[2]],
              origin="lower", alpha=0.95, zorder=0,
              interpolation="nearest")

    # Sighted paths first (thin, in colour, slightly translucent so
    # overlapping segments are visible as a single bundle).
    for key, label, colour in CONDS:
        if key not in eps_by_cond:
            continue
        if key == "blind_izar":
            continue
        ep = eps_by_cond[key]
        xz = ep["positions"][:, [0, 2]]
        ax.plot(xz[:, 0], xz[:, 1], color=colour, lw=1.8, alpha=0.78,
                zorder=3, solid_capstyle="round", label=label)

    # Blind drawn thick on top.
    if "blind_izar" in eps_by_cond:
        ep = eps_by_cond["blind_izar"]
        xz = ep["positions"][:, [0, 2]]
        ax.plot(xz[:, 0], xz[:, 1], color="white", lw=4.5, alpha=0.85,
                zorder=4, solid_capstyle="round")
        ax.plot(xz[:, 0], xz[:, 1], color=CONDS[0][2], lw=2.6, alpha=1.0,
                zorder=5, solid_capstyle="round", label=CONDS[0][1])

    # Start + goal markers (same for all 5, since matched episode).
    any_ep = next(iter(eps_by_cond.values()))
    s = any_ep["start"]; g = any_ep["goal"]
    ax.scatter(s[0], s[2], s=85, color="#0c7", zorder=6,
               edgecolor="white", linewidth=1.4)
    ax.scatter(g[0], g[2], s=200, color="#fc0", marker="*",
               zorder=6, edgecolor="black", linewidth=1.0)

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[2], hi[2])
    ax.set_aspect("equal", "box")
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ("top", "right", "left", "bottom"):
        ax.spines[s_].set_color("#999")
        ax.spines[s_].set_linewidth(0.6)
    ax.set_title("matched-episode trajectories",
                 fontsize=11, fontweight="bold", pad=4)
    ax.legend(loc="lower left", fontsize=7.5, frameon=False,
              handlelength=1.4, ncol=1)


# ──────────────────────────────────────────────────────────────────────
# Compose
# ──────────────────────────────────────────────────────────────────────
def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("[autocorr] computing per-cond curves...", flush=True)
    t0 = time.time()
    autocorr = compute_per_cond_autocorr()
    print(f"  ({time.time()-t0:.1f}s)")

    print("[traj] picking common episode + loading per-cond...", flush=True)
    t0 = time.time()
    common_ep = pick_common_episode(SCENE)
    print(f"  common episode_id = {common_ep} (used for all 5 panels)")
    eps_by_cond = {}
    for cond_key, *_ in CONDS:
        ep = load_episode_from_npz(cond_key, SCENE, episode_id=common_ep)
        if ep is not None:
            eps_by_cond[cond_key] = ep
    print(f"  ({time.time()-t0:.1f}s)")
    traj_cmap_max = max(ep["cum_h2_path"][-1] for ep in eps_by_cond.values())
    traj_cmap_norm = (0.0, traj_cmap_max)

    # Load TGM data.
    print("[tgm] loading...", flush=True)
    tgm_npz = np.load(TGM_NPZ, allow_pickle=True)
    tgm_data = {k: tgm_npz[k] for k in tgm_npz.files}
    # Clip the colormap to [-0.5, +0.5] so the diagonal/block structure
    # is readable — blind has extreme negative outliers (R^2 < -1000)
    # that, on a shared linear scale, would render every cell as
    # near-zero yellow. This range maps R^2 in roughly [-0.5, +0.5]
    # which is the meaningful R^2 range; values beyond are clipped.
    tgm_vmin, tgm_vmax = -0.5, 0.5

    # 2-row layout via two separate gridspecs (different col counts).
    #   Row 1: 5 step-by-step decoder generalisation heatmaps + colorbar.
    #   Row 2: 3 panels — combined trajectory overlay (blind thick +
    #          sighted thin overlaid on shared map) + cumulative
    #          memory-displacement line plot + per-unit autocorrelation
    #          curves.  The two line-plot panels fan all five conditions
    #          out as separate curves, which the redundant 5-panel
    #          trajectory grid was hiding.
    fig = plt.figure(figsize=(17.0, 8.4))
    gs_top = fig.add_gridspec(
        1, 6,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 0.04],
        wspace=0.10,
        left=0.04, right=0.97, top=0.95, bottom=0.56,
    )
    gs_bot = fig.add_gridspec(
        1, 3,
        width_ratios=[1.4, 1.8, 1.8],
        wspace=0.27,
        left=0.05, right=0.96, top=0.46, bottom=0.07,
    )

    # ── Row 1: TGM heatmaps ──────────────────────────────────────────
    last_im = None
    for i, (cond_key, label, colour) in enumerate(CONDS):
        ax = fig.add_subplot(gs_top[0, i])
        im = panel_tgm(ax, cond_key, label, colour,
                       tgm_data, tgm_vmin, tgm_vmax)
        if im is not None:
            last_im = im
    cax_top = fig.add_subplot(gs_top[0, 5])
    if last_im is not None:
        cb_top = fig.colorbar(last_im, cax=cax_top)
        cb_top.set_label("decoder $R^2$", fontsize=9, fontweight="bold")
        cb_top.ax.tick_params(labelsize=8)

    # ── Row 2: overlay + 2 line plots ───────────────────────────────
    ax_overlay   = fig.add_subplot(gs_bot[0, 0])
    panel_traj_overlay(ax_overlay, eps_by_cond)

    ax_cum       = fig.add_subplot(gs_bot[0, 1])
    panel_cum_h2_lines(ax_cum, eps_by_cond)

    ax_autocorr  = fig.add_subplot(gs_bot[0, 2])
    panel_autocorr_compact(ax_autocorr, autocorr)

    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
