"""3-panel consumption figure for §5.

(a) Dissociation scatter — probe-readability (linear GPS R²) vs. policy
    reliance (shortcut SPL drop). Coarse and uniform sit in opposite
    off-diagonal quadrants, the dissociation evidence.

(b) Aggregate margin per condition — across all paired episodes,
    distribution of (dist(persistent_end, OLD_goal) − dist(persistent_end,
    NEW_goal)). Negative ⇒ persistent agent ends closer to NEW goal
    (tries-new). Positive ⇒ closer to OLD goal (locks-onto-old).

(c) Paired-scene contrast — same Gibson scene, same paired episode
    (OLD trial then NEW trial). Two sub-axes:
      Left:  Uniform — its NEW persistent trajectory loops back toward
             the OLD goal, even though OLD is no longer the target.
      Right: Log-polar — its NEW persistent trajectory converges on
             the NEW goal, ignoring the OLD trial's destination.
    Visceral, single-scene evidence for the §5 dissociation claim.
    Scene + episode chosen by largest contrast in panel (b)'s
    aggregate margin (declared in caption — not cherry-picked).

Reads:
  - GPS R²:        /tmp/rcp_analysis/<cond>_det_analysis.json
  - shortcut SPL:  results/shortcut_results/<cond>_gibson.json
  - traj data:     results/shortcut_results/<cond>_gibson_traj.npz

Writes: docs/manuscript/fig/fig5_consumption.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── Condition styling (matches paper-canonical palette) ───────────────
# (rcp_key,            traj_cond,           scene_id_key,  short_label, colour,    marker)
CONDS = [
    # rcp_key: used for /tmp/rcp_analysis/{rcp_key}_det_analysis.json or _ckpt49_analysis.json
    # traj_key: used for results/shortcut_results/{traj_key}_traj.{json,npz}
    ("blind",             "blind",             "blind",             "Blind",      "#444444", "o"),
    ("coarse",            "coarse",            "coarse",            "Coarse",     "#377eb8", "s"),
    ("foveated_logpolar", "foveated_logpolar", "foveated_logpolar", "Log-polar",  "#984ea3", "v"),
    ("fnorm",             "fnorm",             "foveated",          "Foveated",   "#e41a1c", "D"),
    ("uniform",           "uniform",           "uniform",           "Uniform",    "#4daf4a", "^"),
]
RCP_DIR = Path("/tmp/rcp_analysis")
RCP_V3 = Path("/tmp/rcp_analysis_v3")
TRAJ_DIR = Path("results/shortcut_results")
SHORTCUT_DIR = Path("results/shortcut_results")  # same for our setup


# ════════════════════════════════════════════════════════════════════════
# PANEL A: Dissociation scatter
# ════════════════════════════════════════════════════════════════════════

def panel_a_dissociation(ax) -> None:
    """Probe-readability vs policy reliance. Uniform in upper-left
    (UNREADABLE × USED) is the headline paradox; coarse in lower-right
    (READABLE × UNUSED) is its mirror."""
    rows = []
    for rcp_key, traj_key, _, label, colour, marker in CONDS:
        gp = RCP_DIR / f"{rcp_key}_det_analysis.json"
        sp = SHORTCUT_DIR / f"{traj_key}_traj.json"
        if not (gp.exists() and sp.exists()):
            continue
        gd = json.loads(gp.read_text())
        sd = json.loads(sp.read_text())
        gps_r2 = gd["1b_global_gps_compass"]["gps_cv_r2_mean"]
        reset = sd["reset_mean_spl"]
        persist = sd["persistent_mean_spl"]
        drop = (reset - persist) / reset if reset > 0 else 0.0
        rows.append({
            "label": label, "colour": colour, "marker": marker,
            "gps_r2": gps_r2, "drop": drop,
        })

    GPS_THRESH = 0.3
    DROP_THRESH = 0.20
    X_CLIP = -1.5

    ax.axhline(DROP_THRESH, ls="--", color="#aaa", lw=0.9, zorder=1)
    ax.axvline(GPS_THRESH, ls="--", color="#aaa", lw=0.9, zorder=1)

    # Quadrant labels
    ax.text(0.03, 0.97, "UNREADABLE, USED",
            transform=ax.transAxes, fontsize=11, color="#a02528",
            ha="left", va="top", style="italic", weight="bold", alpha=0.85)
    ax.text(0.97, 0.97, "READABLE, USED",
            transform=ax.transAxes, fontsize=11, color="#3a7d3a",
            ha="right", va="top", style="italic", weight="bold", alpha=0.85)
    ax.text(0.03, 0.03, "UNREADABLE, UNUSED",
            transform=ax.transAxes, fontsize=11, color="#777",
            ha="left", va="bottom", style="italic", alpha=0.75)
    ax.text(0.97, 0.03, "READABLE, UNUSED",
            transform=ax.transAxes, fontsize=11, color="#bb8800",
            ha="right", va="bottom", style="italic", weight="bold", alpha=0.85)

    for r in rows:
        x = max(r["gps_r2"], X_CLIP)
        ax.scatter(x, r["drop"], s=300, color=r["colour"],
                   marker=r["marker"], edgecolor="white", linewidth=1.6,
                   zorder=4)
        # Right-side label
        ax.annotate(r["label"], (x, r["drop"]),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=14, color=r["colour"], weight="bold",
                    va="center", zorder=5)

    ax.set_xlabel("linear GPS $R^2$ on $\\mathbf{h}_2$",
                  fontsize=18, fontweight="bold")
    ax.set_ylabel("shortcut SPL drop\n(persist vs. reset)",
                  fontsize=18, fontweight="bold")
    ax.set_xlim(X_CLIP - 0.1, 1.15)
    ax.set_ylim(-0.02, 0.60)
    ax.tick_params(axis="both", labelsize=14)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(linestyle=":", alpha=0.25)


# ════════════════════════════════════════════════════════════════════════
# PANEL B: Aggregate margin per condition
# ════════════════════════════════════════════════════════════════════════

def compute_margins_per_cond() -> dict:
    """For each condition, return margin values across paired episodes.
    margin = d(persistent_end, NEW) − d(persistent_end, OLD).
    Negative ⇒ closer to NEW; positive ⇒ closer to OLD (locks-onto-old).

    Filter (matches §5 prose claim):
      • Same-floor: |y_old_goal − y_new_goal| < 1.5 m (otherwise OLD goal
        is on a different floor and the comparison is meaningless).
      • Persistent failure: persistent SPL < 0.2 (the persistent agent
        didn't reach the new goal — only failure cases reveal where the
        agent's memory pulls it).
    Without these filters, most paired episodes are persistent successes,
    where persistent_end ≈ NEW goal and the margin trivially negative
    for every condition.
    """
    out = {}
    SAME_FLOOR_Y = 1.5
    PERSIST_FAIL_SPL = 0.2
    for rcp_key, traj_key, _, label, colour, _ in CONDS:
        p = TRAJ_DIR / f"{traj_key}_traj.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        scenes = d["scenes"]
        eps = d["ep_idx"]
        conds = d["conditions"]
        positions = d["positions"]
        goals = d["goals"]
        spls = d["spl"]

        margins = []
        for i in range(len(scenes)):
            if conds[i] != "persistent":
                continue
            sc = scenes[i]
            ei = int(eps[i])
            prev_mask = (scenes == sc) & (eps == ei - 1)
            if not prev_mask.any():
                continue
            prev_idx = np.where(prev_mask)[0][0]
            old_goal = np.array(goals[prev_idx], dtype=float)
            new_goal = np.array(goals[i], dtype=float)
            # Same-floor filter
            if abs(old_goal[1] - new_goal[1]) > SAME_FLOOR_Y:
                continue
            # Persistent-failure filter
            if float(spls[i]) > PERSIST_FAIL_SPL:
                continue
            traj = np.array(positions[i], dtype=float)
            if traj.ndim != 2 or traj.shape[0] < 2:
                continue
            persistent_end = traj[-1]
            d_new = float(np.linalg.norm(persistent_end[[0, 2]] - new_goal[[0, 2]]))
            d_old = float(np.linalg.norm(persistent_end[[0, 2]] - old_goal[[0, 2]]))
            margins.append(d_new - d_old)
        out[rcp_key] = {"margins": np.array(margins), "label": label, "colour": colour}
    return out


def panel_b_margin(ax) -> None:
    """Strip plot of per-episode margin values per condition.
    Positive = locks-onto-old; negative = tries-new."""
    data = compute_margins_per_cond()
    rng = np.random.default_rng(42)

    positions = list(range(len(CONDS)))
    labels = []
    medians = []
    colours = []

    ax.axvline(0, ls="--", color="#888", lw=1.0, zorder=1)
    ax.axvspan(-0.05, 100, alpha=0.06, color="#3a7d3a", zorder=0)
    ax.axvspan(-100, -0.05, alpha=0.06, color="#3a7d3a", zorder=0)

    for i, (rcp_key, _, _, label, colour, _) in enumerate(CONDS):
        if rcp_key not in data:
            continue
        m = data[rcp_key]["margins"]
        if len(m) == 0:
            continue
        # Strip plot: jittered points
        y_jitter = rng.uniform(-0.18, 0.18, size=len(m))
        ax.scatter(m, [i] * len(m) + y_jitter,
                   s=18, color=colour, alpha=0.45,
                   edgecolor="none", zorder=3)
        # Median line
        med = float(np.median(m))
        ax.scatter([med], [i], s=160, color=colour,
                   marker="|", linewidth=4.0, zorder=5)
        # Inline text: median value + N
        ax.text(8.0, i, f"med={med:+.2f}m  ($n{{=}}{len(m)}$)",
                fontsize=11, color=colour, va="center", weight="bold")
        labels.append(label)
        medians.append(med)
        colours.append(colour)

    ax.set_xlabel(r"$d(\mathrm{end},\mathrm{NEW}) - d(\mathrm{end},\mathrm{OLD})$  (m)",
                  fontsize=18, fontweight="bold")
    ax.set_yticks(positions)
    ax.set_yticklabels([c[3] for c in CONDS], fontsize=14, fontweight="bold")
    for tick, c in zip(ax.get_yticklabels(),
                        [c[4] for c in CONDS]):
        tick.set_color(c)
    ax.set_xlim(-9.5, 14)
    ax.set_ylim(-0.6, len(CONDS) - 0.4)
    # Annotation: arrow + label "tries-new" / "locks-onto-old"
    ax.text(-7.5, len(CONDS) - 0.3, "← tries NEW", fontsize=12,
            color="#3a7d3a", style="italic", weight="bold", ha="center")
    ax.text(7.5, len(CONDS) - 0.3, "locks OLD →", fontsize=12,
            color="#a02528", style="italic", weight="bold", ha="center")
    ax.tick_params(axis="x", labelsize=13)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.2)


# ════════════════════════════════════════════════════════════════════════
# PANEL C: Paired-scene trajectory contrast
# ════════════════════════════════════════════════════════════════════════

# Selected scene + episode: largest paired contrast (uniform locks back to
# OLD vs. log-polar finds NEW), among same-floor and persistent-failure
# cases. Chosen via top-of-list ranking on
# (uniform_margin − log-polar_margin). Logic in
# scripts/paper_figures/_select_panel_c_scene.py (or just regenerate the
# ranking from compute_margins_per_cond above).
SELECTED_SCENE = "17DRP5sb8fy"
SELECTED_EP = 3   # NEW episode index (OLD = SELECTED_EP - 1)


def _fetch_paired(cond_traj_key: str, scene: str, ep: int) -> dict:
    """Pull OLD and NEW trajectories for one (scene, ep) pair."""
    d = np.load(TRAJ_DIR / f"{cond_traj_key}_traj.npz",
                allow_pickle=True)
    mask_new = (d["scenes"] == scene) & (d["ep_idx"] == ep) & \
               (d["conditions"] == "persistent")
    mask_old = (d["scenes"] == scene) & (d["ep_idx"] == ep - 1) & \
               (d["conditions"] == "persistent")
    i = int(np.where(mask_new)[0][0])
    pi = int(np.where(mask_old)[0][0])
    return {
        "old_traj": np.array(d["positions"][pi], dtype=float),
        "new_traj": np.array(d["positions"][i], dtype=float),
        "old_goal": np.array(d["goals"][pi], dtype=float),
        "new_goal": np.array(d["goals"][i], dtype=float),
        "new_start": np.array(d["starts"][i], dtype=float),
        "spl_new": float(d["spl"][i]),
        "spl_old": float(d["spl"][pi]),
    }


def _draw_one_traj_axis(
    ax, paired: dict, cond_label: str, cond_colour: str,
    xlim: tuple, zlim: tuple, *, show_y_axis: bool = True,
) -> None:
    """Render a single condition's paired-episode top-down view.

    Plotting conventions (axes are top-down (x, z) in metres):
      • OLD trial path: dashed light-grey, low alpha (context, executed before).
      • OLD start: small hollow grey circle (where OLD episode began).
      • NEW persistent path: solid in condition colour, the focal trajectory.
      • NEW start: black filled circle (where NEW episode begins, with
        memory carried over from the end of OLD).
      • OLD goal: hollow grey star.
      • NEW goal: filled red star.
    """
    old = paired["old_traj"][:, [0, 2]]
    new = paired["new_traj"][:, [0, 2]]
    old_goal = paired["old_goal"][[0, 2]]
    new_goal = paired["new_goal"][[0, 2]]
    new_start = paired["new_start"][[0, 2]]
    old_start = old[0]  # First step of OLD trajectory

    # OLD ghost trail — kept light to subordinate it visually to NEW
    ax.plot(old[:, 0], old[:, 1], ls="--", color="#888", lw=1.4,
            alpha=0.40, zorder=2,
            label=f"OLD trial (SPL={paired['spl_old']:.2f})")

    # OLD start — small hollow circle, no fill
    ax.scatter(old_start[0], old_start[1], s=70, facecolor="white",
               edgecolor="#666", linewidth=1.4, zorder=6)
    ax.text(old_start[0], old_start[1] - 0.5, "OLD start", fontsize=9,
            ha="center", va="top", weight="bold", color="#555",
            zorder=8)

    # NEW persistent trajectory — the focal data
    ax.plot(new[:, 0], new[:, 1], ls="-", color=cond_colour, lw=2.6,
            alpha=0.92, zorder=4,
            label=f"NEW persistent (SPL={paired['spl_new']:.2f})")
    # End-of-trajectory marker
    ax.scatter(new[-1, 0], new[-1, 1], s=80, color=cond_colour,
               marker="o", edgecolor="white", linewidth=1.4, zorder=6)

    # NEW start — filled black circle, labelled "NEW start" to disambiguate
    ax.scatter(new_start[0], new_start[1], s=110, color="black",
               marker="o", edgecolor="white", linewidth=1.4, zorder=7)
    ax.text(new_start[0], new_start[1] + 0.5, "NEW start", fontsize=10,
            ha="center", va="bottom", weight="bold", color="black",
            zorder=8)

    # OLD goal — hollow grey star
    ax.scatter(old_goal[0], old_goal[1], s=380, marker="*",
               facecolor="white", edgecolor="#666", linewidth=2.0,
               zorder=7)
    ax.text(old_goal[0] + 0.4, old_goal[1] - 0.2, "OLD goal", fontsize=11,
            color="#444", weight="bold", va="top", zorder=8)

    # NEW goal — filled red star
    ax.scatter(new_goal[0], new_goal[1], s=380, marker="*",
               color="#d62728", edgecolor="white", linewidth=1.6,
               zorder=7)
    ax.text(new_goal[0] + 0.4, new_goal[1] + 0.2, "NEW goal", fontsize=11,
            color="#d62728", weight="bold", va="bottom", zorder=8)

    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=14, fontweight="bold")
    if show_y_axis:
        ax.set_ylabel("z (m)", fontsize=14, fontweight="bold")
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="both", labelsize=11)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(linestyle=":", alpha=0.25)

    # Condition tag + outcome verdict in upper-left (offset clear of y-axis)
    end_xy = new[-1]
    d_old = float(np.linalg.norm(end_xy - old_goal))
    d_new = float(np.linalg.norm(end_xy - new_goal))
    closer = "OLD" if d_old < d_new else "NEW"
    verdict_colour = "#a02528" if closer == "OLD" else "#3a7d3a"
    ax.text(0.06, 0.97, cond_label,
            transform=ax.transAxes, fontsize=16, color=cond_colour,
            weight="bold", ha="left", va="top", zorder=9)
    ax.text(0.06, 0.90,
            f"ends {d_old:.1f} m from OLD,  {d_new:.1f} m from NEW",
            transform=ax.transAxes, fontsize=10.5, color="#222",
            ha="left", va="top", zorder=9)
    ax.text(0.06, 0.84,
            f"$\\Rightarrow$ closer to {closer}",
            transform=ax.transAxes, fontsize=11.5, color=verdict_colour,
            weight="bold", style="italic", ha="left", va="top", zorder=9)


def panel_c_paired_scene(ax_left, ax_right) -> None:
    """Two sub-axes: same paired episode, two conditions side-by-side.

    Conditions chosen for sharpest contrast: uniform (locks back to OLD)
    vs. log-polar (finds NEW). Same scene, same start, same goals — only
    visual encoding differs.
    """
    uni = _fetch_paired("uniform", SELECTED_SCENE, SELECTED_EP)
    lp  = _fetch_paired("foveated_logpolar", SELECTED_SCENE, SELECTED_EP)

    # Joint axis range across both conditions for fair comparison.
    # Pad asymmetrically so left edge has more room for in-axis text.
    all_xz = np.concatenate([
        uni["old_traj"][:, [0, 2]], uni["new_traj"][:, [0, 2]],
        lp["old_traj"][:, [0, 2]], lp["new_traj"][:, [0, 2]],
        uni["old_goal"][None, [0, 2]], uni["new_goal"][None, [0, 2]],
    ])
    pad = 0.6
    xlim = (all_xz[:, 0].min() - pad - 1.2,  # extra left pad for inset text
            all_xz[:, 0].max() + pad)
    zlim = (all_xz[:, 1].min() - pad,
            all_xz[:, 1].max() + pad + 1.0)  # extra top pad for inset text

    # Find which conditions to draw
    uni_meta = next(c for c in CONDS if c[0] == "uniform")
    lp_meta  = next(c for c in CONDS if c[0] == "foveated_logpolar")

    _draw_one_traj_axis(ax_left,  uni, uni_meta[3], uni_meta[4],
                        xlim, zlim, show_y_axis=True)
    _draw_one_traj_axis(ax_right, lp,  lp_meta[3],  lp_meta[4],
                        xlim, zlim, show_y_axis=False)

    # One shared legend on the right axis (top-right) — small + frameless
    ax_right.legend(loc="lower right", fontsize=9, frameon=False,
                    handlelength=1.6, borderpad=0.3)


# ════════════════════════════════════════════════════════════════════════
# Compose
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("docs/manuscript/fig/fig5_consumption.pdf"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(23.0, 6.5))
    # Panel (c) splits into two equal sub-axes
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[1.0, 1.35, 1.0, 1.0],
        wspace=0.32,
        top=0.85, bottom=0.16, left=0.035, right=0.995,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c_left  = fig.add_subplot(gs[0, 2])
    ax_c_right = fig.add_subplot(gs[0, 3])

    panel_a_dissociation(ax_a)
    panel_b_margin(ax_b)
    panel_c_paired_scene(ax_c_left, ax_c_right)

    # Tighten the gap between panel (c)'s two sub-axes (shared spatial map)
    pos_l = ax_c_left.get_position()
    pos_r = ax_c_right.get_position()
    new_x0 = pos_l.x1 + 0.012
    ax_c_right.set_position([new_x0, pos_r.y0,
                              pos_r.width, pos_r.height])

    # Panel titles aligned at fig-coord y, x = panel left edge
    title_kw = dict(fontsize=20, fontweight="bold", ha="left", va="top")
    fig.text(ax_a.get_position().x0, 0.97,
             "(a) Probe-readability vs. policy use", **title_kw)
    fig.text(ax_b.get_position().x0, 0.97,
             "(b) Where does persistent memory pull?", **title_kw)
    fig.text(ax_c_left.get_position().x0, 0.97,
             f"(c) Same paired episode — Uniform vs. Log-polar  "
             f"(scene {SELECTED_SCENE}, ep{SELECTED_EP})", **title_kw)

    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    fig.savefig(str(args.out).replace(".pdf", ".png"), dpi=200,
                bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
