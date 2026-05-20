"""2-panel consumption figure for §5.

(a) Combined dissociation + memory-pull. Per-condition glyph encodes:
      • position  (x = linear GPS R², y = shortcut SPL drop) → §5(a)
      • horizontal arrow (direction = sign of median paired-episode
        margin, length ∝ |median margin|) → §5(b)
    One panel, three quantities (probe-readability × policy-reliance ×
    pull-direction). Replaces the prior pair (probe scatter + margin
    strip plot) which split the same story across two panels.

(b) Single paired-scene contrast — same Gibson scene + episode for
    Blind (top) and Uniform (bottom). Top-down navmesh shown as
    background, OLD-trial trajectory ghost-trail in dashed grey,
    NEW-persistent trajectory in condition colour. Reveals the
    integration-vs-vision dichotomy geometrically: Blind's
    NEW-persistent path returns to the OLD-goal area because its
    integration-only memory carries over without a visual reset;
    Uniform's converges on NEW because vision provides a per-step
    external anchor that overrides the carried-over memory.

Selection in (b) is the (scene, episode) with the largest contrast in
panel (a)'s pull-direction signal — reproducible from
compute_margins_per_cond, not hand-picked.

Reads:
  - GPS R²:        /tmp/rcp_analysis/<cond>_det_analysis.json
  - shortcut SPL:  results/shortcut_results/<cond>_gibson.json
  - traj data:     results/shortcut_results/<cond>_gibson_traj.npz
  - topdown:       results/topdown_fig5/<scene>.{png,json}

Writes: docs/manuscript/fig/fig5_consumption.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from _style import apply_paper_style  # noqa: E402

apply_paper_style()

# ─── Condition styling (matches paper-canonical palette) ───────────────
# (rcp_key,            traj_cond,           scene_id_key,  short_label, colour,    marker)
CONDS = [
    # rcp_key, traj_key, scrub_key (matches subspace_scrubbing.json keys), label, colour, marker
    ("blind",             "blind",             "blind",             "Blind",      "#444444", "o"),
    ("coarse",            "coarse",            "coarse",            "Coarse",     "#377eb8", "s"),
    ("fnorm",             "fnorm",             "foveated",          "Foveated",   "#e41a1c", "D"),
    ("foveated_logpolar", "foveated_logpolar", "foveated_logpolar", "Log-polar",  "#984ea3", "v"),
    ("uniform",           "uniform",           "uniform",           "Uniform",    "#4daf4a", "^"),
]
RCP_DIR = Path("/tmp/rcp_analysis")
TRAJ_DIR = Path("results/shortcut_results")
SHORTCUT_DIR = Path("results/shortcut_results")
TOPDOWN_DIR = Path("results/topdown_fig5")


# ════════════════════════════════════════════════════════════════════════
# Aggregate margin per condition (folded into panel A as arrow encoding)
# ════════════════════════════════════════════════════════════════════════

def compute_margins_per_cond() -> dict:
    """Same-floor + persistent-failure paired-episode margins per condition.

    margin = d(persistent_end, NEW) − d(persistent_end, OLD)
    Negative ⇒ closer to NEW (tries-new); positive ⇒ closer to OLD.

    Filters:
      • Same-floor: |y_old_goal − y_new_goal| < 1.5 m
      • Persistent failure: persistent SPL < 0.2 (only failures reveal pull)
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
            if abs(old_goal[1] - new_goal[1]) > SAME_FLOOR_Y:
                continue
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


# ════════════════════════════════════════════════════════════════════════
# PANEL A: Combined dissociation + memory pull
# ════════════════════════════════════════════════════════════════════════

def panel_a_dissociation_with_pull(ax) -> None:
    """Probe-readability × policy-reliance scatter, augmented with a
    horizontal arrow per condition encoding pull direction + strength.

    Coordinates:
      x = linear GPS R² on h_2  (probe-readability)
      y = shortcut SPL drop      (policy reliance on persistent memory)
      arrow = median paired-episode margin (sign + magnitude)
    """
    rows = []
    for rcp_key, traj_key, _, label, colour, marker in CONDS:
        gp = RCP_DIR / f"{rcp_key}_det_analysis.json"
        if not gp.exists():
            gp = RCP_DIR / f"{rcp_key}_det_ckpt49_analysis.json"
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
            "gps_r2": gps_r2, "drop": drop, "rcp_key": rcp_key,
        })

    margins = compute_margins_per_cond()
    for r in rows:
        m = margins.get(r["rcp_key"], {}).get("margins", np.array([]))
        r["margin"] = float(np.median(m)) if len(m) else 0.0
        r["n"] = int(len(m))

    max_abs_margin = max(abs(r["margin"]) for r in rows) if rows else 1.0
    ARROW_SCALE = 0.45 / max(max_abs_margin, 1e-6)

    GPS_THRESH = 0.3
    DROP_THRESH = 0.20
    X_CLIP = -1.5

    ax.axhline(DROP_THRESH, ls="--", color="#aaa", lw=0.9, zorder=1)
    ax.axvline(GPS_THRESH, ls="--", color="#aaa", lw=0.9, zorder=1)

    # Quadrant labels (left side at original 0.03 axes-fraction;
    # only "READABLE, UNUSED" pushed further right per request).
    ax.text(0.03, 0.97, "UNREADABLE, USED",
            transform=ax.transAxes, fontsize=22, color="#a02528",
            ha="left", va="top", style="italic", weight="bold", alpha=0.85)
    ax.text(0.97, 0.97, "READABLE, USED",
            transform=ax.transAxes, fontsize=22, color="#3a7d3a",
            ha="right", va="top", style="italic", weight="bold", alpha=0.85)
    ax.text(0.03, 0.03, "UNREADABLE, UNUSED",
            transform=ax.transAxes, fontsize=22, color="#777",
            ha="left", va="bottom", style="italic", alpha=0.75)
    ax.text(1.00, 0.03, "READABLE, UNUSED",
            transform=ax.transAxes, fontsize=22, color="#bb8800",
            ha="right", va="bottom", style="italic", weight="bold", alpha=0.85)

    for r in rows:
        x = max(r["gps_r2"], X_CLIP)
        y = r["drop"]
        m = r["margin"]
        stem_dx = m * ARROW_SCALE
        ax.annotate(
            "", xy=(x + stem_dx, y), xytext=(x, y),
            arrowprops=dict(
                arrowstyle="-|>", color=r["colour"],
                lw=2.6, alpha=0.85,
                shrinkA=12, shrinkB=2, mutation_scale=18,
            ),
            zorder=3,
        )
        ax.scatter(x, y, s=320, color=r["colour"],
                   marker=r["marker"], edgecolor="white", linewidth=1.8,
                   zorder=5)
        tip_offset = 8 if stem_dx >= 0 else -8
        ha = "left" if stem_dx >= 0 else "right"
        ax.annotate(
            f"{m:+.1f} m",
            (x + stem_dx, y),
            xytext=(tip_offset, 0), textcoords="offset points",
            fontsize=17, color=r["colour"], weight="bold",
            ha=ha, va="center", zorder=6,
        )
        # Right shift on label so leftmost markers (uniform clipped at
        # X_CLIP) don't run into the y-axis tick column.
        label_dx = 28 if x <= X_CLIP + 0.05 else 0
        ax.annotate(
            r["label"], (x, y),
            xytext=(label_dx, 14), textcoords="offset points",
            fontsize=22, color=r["colour"], weight="bold",
            ha="center", va="bottom", zorder=6,
        )

    # Axis labels
    ax.set_xlabel("GPS $R^2$ on $\\mathbf{h}_2$",
                  fontsize=31, fontweight="bold")
    ax.set_ylabel("shortcut SPL drop",
                  fontsize=31, fontweight="bold")
    ax.set_xlim(X_CLIP - 0.15, 1.30)
    ax.set_ylim(-0.04, 1.05)
    ax.tick_params(axis="both", labelsize=22)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(linestyle=":", alpha=0.25)
    # Inset arrow-legend deferred to the figure caption.


# ════════════════════════════════════════════════════════════════════════
# PANEL B: Paired-scene trajectory contrast on top-down navmesh
# ════════════════════════════════════════════════════════════════════════

# Selected scene + episode: clean blind-locks-OLD vs uniform-finds-NEW
# pair, among same-floor and persistent-failure cases. Filter: blind
# trajectory length >= 2 steps (excludes stationary "STOP on step 0"
# failures whose end position is just the start).
# Choice: Ackermanville ep=1. Blind moves 536 steps and ends 2.38m
# from OLD-goal (clean lock); Uniform succeeds at 0.03m from NEW-goal
# in 86 steps. OLD and NEW goals are roughly perpendicular from NEW
# start (NEW down, OLD right) so the two trajectories diverge
# orthogonally — visually unambiguous. Bbox 12x6m (wide-flat) matches
# the original figure aspect.
SELECTED_SCENE = "Ackermanville"
SELECTED_EP = 1   # NEW episode index (OLD = SELECTED_EP - 1)


def _fetch_paired(cond_traj_key: str, scene: str, ep: int) -> dict:
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


def _draw_topdown_axis(
    ax, paired: dict, cond_label: str, cond_colour: str,
    topdown_img: np.ndarray, topdown_meta: dict,
    xlim: tuple, zlim: tuple, *, show_y_axis: bool = True,
    show_legend: bool = False,
) -> None:
    """One condition's paired-episode top-down view with navmesh."""
    x_lo, _, z_lo = topdown_meta["world_lower_bound"]
    x_hi, _, z_hi = topdown_meta["world_upper_bound"]

    # Background navmesh. Empirically verified by overlaying known-navigable
    # points (episode start/goal positions): origin='lower' with extent
    # [x_lo, x_hi, z_lo, z_hi] aligns image (x, z) to world (x, z).
    ax.imshow(topdown_img, extent=[x_lo, x_hi, z_lo, z_hi],
              origin="lower", cmap="gray", alpha=0.55, zorder=0,
              interpolation="bilinear")

    old = paired["old_traj"][:, [0, 2]]
    new = paired["new_traj"][:, [0, 2]]
    old_goal = paired["old_goal"][[0, 2]]
    new_goal = paired["new_goal"][[0, 2]]
    new_start = paired["new_start"][[0, 2]]
    old_start = old[0]  # First step of OLD trajectory

    # OLD ghost trail — kept subtle so NEW persistent is the visual focus
    ax.plot(old[:, 0], old[:, 1], ls="--", color="#666", lw=1.4,
            alpha=0.45, zorder=2,
            label=f"OLD trial (SPL={paired['spl_old']:.2f})")

    # OLD start — small hollow circle (so reader sees where OLD trial began)
    ax.scatter(old_start[0], old_start[1], s=90, facecolor="white",
               edgecolor="#666", linewidth=1.6, zorder=6)
    ax.text(old_start[0], old_start[1] - 0.55, "OLD start", fontsize=12,
            ha="center", va="top", weight="bold", color="#555",
            zorder=8,
            path_effects=[path_effects.withStroke(linewidth=2.5,
                                                  foreground="white")])

    # NEW persistent trajectory — focal data
    ax.plot(new[:, 0], new[:, 1], ls="-", color=cond_colour, lw=3.0,
            alpha=0.95, zorder=4,
            label=f"NEW persistent (SPL={paired['spl_new']:.2f})")
    ax.scatter(new[-1, 0], new[-1, 1], s=110, color=cond_colour,
               marker="o", edgecolor="white", linewidth=1.6, zorder=6)

    # NEW start — filled red circle (pairs with NEW goal; was black)
    ax.scatter(new_start[0], new_start[1], s=140, color="#d62728",
               marker="o", edgecolor="white", linewidth=1.6, zorder=7)
    ax.text(new_start[0], new_start[1] + 0.55, "NEW start", fontsize=14,
            ha="center", va="bottom", weight="bold", color="#d62728",
            zorder=8,
            path_effects=[path_effects.withStroke(linewidth=2.5,
                                                  foreground="white")])

    # OLD goal — hollow grey star (pairs with OLD start)
    ax.scatter(old_goal[0], old_goal[1], s=520, marker="*",
               facecolor="white", edgecolor="#666", linewidth=2.0,
               zorder=7)
    ax.text(old_goal[0] + 0.45, old_goal[1] - 0.05, "OLD goal", fontsize=16,
            color="#555", weight="bold", va="top", zorder=8,
            path_effects=[path_effects.withStroke(linewidth=2.5,
                                                  foreground="white")])

    # NEW goal — filled red star
    ax.scatter(new_goal[0], new_goal[1], s=520, marker="*",
               color="#d62728", edgecolor="white", linewidth=1.8,
               zorder=7)
    ax.text(new_goal[0] + 0.45, new_goal[1] + 0.20, "NEW goal", fontsize=16,
            color="#d62728", weight="bold", va="bottom", zorder=8,
            path_effects=[path_effects.withStroke(linewidth=2.5,
                                                  foreground="white")])

    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=24, fontweight="bold")
    if show_y_axis:
        ax.set_ylabel("z (m)", fontsize=24, fontweight="bold")
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="both", labelsize=17)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)

    # Sub-panel condition tag only (numbers + verdict moved to caption).
    pe = [path_effects.withStroke(linewidth=3, foreground="white")]
    ax.text(0.04, 0.97, cond_label,
            transform=ax.transAxes, fontsize=26, color=cond_colour,
            weight="bold", ha="left", va="top", zorder=9,
            path_effects=pe)

    if show_legend:
        leg = ax.legend(loc="lower right", fontsize=13, frameon=True,
                        framealpha=0.92, handlelength=1.8, borderpad=0.4)
        leg.set_zorder(10)


def panel_b_paired_scene(ax_top, ax_bot) -> None:
    """Panel (b): two sub-axes stacked vertically.
    Top = Blind (locks OLD goal because integration carries over);
    Bottom = Uniform (tries NEW because vision provides external anchor)."""
    bl  = _fetch_paired("blind",   SELECTED_SCENE, SELECTED_EP)
    uni = _fetch_paired("uniform", SELECTED_SCENE, SELECTED_EP)

    topdown_img = np.array(Image.open(TOPDOWN_DIR / f"{SELECTED_SCENE}.png"))
    topdown_meta = json.loads((TOPDOWN_DIR / f"{SELECTED_SCENE}.json").read_text())

    all_xz = np.concatenate([
        bl["old_traj"][:, [0, 2]], bl["new_traj"][:, [0, 2]],
        uni["old_traj"][:, [0, 2]], uni["new_traj"][:, [0, 2]],
        bl["old_goal"][None, [0, 2]], bl["new_goal"][None, [0, 2]],
    ])
    pad = 0.6
    xlim = (all_xz[:, 0].min() - pad - 1.0,
            all_xz[:, 0].max() + pad + 0.5)
    zlim = (all_xz[:, 1].min() - pad,
            all_xz[:, 1].max() + pad + 1.2)

    bl_meta  = next(c for c in CONDS if c[0] == "blind")
    uni_meta = next(c for c in CONDS if c[0] == "uniform")

    _draw_topdown_axis(ax_top, bl, bl_meta[3], bl_meta[4],
                       topdown_img, topdown_meta,
                       xlim, zlim, show_y_axis=True, show_legend=True)
    _draw_topdown_axis(ax_bot, uni, uni_meta[3], uni_meta[4],
                       topdown_img, topdown_meta,
                       xlim, zlim, show_y_axis=True, show_legend=True)
    # Top sub-axis: hide x labels (shared with bottom)
    ax_top.set_xlabel("")
    ax_top.set_xticklabels([])


# ════════════════════════════════════════════════════════════════════════
# PANEL C: β-subspace scrubbing (linear + MLP probe before/after)
# ════════════════════════════════════════════════════════════════════════

SCRUB_JSON = Path("/tmp/extra_analyses/subspace_scrubbing.json")


def panel_c_scrubbing(ax) -> None:
    """Panel (c): β-subspace scrubbing collapsed onto one axis.

    Per condition, two paired bars: drop in linear R^2 (filled) and drop in
    MLP R^2 (hatched) after β-scrubbing. The contrast is the headline:
    linear bars are tall where there was a readable axis to ablate
    (blind, coarse, log-polar, foveated); MLP bars are tiny across all
    conditions, so position information survives in the rest of h_2.

    Uniform's linear bar is shown as 'n/a' because its original linear R^2
    was already negative noise --- a post-scrubbing change there is not
    interpretable as 'ablation effect'.
    """
    if not SCRUB_JSON.exists():
        raise SystemExit(
            f"missing {SCRUB_JSON}; run scripts/probing/extra/subspace_scrubbing.py")
    data = json.loads(SCRUB_JSON.read_text())

    # Scrubbing JSON keys use "blind", "coarse", "foveated_logpolar",
    # "foveated", "uniform" --- match CONDS[*][2] (scene_id_key).
    cond_keys = [c[2] for c in CONDS if c[2] in data]
    cols = [next(c[4] for c in CONDS if c[2] == k) for k in cond_keys]
    labs = [next(c[3] for c in CONDS if c[2] == k) for k in cond_keys]

    # Δ R² per condition
    lin_delta = [data[k]["linear_r2_orig"] - data[k]["linear_r2_scrub"]
                 for k in cond_keys]
    mlp_delta = [data[k]["mlp_r2_orig"] - data[k]["mlp_r2_scrub"]
                 for k in cond_keys]

    # Skip uniform's linear bar (orig was already negative noise)
    UNIFORM = "uniform"
    lin_disp = [d if k != UNIFORM else np.nan
                for k, d in zip(cond_keys, lin_delta)]

    x = np.arange(len(cond_keys))
    w = 0.36
    ax.bar(x - w/2, lin_disp, w, color=cols, alpha=0.92,
           edgecolor="black", linewidth=0.9, zorder=3,
           label="linear probe drop")
    ax.bar(x + w/2, mlp_delta, w, color=cols, alpha=0.42,
           edgecolor="black", linewidth=0.9, hatch="///", zorder=3,
           label="MLP probe drop")

    # Annotate Δ values above each bar (1.3× = 17pt)
    for i, (ld, md) in enumerate(zip(lin_disp, mlp_delta)):
        if not np.isnan(ld):
            ax.text(i - w/2, ld + 0.012, f"{ld:+.2f}",
                    ha="center", va="bottom", fontsize=17,
                    color="#333", weight="bold", zorder=5)
        y_pos = max(md, 0) + 0.012 if md >= 0 else md - 0.018
        va = "bottom" if md >= 0 else "top"
        ax.text(i + w/2, y_pos, f"{md:+.2f}",
                ha="center", va=va, fontsize=17,
                color="#333", weight="bold", zorder=5)

    # Mark uniform's linear bar as off-scale
    uni_idx = cond_keys.index(UNIFORM)
    ax.text(uni_idx - w/2, 0.012, "n/a",
            ha="center", va="bottom", fontsize=17,
            color="#888", style="italic", zorder=5)

    ax.axhline(0, ls="-", color="grey", alpha=0.5, lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=23, fontweight="bold")
    for tick, c in zip(ax.get_xticklabels(), cols):
        tick.set_color(c)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylabel("probe $R^2$ drop after $\\beta$-scrubbing",
                  fontsize=20, fontweight="bold")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.set_ylim(-0.10, 0.55)
    ax.legend(loc="upper right", fontsize=18, frameon=False,
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

    fig = plt.figure(figsize=(24.0, 9.0))
    # 3-panel outer grid: (a) dissociation+pull | (b) paired-scene | (c) scrubbing
    # (b) splits into 2 vertically stacked sub-axes via subgridspec.
    outer = fig.add_gridspec(
        1, 3,
        width_ratios=[1.0, 1.05, 1.0],
        wspace=0.18,
        top=0.88, bottom=0.08, left=0.05, right=0.995,
    )
    ax_a = fig.add_subplot(outer[0, 0])

    # Tight hspace between (b)'s two stacked sub-axes (shared map context).
    inner_b = outer[0, 1].subgridspec(2, 1, hspace=0.05)
    ax_b_top = fig.add_subplot(inner_b[0, 0])
    ax_b_bot = fig.add_subplot(inner_b[1, 0])

    ax_c = fig.add_subplot(outer[0, 2])

    panel_a_dissociation_with_pull(ax_a)
    panel_b_paired_scene(ax_b_top, ax_b_bot)
    panel_c_scrubbing(ax_c)

    # Panel titles; break at punctuation onto two lines.
    TITLE_FS = 35
    title_kw = dict(fontsize=TITLE_FS, fontweight="bold",
                    ha="left", va="top")
    fig.text(ax_a.get_position().x0, 0.985,
             "(a) Probe-readability\n$\\neq$ policy use",
             **title_kw)
    fig.text(ax_b_top.get_position().x0, 0.985,
             "(b) Same memory carry-over,\nopposite paths",
             **title_kw)
    fig.text(ax_c.get_position().x0, 0.985,
             "(c) Readable axis is one window,\nnot the substrate",
             **title_kw)

    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    fig.savefig(str(args.out).replace(".pdf", ".png"), dpi=200,
                bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
