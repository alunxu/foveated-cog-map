"""Plot Module 4 (consumption) summary as 2 single-panel PDFs:
  (a) shortcut SPL drop vs linear GPS R^2 — off-diagonal cases (probe-readable
      != policy-relied)
  (b) end-trajectory margin (avg dist to OLD goal − avg dist to NEW goal)
      across paired-episode failure cases — only uniform goes to OLD goal

Composed in main.tex into a 3-panel layout with the existing
fig5_shortcut_canonical (which becomes panel c).
"""
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = ["blind", "coarse", "foveated", "foveated_logpolar", "uniform"]
NICE = {"blind": "blind", "coarse": "coarse 1×1", "foveated": "foveated 4×4",
         "uniform": "uniform 4×4", "foveated_logpolar": "fov-logpolar"}
COLORS = {"blind": "#5b5b5b", "coarse": "#d97a35", "foveated": "#2c7fb8",
           "uniform": "#6a51a3", "foveated_logpolar": "#7fcdbb"}
MARKERS = {"blind": "o", "coarse": "s", "foveated": "D",
            "uniform": "^", "foveated_logpolar": "v"}


def plot_scatter(out_path):
    """Panel (a): SPL drop vs linear GPS R^2."""
    # Linear GPS R^2 from Fig 2(a) capacity_allocation
    linear_gps_r2 = {
        "blind": 0.94, "coarse": 0.55, "foveated": 0.16,
        "foveated_logpolar": -0.50, "uniform": -1.19,
    }
    # SPL drops from /data/shortcut/*_gibson.json
    spl_drop = {
        "blind": 0.222, "coarse": 0.090, "foveated": 0.162,
        "foveated_logpolar": 0.126, "uniform": 0.319,
    }

    fig, ax = plt.subplots(1, 1, figsize=(4.4, 3.6), constrained_layout=True)
    for c in CONDITIONS:
        ax.scatter(linear_gps_r2[c], spl_drop[c], s=180, color=COLORS[c],
                    marker=MARKERS[c], edgecolor="black", linewidth=0.8,
                    zorder=3, label=NICE[c])
        # Smart labelling to avoid overlap
        offsets = {"blind": (8, 6), "coarse": (-15, 8), "foveated": (8, 6),
                    "foveated_logpolar": (8, -10), "uniform": (8, -12)}
        ax.annotate(NICE[c], (linear_gps_r2[c], spl_drop[c]),
                     xytext=offsets.get(c, (8, 6)), textcoords="offset points",
                     fontsize=9)

    # Median lines + quadrant labels (x: left=unreadable, right=readable;
    # y: top=used, bottom=unused)
    ax.axhline(0.18, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.axvline(0.30, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.text(0.40, 0.97, "UNREADABLE, USED",
             transform=ax.transAxes, ha="right", va="top",
             fontsize=7.5, color="red", style="italic", weight="bold", alpha=0.85)
    ax.text(0.60, 0.97, "READABLE, USED",
             transform=ax.transAxes, ha="left", va="top",
             fontsize=7.5, color="green", style="italic", weight="bold", alpha=0.6)
    ax.text(0.40, 0.03, "UNREADABLE, UNUSED",
             transform=ax.transAxes, ha="right", va="bottom",
             fontsize=7.5, color="gray", style="italic", alpha=0.6)
    ax.text(0.60, 0.03, "READABLE, UNUSED",
             transform=ax.transAxes, ha="left", va="bottom",
             fontsize=7.5, color="orange", style="italic", weight="bold", alpha=0.85)

    ax.set_xlabel("linear GPS $R^2$ on $\\mathbf{h}_2$\n(probe-readable position)",
                    fontsize=10)
    ax.set_ylabel("shortcut SPL drop\n(persistent vs reset; policy reliance)",
                    fontsize=10)
    ax.set_title("Probe-readability $\\neq$ policy reliance", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def _compute_margin_from_npz(npz_path, persist_spl_max=0.2,
                              reset_spl_min=0.5, min_steps=30,
                              same_floor_dy=0.5):
    """Compute margin = dist(persist_end, NEW goal) - dist(persist_end, OLD
    goal) over same-floor failure cases.  Sign convention matches the
    figure y-axis label "↑ closer to OLD": positive value means persist
    end is FARTHER from new than from old, i.e. closer to old."""
    d = np.load(npz_path, allow_pickle=True)
    scenes = d["scenes"]; ep_idx = d["ep_idx"]; conds = d["conditions"]
    spl = d["spl"]; steps = d["steps"]; goals = d["goals"]; positions = d["positions"]
    margins = []
    for sc, ep in sorted(set(zip(scenes, ep_idx))):
        if ep == 0:
            continue
        mask = (scenes == sc) & (ep_idx == ep)
        if mask.sum() != 2:
            continue
        r_idx = np.where(mask & (conds == "reset"))[0]
        p_idx = np.where(mask & (conds == "persistent"))[0]
        if len(r_idx) != 1 or len(p_idx) != 1:
            continue
        r, p = r_idx[0], p_idx[0]
        if (spl[r] <= reset_spl_min or spl[p] >= persist_spl_max
                or steps[p] < min_steps):
            continue
        prev = (scenes == sc) & (ep_idx == ep - 1) & (conds == "reset")
        if not prev.any():
            continue
        old_goal = goals[np.where(prev)[0][0]]
        new_goal = goals[r]
        if abs(old_goal[1] - new_goal[1]) >= same_floor_dy:
            continue
        p_end = positions[p][-1]
        d_old = float(np.linalg.norm(p_end[[0, 2]] - old_goal[[0, 2]]))
        d_new = float(np.linalg.norm(p_end[[0, 2]] - new_goal[[0, 2]]))
        margins.append(d_new - d_old)        # positive = closer to OLD
    return np.array(margins)


def plot_margin(out_path, traj_dir=None):
    """Panel (b): per-condition margin (dist-to-NEW − dist-to-OLD goal,
    so positive = closer to OLD goal, matching the figure y-axis label).
    Margins computed live from each condition's shortcut NPZ rather than
    hardcoded — the analysis is reproducible from data on every render.
    """
    if traj_dir is None:
        traj_dir = Path(__file__).resolve().parents[3] / "results" / "shortcut_results"
    cond_to_npz_stem = {                       # NPZ uses encoder-side name
        "blind": "blind", "coarse": "matched", "foveated": "foveated",
        "foveated_logpolar": "foveated_logpolar", "uniform": "uniform",
    }
    conds = ["blind", "coarse", "foveated_logpolar", "foveated", "uniform"]
    vals, ns = [], []
    for c in conds:
        npz = Path(traj_dir) / f"{cond_to_npz_stem[c]}_gibson_traj.npz"
        if not npz.exists():
            print(f"WARN: missing {npz}")
            vals.append(0.0); ns.append(0); continue
        m = _compute_margin_from_npz(npz)
        vals.append(float(m.mean()) if len(m) else 0.0)
        ns.append(int(len(m)))
        print(f"  {c}: margin={vals[-1]:+.2f}m, n={ns[-1]}")

    fig, ax = plt.subplots(1, 1, figsize=(4.4, 3.6), constrained_layout=True)
    xs = np.arange(len(conds))
    bars = ax.bar(xs, vals, color=[COLORS[c] for c in conds], edgecolor="black",
                    linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([NICE[c] for c in conds], rotation=30, ha="right",
                        fontsize=9)
    ax.set_ylabel("end-trajectory margin (m)\n  ↓ closer to NEW   |   closer to OLD ↑",
                    fontsize=9)
    ax.set_title("Persistent agent goes to OLD goal --- uniform only", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    pad = max(0.4, max(abs(min(vals)), abs(max(vals))) * 0.15)
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax.text(i, v + (pad * 0.5 if v >= 0 else -pad * 0.5),
                f"{v:+.2f}m\nn={n}",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8, color="black", weight="bold" if v > 0 else "normal")
    ax.set_ylim(min(vals) - pad * 1.5, max(vals) + pad * 1.5)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_scatter", required=True)
    ap.add_argument("--out_margin", required=True)
    args = ap.parse_args()
    plot_scatter(args.out_scatter)
    plot_margin(args.out_margin)


if __name__ == "__main__":
    main()
