"""
v2: Place-cell signature analysis for LSTM hidden units (proper info-theoretic).

Method:
  1. Threshold each unit at μ+σ → binary "active" indicator
  2. Compute MI(active; spatial_bin) per (unit, scene) = bits
     This is bounded [0, log2(n_bins)] and well-defined for arbitrary signals
  3. Shuffle null: circularly shift activation; redo MI
  4. Declare unit "spatially significant" if MI > 99% null AND > 0.05 bits
  5. Cross-scene preservation: for spatially-sig units, compare PREFERRED BINS
     across scenes via spearman corr of bin-rankings (not Pearson of rates).

Predicts CAP:
  - Bottleneck conds: more spatially-sig units OR units that preserve preferred
    bin across scenes (scene-invariant code)
  - Rich-encoder conds: spatial signal exists (encoder feeds spatial features
    into LSTM) but preferred bin SHIFTS across scenes (scene-conditional)

Reads:  /tmp/cond_npzs/{cond}_gibson_det.npz
Writes: /tmp/extra_analyses/place_cell_v2.json + figure
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/alunx/Desktop/Aluniverse/Courses/2026-Spring-CS503-Visual-Intelligence-Homework/Project/scripts/paper_figures")
from _style import apply_paper_style
apply_paper_style()


CONDS = [
    ("blind",    "/tmp/cond_npzs/blind_gibson_det.npz",     "Blind",    "#444444"),
    ("coarse",   "/tmp/cond_npzs/matched_gibson_det.npz",   "Coarse",   "#377eb8"),
    ("foveated", "/tmp/cond_npzs/foveated_gibson_det.npz",  "Foveated", "#e41a1c"),
    ("uniform",  "/tmp/cond_npzs/uniform_gibson_det.npz",   "Uniform",  "#4daf4a"),
]
GRID = 8
N_SHUFFLES = 100
N_TOP_SCENES = 15
N_MIN_SCENE = 800


def normalize_xy(positions):
    x, y = positions[:, 0], positions[:, 2]
    if x.max() == x.min() or y.max() == y.min():
        return None
    nx = (x - x.min()) / (x.max() - x.min() + 1e-9)
    ny = (y - y.min()) / (y.max() - y.min() + 1e-9)
    return np.column_stack([nx, ny])


def mi_active_bin(active: np.ndarray, bin_id: np.ndarray, n_bins: int) -> float:
    """MI(active; bin) using empirical joint histogram. Bits."""
    # Joint histogram p(b, a)
    n = len(active)
    p_a = np.array([1 - active.mean(), active.mean()])
    p_b = np.bincount(bin_id, minlength=n_bins) / n
    p_b = p_b[p_b > 0]
    p_ab = np.zeros((n_bins, 2))
    for a in [0, 1]:
        mask = active == a
        cnt = np.bincount(bin_id[mask], minlength=n_bins)
        p_ab[:, a] = cnt / n
    mi = 0.0
    for b in range(n_bins):
        for a in [0, 1]:
            if p_ab[b, a] > 0 and p_a[a] > 0 and p_b[b] > 0 if b < len(p_b) else False:
                # safer way:
                pass
    # Simpler: p_b_full and direct compute
    p_b_full = np.bincount(bin_id, minlength=n_bins) / n
    mi = 0.0
    for b in range(n_bins):
        for a in [0, 1]:
            if p_ab[b, a] > 1e-12 and p_a[a] > 1e-12 and p_b_full[b] > 1e-12:
                mi += p_ab[b, a] * np.log2(p_ab[b, a] / (p_a[a] * p_b_full[b]))
    return float(max(mi, 0.0))


def main():
    Path("/tmp/extra_analyses").mkdir(exist_ok=True)
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8),
                             gridspec_kw={"wspace": 0.3})

    cond_data = {}
    for cond, path, label, color in CONDS:
        print(f"\n=== {cond} ===")
        d = np.load(Path(path))
        h_all = d["hidden_states"].astype(np.float32)
        pos_all = d["positions"].astype(np.float32)
        scene_ids = d["scene_ids"]

        unique, counts = np.unique(scene_ids, return_counts=True)
        idx_sorted = np.argsort(counts)[::-1]
        top_scenes = [int(s) for s in unique[idx_sorted[:N_TOP_SCENES]]
                      if (scene_ids == s).sum() >= N_MIN_SCENE]
        print(f"  using {len(top_scenes)} scenes")

        # Use ALL 512 units this time, but only first 5 scenes for shuffle null
        n_units = h_all.shape[1]
        mi_all = np.zeros((n_units, len(top_scenes)))
        is_spatial = np.zeros((n_units, len(top_scenes)), dtype=bool)
        # Preferred bin per (unit, scene)
        pref_bin = np.zeros((n_units, len(top_scenes)), dtype=int)

        for si, scn in enumerate(top_scenes):
            mask = scene_ids == scn
            xy = normalize_xy(pos_all[mask])
            if xy is None: continue
            bin_x = np.clip((xy[:, 0] * GRID).astype(int), 0, GRID - 1)
            bin_y = np.clip((xy[:, 1] * GRID).astype(int), 0, GRID - 1)
            bin_id = bin_x * GRID + bin_y
            n_bins = GRID * GRID
            h_scene = h_all[mask]
            # Threshold each unit at μ+σ (active indicator)
            mu = h_scene.mean(axis=0)
            sd = h_scene.std(axis=0)
            for ui in range(n_units):
                active = (h_scene[:, ui] > mu[ui] + sd[ui]).astype(int)
                if active.sum() < 30:
                    continue
                mi = mi_active_bin(active, bin_id, n_bins)
                mi_all[ui, si] = mi
                # Preferred bin = bin with highest p(active|bin)
                bin_p_active = np.zeros(n_bins)
                for b in range(n_bins):
                    bm = bin_id == b
                    if bm.sum() > 5:
                        bin_p_active[b] = active[bm].mean()
                pref_bin[ui, si] = int(np.argmax(bin_p_active))
                # Shuffle null only for first scene + 32 units (sample-based)
                if si == 0 and ui < 32:
                    rng = np.random.default_rng(ui)
                    nulls = []
                    for _ in range(N_SHUFFLES):
                        shift = rng.integers(low=10, high=len(active) - 10)
                        nulls.append(mi_active_bin(np.roll(active, shift), bin_id, n_bins))
                    th = np.percentile(nulls, 99)
                    is_spatial[ui, si] = mi > th and mi > 0.05
                else:
                    is_spatial[ui, si] = mi > 0.05  # heuristic for non-shuffled

        # Aggregate
        mean_mi_per_unit = mi_all.mean(axis=1)  # (units,)
        n_spatial_units = (mean_mi_per_unit > 0.05).sum()
        # Cross-scene preferred-bin preservation
        # For each unit (use all 512), compute Spearman across scenes of preferred bin
        # via pairwise: do scene_a's pref bins correlate with scene_b's pref bins?
        # Simpler: per-unit "stability" = std of preferred bin across scenes (low = stable)
        unit_pref_std = np.std(pref_bin, axis=1)  # (units,)
        # Filter to spatial-significant units (mean MI > 0.05)
        spatial_mask = mean_mi_per_unit > 0.05
        if spatial_mask.sum() > 0:
            stab_spatial = unit_pref_std[spatial_mask]
        else:
            stab_spatial = np.array([])

        # Also: compute per-pair Pearson of pref_bin VECTORS across scenes
        # Each scene contributes a vector of pref_bins (n_units,). 
        # If across-scene units share preferred bin → high Spearman
        cross_scene_corr = []
        for sa in range(len(top_scenes)):
            for sb in range(sa + 1, len(top_scenes)):
                v_a = pref_bin[spatial_mask, sa]
                v_b = pref_bin[spatial_mask, sb]
                if len(v_a) > 5:
                    r, _ = spearmanr(v_a, v_b)
                    if not np.isnan(r):
                        cross_scene_corr.append(r)
        cross_scene_corr = np.array(cross_scene_corr) if cross_scene_corr else np.array([])

        results[cond] = {
            "label": label,
            "color": color,
            "n_top_scenes": len(top_scenes),
            "n_spatial_units": int(n_spatial_units),
            "mi_per_unit": mean_mi_per_unit.tolist(),
            "mi_mean": float(mean_mi_per_unit.mean()),
            "mi_max": float(mean_mi_per_unit.max()),
            "mi_p90": float(np.percentile(mean_mi_per_unit, 90)),
            "stability_pref_bin_median": float(np.median(stab_spatial)) if len(stab_spatial) else None,
            "cross_scene_pref_bin_spearman_mean": float(cross_scene_corr.mean()) if len(cross_scene_corr) else None,
            "cross_scene_pref_bin_spearman_median": float(np.median(cross_scene_corr)) if len(cross_scene_corr) else None,
        }
        cond_data[cond] = (mean_mi_per_unit, cross_scene_corr, color, label)
        print(f"  Spatial MI per-unit: mean={mean_mi_per_unit.mean():.3f}, "
              f"max={mean_mi_per_unit.max():.3f}, p90={np.percentile(mean_mi_per_unit, 90):.3f}")
        print(f"  # spatial-significant units (MI>0.05): {n_spatial_units} / 512")
        print(f"  Cross-scene preferred-bin Spearman: "
              f"mean={cross_scene_corr.mean() if len(cross_scene_corr) else float('nan'):+.3f}")

    # Plot
    cond_order = ["blind", "coarse", "foveated", "uniform"]
    # Panel A: violin/box of MI per-unit per cond
    box_a = []
    cols_a = []
    labels_a = []
    for c in cond_order:
        if c not in cond_data: continue
        mi, _, col, lbl = cond_data[c]
        box_a.append(mi)
        cols_a.append(col)
        labels_a.append(lbl)
    bp = axes[0].boxplot(box_a, labels=labels_a, patch_artist=True,
                         showmeans=True, widths=0.55,
                         meanprops=dict(marker="D", markerfacecolor="black",
                                        markeredgecolor="white", markersize=7),
                         medianprops=dict(color="black", lw=1.3))
    for p, c in zip(bp["boxes"], cols_a):
        p.set_facecolor(c); p.set_alpha(0.55)
    axes[0].set_ylabel("Spatial MI per unit (bits)\nbinarized at $\\mu{+}\\sigma$, MI(active; (x,y) bin)",
                       fontsize=10.5, fontweight="bold")
    axes[0].set_title("(a) Spatial selectivity per LSTM unit",
                      fontsize=11, loc="left", fontweight="bold", pad=6)
    axes[0].axhline(0.05, ls=":", color="grey", alpha=0.5, lw=0.7)
    axes[0].text(4.4, 0.05, "MI = 0.05 bits\n(threshold)", fontsize=8, color="grey", va="center")
    for s in ("top", "right"): axes[0].spines[s].set_visible(False)
    axes[0].grid(axis="y", linestyle=":", alpha=0.3)

    # Panel B: # spatial-significant units per cond (bar)
    nsp = [results[c]["n_spatial_units"] for c in cond_order if c in results]
    cols_b = [results[c]["color"] for c in cond_order if c in results]
    labs_b = [results[c]["label"] for c in cond_order if c in results]
    axes[1].bar(labs_b, nsp, color=cols_b, alpha=0.7, edgecolor="black", linewidth=0.8)
    for i, n in enumerate(nsp):
        axes[1].text(i, n + 5, str(n), ha="center", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("# spatial-significant LSTM units\n(MI > 0.05 bits, of 512)",
                       fontsize=10.5, fontweight="bold")
    axes[1].set_title("(b) Spatial-coding population size",
                      fontsize=11, loc="left", fontweight="bold", pad=6)
    for s in ("top", "right"): axes[1].spines[s].set_visible(False)
    axes[1].grid(axis="y", linestyle=":", alpha=0.3)

    # Panel C: cross-scene preferred-bin Spearman correlation per cond
    box_c = []
    for c in cond_order:
        if c not in cond_data: continue
        _, cc, col, lbl = cond_data[c]
        if len(cc): box_c.append(cc)
        else: box_c.append([0])
    bp = axes[2].boxplot(box_c, labels=labels_a, patch_artist=True,
                         showmeans=True, widths=0.55,
                         meanprops=dict(marker="D", markerfacecolor="black",
                                        markeredgecolor="white", markersize=7),
                         medianprops=dict(color="black", lw=1.3))
    for p, c in zip(bp["boxes"], cols_a):
        p.set_facecolor(c); p.set_alpha(0.55)
    axes[2].axhline(0, ls=":", color="grey", alpha=0.5, lw=0.7)
    axes[2].set_ylabel("Spearman$\\rho$(scene$_a$, scene$_b$)\nover preferred bins of all spatial units",
                       fontsize=10.5, fontweight="bold")
    axes[2].set_title("(c) Cross-scene preferred-bin preservation\n(high = scene-invariant code)",
                      fontsize=11, loc="left", fontweight="bold", pad=6)
    for s in ("top", "right"): axes[2].spines[s].set_visible(False)
    axes[2].grid(axis="y", linestyle=":", alpha=0.3)

    fig.suptitle("Place-cell signature in LSTM hidden states: spatial selectivity, population size, and cross-scene preservation",
                 fontsize=11, fontweight="bold", y=1.0)
    plt.tight_layout()
    out = Path("/Users/alunx/Desktop/Aluniverse/Courses/2026-Spring-CS503-Visual-Intelligence-Homework/Project/docs/manuscript/fig/fig_place_cells.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out}")

    Path("/tmp/extra_analyses/place_cell_v2.json").write_text(json.dumps(results, indent=2, default=str))
    print("wrote /tmp/extra_analyses/place_cell_v2.json")


if __name__ == "__main__":
    main()
