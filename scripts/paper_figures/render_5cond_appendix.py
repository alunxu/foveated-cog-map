"""Render the 5-condition versions of figa8 (CKA), figa9 (population
coding: spatial info + sparse decoding), figa11 (t-SNE), figa12
(position-axis), and figa13 (PC cumulative) from the JSONs produced
by scripts/probing/extra/compute_5cond_appendix.py."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _style import apply_paper_style
apply_paper_style()

DATA = Path("/tmp")
OUT = Path("/Users/alunx/Desktop/Aluniverse/Courses/2026-Spring-CS503-Visual-Intelligence-Homework/Project/docs/manuscript/fig")

CONDS = [
    ("blind",             "Blind",     "#444444"),
    ("coarse",            "Coarse",    "#377eb8"),
    ("foveated_logpolar", "Log-polar", "#984ea3"),
    ("foveated",          "Foveated",  "#e41a1c"),
    ("uniform",           "Uniform",   "#4daf4a"),
]
COLOR = {k: c for k, _, c in CONDS}
LABEL = {k: l for k, l, _ in CONDS}


def render_cka():
    d = json.loads((DATA / "cka_5x5.json").read_text())
    M = np.array(d["matrix"])
    keys = d["conds"]
    labs = [LABEL[k] for k in keys]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=0.05)
    n = len(keys)
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labs, rotation=20, ha="right")
    ax.set_yticklabels(labs)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            txt_color = "white" if v < 0.025 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=10, color=txt_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("CKA (clipped at 0.05)")
    ax.set_title(r"Unaligned linear CKA (h$_2$)  ($n{=}30{,}000$)")
    plt.tight_layout()
    fig.savefig(OUT / "figa8_cka_heatmap.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote figa8_cka_heatmap.pdf")


def render_tsne():
    d = json.loads((DATA / "tsne_5cond.json").read_text())
    emb = np.array(d["embedding"])
    labels = np.array(d["labels"])
    dtg = np.array(d["dtg"])
    keys = d["conds"]
    n_conds = len(keys)
    fig = plt.figure(figsize=(13.5, 5.0))
    gs = fig.add_gridspec(2, n_conds + 1,
                          width_ratios=[1.4] + [1.0] * n_conds,
                          height_ratios=[1.0, 1.0],
                          wspace=0.18, hspace=0.32)
    # Joint: span rows 0+1 in col 0
    ax_joint = fig.add_subplot(gs[:, 0])
    for i, k in enumerate(keys):
        m = labels == i
        ax_joint.scatter(emb[m, 0], emb[m, 1], s=8, color=COLOR[k],
                         alpha=0.7, label=LABEL[k], linewidths=0)
    ax_joint.legend(fontsize=9, frameon=False, loc="best")
    ax_joint.set_xticks([]); ax_joint.set_yticks([])
    ax_joint.set_title("Joint t-SNE (5 conditions pooled)", fontsize=10.5,
                       fontweight="bold", pad=4)
    for s in ("top", "right", "left", "bottom"):
        ax_joint.spines[s].set_color("#999")
    # Per-condition: top row, then split across 2 rows
    for i, k in enumerate(keys):
        row = i // ((n_conds + 1) // 2)
        col = (i % ((n_conds + 1) // 2)) + 1
        ax = fig.add_subplot(gs[row, col])
        m = labels == i
        sc = ax.scatter(emb[m, 0], emb[m, 1], c=dtg[m], cmap="viridis",
                        s=6, alpha=0.85, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(LABEL[k], fontsize=10, color=COLOR[k], fontweight="bold")
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_color("#bbb")
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    fig.colorbar(sc, cax=cbar_ax).set_label("distance to goal", fontsize=9)
    plt.subplots_adjust(left=0.04, right=0.9, top=0.93, bottom=0.05)
    fig.savefig(OUT / "figa11_tsne.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote figa11_tsne.pdf")


def render_population_coding():
    info = json.loads((DATA / "spatial_info_5cond.json").read_text())
    sparse = json.loads((DATA / "sparse_decoding_5cond.json").read_text())
    keys = info["conds"]

    # Combined 2-panel for backwards compatibility
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0),
                             gridspec_kw={"wspace": 0.25})
    # (a) Per-unit spatial info distribution
    ax = axes[0]
    for k in keys:
        if k not in info["per_cond"]: continue
        si = np.array(info["per_cond"][k])
        si_sorted = np.sort(si)[::-1]
        ax.plot(np.arange(1, len(si_sorted) + 1), si_sorted,
                color=COLOR[k], lw=2.0, label=LABEL[k])
    ax.set_xscale("log")
    ax.axhline(1.0, ls="--", color="#888", lw=0.8)
    ax.text(1.1, 1.05, "1-bit threshold", fontsize=8, color="#666")
    ax.set_xlabel("unit rank (sorted by spatial info, descending)")
    ax.set_ylabel("spatial information (bits)")
    ax.set_title("(a) Per-unit spatial-information distribution",
                 fontsize=11, fontweight="bold", pad=4)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(linestyle=":", alpha=0.3)
    # (b) Sparse-vs-distributed GPS decoding
    ax = axes[1]
    k_values = sparse["k_values"]
    for k in keys:
        if k not in sparse["per_cond"]: continue
        r2 = sparse["per_cond"][k]
        ax.plot(k_values, r2, color=COLOR[k], lw=2.0, marker="o",
                markersize=5, label=LABEL[k])
    ax.set_xscale("log")
    ax.axhline(0, color="#888", lw=0.5)
    ax.set_xlabel("number of top-spatial-info units used")
    ax.set_ylabel("GPS $R^2$ (5-fold CV mean)")
    ax.set_title("(b) Sparse-vs-distributed GPS decoding",
                 fontsize=11, fontweight="bold", pad=4)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(linestyle=":", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT / "figa9_population_coding.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote figa9_population_coding.pdf")

    # Separate per-panel PDFs for 3-even-width LaTeX layout
    # Sub-panel a: per-unit spatial-information distribution
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    for k in keys:
        if k not in info["per_cond"]: continue
        si = np.array(info["per_cond"][k])
        si_sorted = np.sort(si)[::-1]
        ax.plot(np.arange(1, len(si_sorted) + 1), si_sorted,
                color=COLOR[k], lw=2.0, label=LABEL[k])
    ax.set_xscale("log")
    ax.axhline(1.0, ls="--", color="#888", lw=0.8)
    ax.text(1.1, 1.05, "1-bit threshold", fontsize=9, color="#666")
    ax.set_xlabel("unit rank (sorted by spatial info, descending)")
    ax.set_ylabel("spatial information (bits)")
    ax.set_title("Per-unit spatial-information distribution",
                 fontsize=11, fontweight="bold", pad=4)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(linestyle=":", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT / "figa9a_per_unit_info.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote figa9a_per_unit_info.pdf")

    # Sub-panel b: sparse-vs-distributed GPS decoding
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    k_values = sparse["k_values"]
    for k in keys:
        if k not in sparse["per_cond"]: continue
        r2 = sparse["per_cond"][k]
        ax.plot(k_values, r2, color=COLOR[k], lw=2.0, marker="o",
                markersize=5, label=LABEL[k])
    ax.set_xscale("log")
    ax.axhline(0, color="#888", lw=0.5)
    ax.set_xlabel("number of top-spatial-info units used")
    ax.set_ylabel("GPS $R^2$ (5-fold CV mean)")
    ax.set_title("Sparse-vs-distributed GPS decoding",
                 fontsize=11, fontweight="bold", pad=4)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(linestyle=":", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT / "figa9b_sparse_decoding.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote figa9b_sparse_decoding.pdf")


def render_position_axis():
    d = json.loads((DATA / "position_axis_5cond.json").read_text())
    keys = d["conds"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for k in keys:
        if k not in d["per_cond"]: continue
        cum_var = np.array(d["per_cond"][k]["cum_var"])
        cum_beta = np.array(d["per_cond"][k]["cum_beta_power"])
        n_pcs = np.arange(1, len(cum_beta) + 1)
        # Truncate to 200 PCs
        mask = n_pcs <= 200
        ax.plot(n_pcs[mask], cum_beta[mask], color=COLOR[k], lw=2.0,
                label=f"{LABEL[k]} pos-axis", linestyle="-")
        ax.plot(n_pcs[mask], cum_var[mask], color=COLOR[k], lw=1.4,
                linestyle="--", alpha=0.6)
    # 90% / 50% guides
    ax.axhline(0.9, ls=":", color="#666", lw=0.7)
    ax.text(195, 0.92, "90%", fontsize=8, color="#666", ha="right")
    ax.axhline(0.5, ls=":", color="#666", lw=0.7)
    ax.text(195, 0.52, "50%", fontsize=8, color="#666", ha="right")
    ax.set_xlabel("# PCs included")
    ax.set_ylabel("Cumulative power")
    ax.set_title("Position-axis lives in low-variance directions of $\\mathbf{h}_2$\n"
                 "(solid = pos-axis power, dashed = explained variance)",
                 fontsize=11, fontweight="bold", pad=4, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right",
              ncol=2, handlelength=1.6)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(linestyle=":", alpha=0.3)
    ax.set_xlim(1, 200); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(OUT / "figa12_position_axis.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote figa12_position_axis.pdf")


def render_pc_cumulative():
    d = json.loads((DATA / "pc_cumulative_5cond.json").read_text())
    keys = d["conds"]
    n_pcs = d["n_pcs"]
    # Single panel: linear readout vs # PCs (PR values inlined in caption).
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    for k in keys:
        if k not in d["per_cond"]: continue
        r2 = d["per_cond"][k]["r2_vs_pcs"]
        ax.plot(n_pcs, r2, color=COLOR[k], lw=2.0, marker="o",
                markersize=3.5, label=LABEL[k])
    ax.axhline(0, color="#888", lw=0.5)
    ax.set_xlabel("# top PCs used as probe input")
    ax.set_ylabel("Ridge probe GPS $R^2$  (5-fold ep-CV mean)")
    ax.set_title("Linear readout vs.\\ \\# PCs",
                 fontsize=11, fontweight="bold", pad=4)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(linestyle=":", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT / "figa13_pc_cumulative.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote figa13_pc_cumulative.pdf")


if __name__ == "__main__":
    render_cka()
    render_tsne()
    render_population_coding()
    render_position_axis()
    render_pc_cumulative()
