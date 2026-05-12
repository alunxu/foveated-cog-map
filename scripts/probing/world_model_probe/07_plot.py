"""Plot per-condition probe R^2 (linear + MLP + static) and the transplant matrix.

Outputs:
  fig_bandwidth_probe.pdf  — main figure (linear vs MLP vs static, per condition)
  fig_transplant.pdf       — 5x5 transplant heatmap (LSTM x encoder pairs)
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = ["blind", "coarse", "foveated", "uniform", "foveated_logpolar"]
NICE = {
    "blind": "blind",
    "coarse": "coarse 1×1",
    "foveated": "foveated 4×4 σ=4",
    "uniform": "uniform 4×4",
    "foveated_logpolar": "fov-logpolar",
}
COLORS = {
    "blind": "#5b5b5b",
    "coarse": "#d97a35",
    "foveated": "#2c7fb8",
    "uniform": "#6a51a3",
    "foveated_logpolar": "#7fcdbb",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_dir", required=True)
    ap.add_argument("--transplant_json", default=None)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = {}
    for c in CONDITIONS:
        p = os.path.join(args.probe_dir, f"{c}.json")
        if os.path.exists(p):
            rows[c] = json.load(open(p))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True)

    # --- Panel A: bar chart of linear vs MLP probe R^2 per condition ---
    ax = axes[0]
    xs = np.arange(len(CONDITIONS))
    lin_r2 = [rows.get(c, {}).get("linear_eval_r2", np.nan) for c in CONDITIONS]
    mlp_r2 = [rows.get(c, {}).get("mlp_eval_r2", np.nan) for c in CONDITIONS]
    static_lin = [(rows.get(c, {}).get("static") or {}).get("linear_eval_r2", np.nan)
                   for c in CONDITIONS]
    bw = 0.27
    ax.bar(xs - bw, static_lin, bw, label="static encoder (linear)",
            color="lightgray", edgecolor="black", linewidth=0.5)
    ax.bar(xs, lin_r2, bw, label="LSTM hidden (linear)",
            color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax.bar(xs + bw, mlp_r2, bw, label="LSTM hidden (4-layer MLP)",
            color="#d62728", edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([NICE[c] for c in CONDITIONS], rotation=30, ha="right",
                        fontsize=9)
    ax.set_ylabel("agent_pos R² (eval, steps 250–500)", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("A — encoder bandwidth → position decode (Memory Maze 9×9)",
                  fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_ylim(top=max(1.05, max([v for v in mlp_r2 if not np.isnan(v)] or [1])) * 1.05)

    # --- Panel B: gap MLP - linear vs bandwidth ---
    ax = axes[1]
    gaps = [rows.get(c, {}).get("mlp_minus_linear_gap", np.nan) for c in CONDITIONS]
    ax.bar(xs, gaps, 0.5, color=[COLORS[c] for c in CONDITIONS],
            edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([NICE[c] for c in CONDITIONS], rotation=30, ha="right",
                        fontsize=9)
    ax.set_ylabel("MLP − linear (R²)", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("B — format-shift recovery (gap)", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.suptitle(f"World-model probe (DINOv2-Base + small LSTM, frozen encoder)",
                  fontsize=11, y=1.04)
    out = os.path.join(args.out_dir, "fig_bandwidth_probe.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {out}")

    # --- Transplant heatmap (optional) ---
    if args.transplant_json and os.path.exists(args.transplant_json):
        d = json.load(open(args.transplant_json))
        for which, name in [("linear", "linear"), ("mlp", "MLP")]:
            mat = np.array(d[which], dtype=float)
            fig2, ax = plt.subplots(figsize=(4.6, 4.0), constrained_layout=True)
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-max(abs(mat).max(), 0.1),
                            vmax=max(abs(mat).max(), 0.1))
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                             color="black", fontsize=8)
            ax.set_xticks(range(5)); ax.set_xticklabels([NICE[c] for c in CONDITIONS],
                                                          rotation=30, ha="right", fontsize=8)
            ax.set_yticks(range(5)); ax.set_yticklabels([NICE[c] for c in CONDITIONS],
                                                          fontsize=8)
            ax.set_xlabel("encoder fed in (column)", fontsize=9)
            ax.set_ylabel("LSTM trained on (row)", fontsize=9)
            ax.set_title(f"Transplant {name} R² — Memory Maze", fontsize=10)
            fig2.colorbar(im, ax=ax, shrink=0.8, label="R²")
            out2 = os.path.join(args.out_dir, f"fig_transplant_{which}.pdf")
            fig2.savefig(out2, bbox_inches="tight", dpi=150)
            fig2.savefig(out2.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
            print(f"wrote {out2}")


if __name__ == "__main__":
    main()
