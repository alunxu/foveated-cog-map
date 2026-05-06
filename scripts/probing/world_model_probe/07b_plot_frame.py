"""Plot frame-level Ridge + MLP probe results: linear/MLP eval R^2 + gap.
"""
import argparse, json
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = ["blind", "coarse", "foveated", "uniform", "foveated_logpolar"]
NICE = {"blind": "blind", "coarse": "coarse 1×1", "foveated": "foveated 4×4 σ=4",
         "uniform": "uniform 4×4", "foveated_logpolar": "fov-logpolar"}
COLORS = {"blind": "#5b5b5b", "coarse": "#d97a35", "foveated": "#2c7fb8",
           "uniform": "#6a51a3", "foveated_logpolar": "#7fcdbb"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ridge_json", required=True)
    ap.add_argument("--mlp_json", required=True)
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    rd = json.load(open(args.ridge_json))
    md = json.load(open(args.mlp_json))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    xs = np.arange(len(CONDITIONS))

    # Pick best alpha per condition for ridge
    lin_r2 = []
    for c in CONDITIONS:
        if c not in rd:
            lin_r2.append(np.nan); continue
        best = max((v["eval_r2_mean"] for k, v in rd[c].items()
                     if isinstance(v, dict)), default=np.nan)
        lin_r2.append(best)
    mlp_r2 = [md.get(c, {}).get("eval_r2", np.nan) for c in CONDITIONS]
    gap = [m - l for m, l in zip(mlp_r2, lin_r2)]

    bw = 0.35
    axes[0].bar(xs - bw/2, lin_r2, bw, label="linear (Ridge, best α)",
                 color="#1f77b4", edgecolor="black", linewidth=0.5)
    axes[0].bar(xs + bw/2, mlp_r2, bw, label="4-layer MLP",
                 color="#d62728", edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([NICE[c] for c in CONDITIONS], rotation=30,
                             ha="right", fontsize=9)
    axes[0].set_ylabel("agent_pos eval R² (frame-level CV, steps 250–500)",
                        fontsize=10)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_title("A — encoder bandwidth → position decode (Memory Maze)",
                       fontsize=10)
    axes[0].legend(fontsize=9, loc="upper left")
    axes[0].grid(axis="y", linestyle=":", alpha=0.3)
    for i, (l, m) in enumerate(zip(lin_r2, mlp_r2)):
        if not np.isnan(l):
            axes[0].text(i - bw/2, l + 0.01, f"{l:.2f}", ha="center", fontsize=8)
        if not np.isnan(m):
            axes[0].text(i + bw/2, m + 0.01, f"{m:.2f}", ha="center", fontsize=8)

    axes[1].bar(xs, gap, 0.55, color=[COLORS[c] for c in CONDITIONS],
                 edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([NICE[c] for c in CONDITIONS], rotation=30,
                             ha="right", fontsize=9)
    axes[1].set_ylabel("MLP − linear R² (format-shift recovery)", fontsize=10)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_title("B — non-linear gap", fontsize=10)
    axes[1].grid(axis="y", linestyle=":", alpha=0.3)
    for i, g in enumerate(gap):
        if not np.isnan(g):
            axes[1].text(i, g + 0.005, f"{g:+.2f}", ha="center", fontsize=8)

    fig.suptitle("World-model probe: frozen DINOv2-B + small LSTM, Memory Maze 9×9 (architecture-agnostic test)",
                  fontsize=11, y=1.04)
    fig.savefig(args.out_path, bbox_inches="tight", dpi=150)
    fig.savefig(args.out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()
