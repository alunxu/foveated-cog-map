"""Plot predictive-coding residual: forward R^2 + mean residual norm + rank-90."""
import argparse, json
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = ["blind", "coarse", "foveated", "uniform", "foveated_logpolar"]
NICE = {"blind": "blind", "coarse": "coarse 1×1", "foveated": "foveated",
         "uniform": "uniform", "foveated_logpolar": "fov-logpolar"}
COLORS = {"blind": "#5b5b5b", "coarse": "#d97a35", "foveated": "#2c7fb8",
           "uniform": "#6a51a3", "foveated_logpolar": "#7fcdbb"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()
    d = json.load(open(args.in_path))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True)
    xs = np.arange(len(CONDITIONS))

    # Panel A: forward R^2
    r2 = [d.get(c, {}).get("forward_R2", np.nan) for c in CONDITIONS]
    axes[0].bar(xs, r2, color=[COLORS[c] for c in CONDITIONS], edgecolor="black",
                 linewidth=0.5)
    axes[0].set_title("Forward-model R²", fontsize=10)
    axes[0].set_ylabel("R² of f̂(h_t, a_t) → h_{t+1}", fontsize=9)
    axes[0].grid(axis="y", linestyle=":", alpha=0.3)

    # Panel B: mean residual norm
    norms = [d.get(c, {}).get("mean_residual_norm", np.nan) for c in CONDITIONS]
    axes[1].bar(xs, norms, color=[COLORS[c] for c in CONDITIONS], edgecolor="black",
                 linewidth=0.5)
    axes[1].set_title("Mean prediction-error magnitude", fontsize=10)
    axes[1].set_ylabel("E[||h_{t+1} − ĥ_{t+1}||]", fontsize=9)
    axes[1].grid(axis="y", linestyle=":", alpha=0.3)

    # Panel C: rank-90% of residual covariance
    rk = [d.get(c, {}).get("rank_90_pct", np.nan) for c in CONDITIONS]
    axes[2].bar(xs, rk, color=[COLORS[c] for c in CONDITIONS], edgecolor="black",
                 linewidth=0.5)
    axes[2].set_title("Residual cov rank (90% var)", fontsize=10)
    axes[2].set_ylabel("# components", fontsize=9)
    axes[2].grid(axis="y", linestyle=":", alpha=0.3)

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels([NICE[c] for c in CONDITIONS], rotation=30, ha="right",
                            fontsize=8)

    fig.suptitle("Predictive-coding residual (Rao & Ballard 1999): one-step forward MLP",
                  fontsize=11, y=1.04)
    fig.savefig(args.out_path, bbox_inches="tight", dpi=150)
    fig.savefig(args.out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()
