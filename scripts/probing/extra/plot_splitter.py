"""Plot splitter-cell analysis: fraction + eta^2 across conditions for two
trajectory features."""
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

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5), constrained_layout=True)
    xs = np.arange(len(CONDITIONS))

    # Panel A: median eta^2 (effect size of interaction)
    for ax, key, label in [(axes[0], "prev_dir", "prev-direction (8 octants)"),
                            (axes[1], "roll5_dir", "5-step rolling direction (8 octants)")]:
        eta2 = []
        for c in CONDITIONS:
            r = d.get(c, {}).get(key, {})
            eta2.append(r.get("median_eta2_split", np.nan))
        ax.bar(xs, eta2, color=[COLORS[c] for c in CONDITIONS], edgecolor="black",
                linewidth=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([NICE[c] for c in CONDITIONS], rotation=30, ha="right",
                            fontsize=9)
        ax.set_ylabel("median η² (sig units)", fontsize=10)
        ax.set_title(f"trajectory feature: {label}", fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        for i, v in enumerate(eta2):
            if not np.isnan(v):
                ax.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=8)

    fig.suptitle("Splitter cells (Wood et al. 2000): trajectory × position interaction",
                  fontsize=11, y=1.04)
    fig.savefig(args.out_path, bbox_inches="tight", dpi=150)
    fig.savefig(args.out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()
