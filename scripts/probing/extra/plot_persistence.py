"""Plot persistent homology summary (Gardner 2022 precedent): Betti-1 counts +
across-vs-within Wasserstein gap.
"""
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

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), constrained_layout=True)

    # Panel A: persistent Betti-1 counts
    xs = np.arange(len(CONDITIONS))
    counts = [d["per_cond"].get(c, {}).get("dgm_1", {}).get("persistent_count_mean", np.nan)
              for c in CONDITIONS]
    stds = [d["per_cond"].get(c, {}).get("dgm_1", {}).get("persistent_count_std", 0)
            for c in CONDITIONS]
    axes[0].bar(xs, counts, yerr=stds, color=[COLORS[c] for c in CONDITIONS],
                 edgecolor="black", linewidth=0.5, capsize=3)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([NICE[c] for c in CONDITIONS], rotation=30,
                             ha="right", fontsize=9)
    axes[0].set_ylabel("# persistent Betti-1 features (>0.1)", fontsize=10)
    axes[0].set_title("Persistent loops in h_t manifold", fontsize=10)
    axes[0].grid(axis="y", linestyle=":", alpha=0.3)
    for i, v in enumerate(counts):
        if not np.isnan(v):
            axes[0].text(i, v + 5, f"{v:.0f}", ha="center", fontsize=8)

    # Panel B: cross-condition Wasserstein vs within-condition seed-to-seed
    within_means = [d["wasserstein_within"].get(c, {}).get("mean", np.nan) for c in CONDITIONS]
    # for each condition, average across-distance to other 4
    across_means = []
    for c in CONDITIONS:
        wsums = []
        for c2 in CONDITIONS:
            if c == c2: continue
            key1 = f"{c}_vs_{c2}"; key2 = f"{c2}_vs_{c}"
            v = d["wasserstein_across"].get(key1, d["wasserstein_across"].get(key2, np.nan))
            wsums.append(v)
        across_means.append(np.nanmean(wsums))
    bw = 0.4
    axes[1].bar(xs - bw/2, within_means, bw, label="within-condition (seed)",
                 color="lightgray", edgecolor="black", linewidth=0.5)
    axes[1].bar(xs + bw/2, across_means, bw, label="across-condition",
                 color=[COLORS[c] for c in CONDITIONS], edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([NICE[c] for c in CONDITIONS], rotation=30,
                             ha="right", fontsize=9)
    axes[1].set_ylabel("Wasserstein-2 distance (dgm_1)", fontsize=10)
    axes[1].set_title("Across-condition gap dominates seed noise", fontsize=10)
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(axis="y", linestyle=":", alpha=0.3)

    fig.suptitle("Persistent homology of h_t manifold (Gardner et al. 2022 precedent)",
                  fontsize=11, y=1.04)
    fig.savefig(args.out_path, bbox_inches="tight", dpi=150)
    fig.savefig(args.out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()
