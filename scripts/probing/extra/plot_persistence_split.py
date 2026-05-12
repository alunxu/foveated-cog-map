"""Re-plot persistent-homology summary as TWO separate single-panel PDFs (for
3-panel composition with timescales in main text)."""
import argparse, json
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = ["blind", "coarse", "foveated_logpolar", "foveated", "uniform"]
NICE = {"blind": "Blind", "coarse": "Coarse", "foveated": "Foveated",
         "uniform": "Uniform", "foveated_logpolar": "Log-polar"}
COLORS = {"blind": "#444444", "coarse": "#377eb8", "foveated": "#e41a1c",
           "uniform": "#4daf4a", "foveated_logpolar": "#984ea3"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_betti", required=True)
    ap.add_argument("--out_wasserstein", required=True)
    args = ap.parse_args()
    d = json.load(open(args.in_path))
    xs = np.arange(len(CONDITIONS))

    # Panel: Betti-1
    fig, ax = plt.subplots(1, 1, figsize=(4.4, 3.6), constrained_layout=True)
    counts = [d["per_cond"].get(c, {}).get("dgm_1", {}).get("persistent_count_mean", np.nan)
              for c in CONDITIONS]
    stds = [d["per_cond"].get(c, {}).get("dgm_1", {}).get("persistent_count_std", 0)
            for c in CONDITIONS]
    ax.bar(xs, counts, yerr=stds, color=[COLORS[c] for c in CONDITIONS],
            edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([NICE[c] for c in CONDITIONS], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("# persistent Betti-1 features (>0.1)", fontsize=10)
    ax.set_title("Persistent loops in $\\mathbf{h}_t$ manifold", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    for i, v in enumerate(counts):
        if not np.isnan(v):
            ax.text(i, v + 5, f"{v:.0f}", ha="center", fontsize=8)
    fig.savefig(args.out_betti, bbox_inches="tight", dpi=150)
    fig.savefig(args.out_betti.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {args.out_betti}")

    # Panel: Wasserstein gap
    fig, ax = plt.subplots(1, 1, figsize=(4.4, 3.6), constrained_layout=True)
    within_means = [d["wasserstein_within"].get(c, {}).get("mean", np.nan) for c in CONDITIONS]
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
    ax.bar(xs - bw/2, within_means, bw, label="within-condition (seed)",
            color="lightgray", edgecolor="black", linewidth=0.5)
    ax.bar(xs + bw/2, across_means, bw, label="across-condition",
            color=[COLORS[c] for c in CONDITIONS], edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([NICE[c] for c in CONDITIONS], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Wasserstein-2 distance (dgm$_1$)", fontsize=10)
    ax.set_title("Across-condition gap > within-seed noise", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    fig.savefig(args.out_wasserstein, bbox_inches="tight", dpi=150)
    fig.savefig(args.out_wasserstein.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"wrote {args.out_wasserstein}")


if __name__ == "__main__":
    main()
