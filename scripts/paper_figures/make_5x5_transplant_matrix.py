"""
Full 5×5 cross-condition transplant matrix figure.

Replaces the "we tested 3 representative pairs" caveat in §4.3 H2 with
a complete behavioural matrix of cross-condition memory-transplant
costs.  20 off-diagonal cells (5 donors × 5 recipients minus 5
self-transplants) plus the self-transplant diagonal.

Each cell value = cross-transplant SPL minus self-transplant SPL
(the isolated condition-mismatch component, with rollout-divergence
mechanics subtracted).  Strongly negative cells = highly incompatible.

Reads:  <results-dir>/<donor>_to_<recipient>.json  (default mid=30)
        <results-dir>/<donor>_to_<donor>.json      (self diag)
Writes: <out-dir>/transplant_5x5.{pdf,png}

Usage:
    python scripts/paper_figures/make_5x5_transplant_matrix.py \\
        --results-dir /tmp/transplant_local \\
        --out-dir docs/manuscript/fig
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from _style import apply_paper_style  # noqa: E402

apply_paper_style()
import numpy as np

CONDS = [
    ("blind",             "Blind"),
    ("coarse",            "Coarse"),
    ("foveated_logpolar", "Log-polar"),
    ("foveated",          "Foveated"),
    ("uniform",           "Uniform"),
]


def load_pair(results_dir: Path, donor: str, recipient: str) -> dict | None:
    """Read transplant JSON for donor->recipient (default mid=30).

    Tries both naming schemes used in the project:
      <donor>_to_<recipient>_mid30.json   (canonical 5-cond cache)
      <donor>_to_<recipient>.json         (legacy default-mid layout)
    """
    for fname in (f"{donor}_to_{recipient}_mid30.json",
                  f"{donor}_to_{recipient}.json"):
        p = results_dir / fname
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    n = len(CONDS)
    matrix = np.full((n, n), np.nan)

    # First pass: collect per-recipient (self_transplant_spl, baseline_spl)
    # from any cross JSON in which this condition is the recipient. The
    # self_transplant field always refers to the recipient driving its
    # own rollout, so it is recipient-specific not donor-specific.
    recip_self: dict[str, float] = {}
    recip_baseline: dict[str, float] = {}
    for i, (donor, _) in enumerate(CONDS):
        for j, (recip, _) in enumerate(CONDS):
            if i == j:
                continue
            pair = load_pair(args.results_dir, donor, recip)
            if pair is None:
                continue
            if recip not in recip_self:
                recip_self[recip] = pair["self_transplant"]["mean_spl"]
                recip_baseline[recip] = pair["baseline"]["mean_spl"]

    # Second pass: fill the 5×5
    for i, (donor, _) in enumerate(CONDS):
        for j, (recip, _) in enumerate(CONDS):
            if i == j:
                # Diagonal: how much SPL the recipient loses simply by
                # being transplanted to its own state mid-episode (the
                # rollout-divergence cost).
                if recip in recip_self and recip in recip_baseline:
                    matrix[i, j] = recip_self[recip] - recip_baseline[recip]
            else:
                pair = load_pair(args.results_dir, donor, recip)
                if pair is None or recip not in recip_self:
                    continue
                # Off-diagonal: cross-transplant SPL minus recipient's
                # self-transplant SPL (isolated condition-mismatch effect).
                matrix[i, j] = (pair["cross_transplant"]["mean_spl"]
                                - recip_self[recip])

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    vmax = max(0.05, np.nanmax(np.abs(matrix))) if not np.all(np.isnan(matrix)) else 0.5
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="equal")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([c[1] for c in CONDS])
    ax.set_yticklabels([c[1] for c in CONDS])
    ax.set_xlabel("Recipient")
    ax.set_ylabel("Donor")

    # Diagonal annotation
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            if np.isnan(v):
                # Light-grey hatched overlay so missing data is clearly
                # distinguishable from a near-zero coloured cell.
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor="#e8e8e8",
                                           edgecolor="none", zorder=2))
                ax.text(j, i, "n/a", ha="center", va="center",
                        fontsize=9, color="#888", zorder=3, style="italic")
                continue
            colour = "white" if abs(v) > 0.6 * vmax else "black"
            label = f"{v:+.2f}"
            if i == j:
                ax.text(j, i, f"self\n{v:+.2f}", ha="center", va="center",
                        fontsize=8.5, color=colour, style="italic")
            else:
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=10, color=colour, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("SPL gap (cross $-$ self)")
    cbar.ax.tick_params(labelsize=9)

    ax.set_title("Cross-condition memory transplants")

    plt.tight_layout()
    out = args.out_dir / "figa7a_transplant_5x5.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()