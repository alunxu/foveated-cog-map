"""β-subspace scrubbing: paper-style 2-panel figure for §5.

(a) Linear probe R² before vs after β-scrubbing
    - Shows the ablation worked: linear R² collapses everywhere it had any
      signal to begin with (Blind 0.91 → 0.83; Coarse 0.54 → 0.28; etc.).
    - Establishes the intervention is doing what it says.

(b) MLP probe R² before vs after β-scrubbing
    - Shows the population-redundancy claim: MLP recovery is essentially
      unchanged (Blind 0.93 → 0.93; Uniform 0.57 → 0.52; etc.).
    - The linearly-readable axis is one window into the code, not its
      substrate.

Reads:  /tmp/extra_analyses/subspace_scrubbing.json
Writes: docs/manuscript/fig/fig6_scrubbing.pdf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _style import apply_paper_style  # noqa: E402

apply_paper_style()


CONDS = [
    # (key,                   short_label, colour,    marker)
    ("blind",                 "Blind",     "#444444", "o"),
    ("coarse",                "Coarse",    "#377eb8", "s"),
    ("foveated_logpolar",     "Log-polar", "#984ea3", "v"),
    ("foveated",              "Foveated",  "#e41a1c", "D"),
    ("uniform",               "Uniform",   "#4daf4a", "^"),
]


def panel_paired_bars(
    ax, data: dict,
    orig_key: str, scrub_key: str,
    orig_err_key: str, scrub_err_key: str,
    *, title: str, ylim: tuple,
    delta_clip: float | None = None,
) -> None:
    cond_order = [c[0] for c in CONDS if c[0] in data]
    cols   = [c[2] for c in CONDS if c[0] in data]
    labs   = [c[1] for c in CONDS if c[0] in data]
    orig   = [data[k][orig_key]   for k in cond_order]
    scrub  = [data[k][scrub_key]  for k in cond_order]
    orig_e = [data[k][orig_err_key]  for k in cond_order]
    scrub_e= [data[k][scrub_err_key] for k in cond_order]

    x = np.arange(len(cond_order))
    w = 0.36
    ax.bar(x - w/2, orig, w, yerr=orig_e, color=cols, alpha=0.92,
           edgecolor="black", linewidth=0.9, capsize=3,
           label="Original $\\mathbf{h}_2$", zorder=3)
    ax.bar(x + w/2, scrub, w, yerr=scrub_e, color=cols, alpha=0.42,
           edgecolor="black", linewidth=0.9, capsize=3, hatch="///",
           label="$\\beta$-scrubbed $\\mathbf{h}_2$", zorder=3)

    # Apply ylim BEFORE Δ annotation so we can clip annotations to axis area.
    ax.set_ylim(*ylim)

    # Δ annotation above whichever bar+errbar is taller, clamped to axis.
    for i, (o, s, oe, se) in enumerate(zip(orig, scrub, orig_e, scrub_e)):
        # Skip if a probe has collapsed below the threshold (panel-specific):
        if delta_clip is not None and (o < delta_clip or s < delta_clip):
            continue
        top = max(o + oe, s + se)
        # Clamp annotation y inside the visible window
        y_text = min(top + 0.04, ylim[1] - 0.06)
        delta = o - s
        sign = "+" if delta >= 0 else "−"
        ax.text(i, y_text, f"$\\Delta {sign}{abs(delta):.2f}$",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#333", zorder=5)

    ax.axhline(0, ls="-", color="grey", alpha=0.5, lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=14, fontweight="bold")
    for tick, c in zip(ax.get_xticklabels(), cols):
        tick.set_color(c)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("probe $R^2$  (5-fold episode-level CV)",
                  fontsize=15, fontweight="bold")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.set_title(title, fontsize=18, fontweight="bold",
                 loc="left", x=0.0, pad=12)


def main() -> None:
    out = Path("docs/manuscript/fig/fig6_scrubbing.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    src = Path("/tmp/extra_analyses/subspace_scrubbing.json")
    if not src.exists():
        raise SystemExit(f"missing {src}; run scripts/probing/extra/subspace_scrubbing.py")
    data = json.loads(src.read_text())

    fig, axs = plt.subplots(1, 2, figsize=(15.0, 5.0))

    # Panel (a): linear probe — establishes the ablation works where there
    # was a linear axis to ablate. Skip Δ for uniform (already collapsed).
    panel_paired_bars(
        axs[0], data,
        orig_key="linear_r2_orig",  scrub_key="linear_r2_scrub",
        orig_err_key="linear_r2_orig_std",
        scrub_err_key="linear_r2_scrub_std",
        title="(a) Linear probe: ablation works",
        ylim=(-1.0, 1.15),
        delta_clip=-0.3,
    )

    # Panel (b): MLP probe — the population-redundancy claim
    panel_paired_bars(
        axs[1], data,
        orig_key="mlp_r2_orig",  scrub_key="mlp_r2_scrub",
        orig_err_key="mlp_r2_orig_std",
        scrub_err_key="mlp_r2_scrub_std",
        title="(b) MLP probe: position survives ablation",
        ylim=(-0.05, 1.10),
    )

    # Single shared legend on the right panel
    axs[1].legend(loc="lower right", fontsize=11, frameon=False,
                  handlelength=1.6, borderpad=0.3)

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.13,
                        top=0.86, wspace=0.18)

    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200,
                bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
