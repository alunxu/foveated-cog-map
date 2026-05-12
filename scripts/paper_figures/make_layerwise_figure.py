"""
Per-layer GPS / compass probe R² across the 3 LSTM layers, for all 5
conditions, on deterministic-rollout hidden states.

The story this figure tells: rich-encoder conditions compute world-frame
position at Layer 0 (closest to the visual encoder, R^2 > 0.9) and
*discard* it through Layers 1-2 as the network composes more
task-specific abstractions; bottleneck conditions retain the spatial
code all the way to the top layer (the policy readout).

Reads:  <in-dir>/{cond}_gibson_det_analysis.json  ->  '1d_multilayer'
Writes: <out-dir>/layerwise_decay.{pdf,png}

Usage:
    python scripts/paper_figures/make_layerwise_figure.py \\
        --in-dir /scratch/izar/wxu/probing_results \\
        --out-dir docs/manuscript/fig
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# cond_key, label, colour, marker, json_stem (matches RCP analysis_results)
CONDS = [
    ("blind",             "Blind",    "#444444", "o", "blind_izar_det"),
    ("coarse",            "Coarse",   "#377eb8", "s", "coarse_det"),
    ("foveated_logpolar", "Log-polar",   "#984ea3", "v", "foveated_logpolar_det"),
    ("foveated",          "Foveated", "#e41a1c", "D", "foveated_det"),
    ("uniform",           "Uniform",  "#4daf4a", "^", "uniform_det"),
]


def load_layers(json_path: Path) -> dict[int, dict[str, float]]:
    """Return {layer: {gps_r2, compass_r2}} for hidden-state probes only."""
    d = json.loads(Path(json_path).read_text())
    out: dict[int, dict[str, float]] = {}
    for entry in d["1d_multilayer"]:
        if entry["state"] != "h":
            continue
        out[entry["layer"]] = {
            "gps_r2": entry["gps_r2"],
            "compass_r2": entry["compass_r2"],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load all conditions
    data = {}
    for cond_key, _, _, _, stem in CONDS:
        p = args.in_dir / f"{stem}_analysis.json"
        if not p.exists():
            # Backwards-compat: also try the older _gibson_det_analysis.json
            alt = args.in_dir / f"{cond_key}_gibson_det_analysis.json"
            if alt.exists():
                p = alt
            else:
                print(f"[skip] {p}")
                continue
        data[cond_key] = load_layers(p)

    if not data:
        print("No data loaded")
        return

    layers = sorted(next(iter(data.values())).keys())
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0), sharey=True)

    for ax, target, ylabel in [
        (axes[0], "gps_r2", "GPS $R^2$"),
        (axes[1], "compass_r2", "Compass $R^2$"),
    ]:
        for cond_key, label, color, marker, _stem in CONDS:
            if cond_key not in data:
                continue
            ys = [data[cond_key][layer][target] for layer in layers]
            ys_clipped = np.clip(ys, -1.5, 1.05)
            ax.plot(layers, ys_clipped, marker=marker, label=label,
                    color=color, linewidth=2, markersize=7)
        ax.axhline(0, linestyle=":", color="grey", alpha=0.5, linewidth=0.8)
        ax.set_xticks(layers)
        ax.set_xticklabels([f"Layer {ly}\n({['near visual encoder', 'middle', 'top (policy readout)'][ly]})"
                            for ly in layers], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-1.6, 1.08)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[1].legend(loc="lower left", fontsize=7, frameon=False, ncol=1)
    fig.tight_layout()

    out = args.out_dir / "figa4a_layerwise_decay.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()