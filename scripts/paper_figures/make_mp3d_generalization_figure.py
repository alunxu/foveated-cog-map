"""
MP3D generalization figure (B2).

Compares key linear-probe R² values between Gibson (training distribution)
and MP3D (held-out of a dataset our agents trained on). Used in paper
§4.7 Generalisation to show the findings are not Gibson-specific.

Targets shown: GPS, compass, distance-to-goal.

Usage:
    python scripts/paper_figures/make_mp3d_generalization_figure.py \\
        --results-dir /tmp/probing_results --out-dir docs/NeurIPS_2026/fig
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CONDS = [
    ("blind",            "Blind",              "#444444"),
    ("uniform",          "Uniform",            "#4daf4a"),
    ("foveated",         "Foveated (fix)",     "#e41a1c"),
    ("foveated_learned", "Foveated (learned)", "#ff7f00"),
    ("matched",          "Matched-compute",    "#377eb8"),
]
TARGETS = [
    ("1b_global_gps_compass", "gps_r2",       "GPS"),
    ("1b_global_gps_compass", "compass_r2",   "Compass"),
    ("1c_distance_to_goal",   "r2",           "Distance-to-goal"),
]


def _get(d: dict, path: list[str]):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def load_one(results_dir: Path, cond: str, split: str, suffix: str = "") -> dict | None:
    """split = 'gibson' or 'mp3d'. suffix e.g. '_det' to load det analysis."""
    p = results_dir / f"{cond}_{split}{suffix}_analysis.json"
    if not p.exists():
        # Fall back to non-suffixed JSON so the script still runs on
        # pre-fix data (the paper caption flags such figures as stale).
        if suffix:
            p = results_dir / f"{cond}_{split}_analysis.json"
            if not p.exists():
                return None
        else:
            return None
    with open(p) as f:
        return json.load(f)


def fig_mp3d(results_dir: Path, out_path: Path, suffix: str = "") -> None:
    rows = []
    for cond, label, colour in CONDS:
        g = load_one(results_dir, cond, "gibson", suffix)
        m = load_one(results_dir, cond, "mp3d", suffix)
        row = {"cond": cond, "label": label, "colour": colour, "gibson": {}, "mp3d": {}}
        for section, key, name in TARGETS:
            if g is not None:
                v = _get(g, [section, key])
                if v is not None:
                    row["gibson"][name] = float(v)
            if m is not None:
                v = _get(m, [section, key])
                if v is not None:
                    row["mp3d"][name] = float(v)
        rows.append(row)

    # Check if we have any MP3D data at all
    any_mp3d = any(r["mp3d"] for r in rows)
    if not any_mp3d:
        print("WARNING: no MP3D data yet; skipping figure generation")
        return

    target_names = [name for _, _, name in TARGETS]
    n_tgt = len(target_names)

    fig, axes = plt.subplots(1, n_tgt, figsize=(9.5, 3.0),
                             gridspec_kw={"wspace": 0.28})

    for tgt_idx, tgt_name in enumerate(target_names):
        ax = axes[tgt_idx]
        x = np.arange(len(rows))
        w = 0.4
        g_vals = [r["gibson"].get(tgt_name, np.nan) for r in rows]
        m_vals = [r["mp3d"].get(tgt_name, np.nan) for r in rows]
        colours = [r["colour"] for r in rows]

        ax.bar(x - w / 2, g_vals, w, color=colours,
               edgecolor="black", linewidth=0.6,
               label="Gibson")
        ax.bar(x + w / 2, m_vals, w, color=colours,
               edgecolor="black", linewidth=0.6,
               hatch="///", alpha=0.7,
               label="MP3D")

        ax.set_xticks(x)
        ax.set_xticklabels([r["label"] for r in rows],
                           rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(f"{tgt_name} $R^2$", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.4)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        if tgt_idx == 0:
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Probe generalisation: Gibson-trained \u2192 MP3D validation", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")

    # Text dump
    print("\nDump for paper:")
    print(f"  {'Condition':<22} {'target':<18} {'Gibson':<10} {'MP3D':<10} {'Δ':<10}")
    for r in rows:
        for tgt_name in target_names:
            g = r["gibson"].get(tgt_name)
            m = r["mp3d"].get(tgt_name)
            if g is None or m is None:
                continue
            print(f"  {r['label']:<22} {tgt_name:<18} {g:<10.3f} {m:<10.3f} {m-g:<+10.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--suffix", type=str, default="",
                    help="Filename suffix, e.g. '_det' to read "
                         "'<cond>_<split>_det_analysis.json' instead of the "
                         "default '<cond>_<split>_analysis.json'.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_mp3d(args.results_dir, args.out_dir / "mp3d_generalization.pdf",
             suffix=args.suffix)


if __name__ == "__main__":
    main()
