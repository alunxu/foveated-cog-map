"""
Training-dynamics figure: when does the encoder--memory race emerge?

Probes the GPS / compass R² at intermediate training checkpoints for
each of the five conditions, and plots R² vs training frames.  The
question this answers: is the H1 ordering (bottleneck > rich-encoder)
present from early training, or does it crystallise at some specific
training stage?

Reads:  <results-dir>/<cond>_gibson_ckpt<N>_det_analysis.json
        (one per (condition, checkpoint) = 22 files for 5 conds × 4-5
         ckpts each)
Writes: <out-dir>/training_dynamics.{pdf,png}

Each input file is the same format as the existing
{cond}_gibson_det_analysis.json; we read 1b_global_gps_compass and
infer training frames from the checkpoint number.

Usage:
    python scripts/paper_figures/make_training_dynamics_figure.py \\
        --results-dir /tmp/training_dynamics_local \\
        --out-dir docs/manuscript/fig
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONDS = [
    ("blind",             "Blind",            "#444444", "o"),
    ("matched",           "Coarse (1×1)",    "#377eb8", "s"),
    ("uniform",           "Uniform",          "#4daf4a", "^"),
    ("foveated",          "Foveated (fix)",   "#e41a1c", "D"),
    ("foveated_learned",  "Foveated (learned)",      "#ff7f00", "v"),
]

# Frames-per-checkpoint mapping (must match training-pipeline value).
# DD-PPO PointNav default writes a checkpoint every ~6.97M frames
# (250M frames / 36 ckpts in our config). For simplicity we use a
# coarse 7M frames/ckpt approximation; scripts can override.
FRAMES_PER_CKPT = 6.97e6

CKPT_RE = re.compile(r"_ckpt(\d+)_")


def parse_files(results_dir: Path) -> dict[str, list[dict]]:
    """Group analysis JSONs by condition. Each entry: {ckpt, frames, gps_r2, gps_std, compass_r2, compass_std, spl}."""
    out: dict[str, list[dict]] = {c[0]: [] for c in CONDS}
    for p in results_dir.iterdir():
        m = CKPT_RE.search(p.name)
        if not m or not p.name.endswith("_det_analysis.json"):
            continue
        ckpt = int(m.group(1))
        # Match condition prefix
        cond_key = None
        for c in CONDS:
            if p.name.startswith(f"{c[0]}_gibson_ckpt"):
                cond_key = c[0]
                break
        if cond_key is None:
            continue
        with open(p) as f:
            d = json.load(f)
        b = d.get("1b_global_gps_compass", {})
        out[cond_key].append({
            "ckpt": ckpt,
            "frames": ckpt * FRAMES_PER_CKPT,
            "gps_r2": b.get("gps_cv_r2_mean"),
            "gps_std": b.get("gps_cv_r2_std", 0.0),
            "compass_r2": b.get("compass_cv_r2_mean"),
            "compass_std": b.get("compass_cv_r2_std", 0.0),
        })
    for c in out:
        out[c].sort(key=lambda r: r["ckpt"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--clip-min", type=float, default=-1.5)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = parse_files(args.results_dir)
    if not any(data.values()):
        raise RuntimeError(f"No training-dynamics JSONs found in {args.results_dir}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5),
                             gridspec_kw={"wspace": 0.28})
    for ax, target_key, std_key, ylabel in [
        (axes[0], "gps_r2",     "gps_std",     "GPS $R^2$"),
        (axes[1], "compass_r2", "compass_std", "Compass $R^2$"),
    ]:
        for cond_key, label, colour, marker in CONDS:
            rows = data.get(cond_key, [])
            if not rows:
                continue
            xs = np.array([r["frames"] / 1e6 for r in rows])
            ys = np.array([r[target_key] if r[target_key] is not None else np.nan
                           for r in rows])
            errs = np.array([r[std_key] for r in rows])
            ys_clip = np.clip(ys, args.clip_min, 1.05)
            ax.errorbar(xs, ys_clip, yerr=errs, color=colour, marker=marker,
                        label=label, linewidth=1.6, markersize=5,
                        capsize=2.5, capthick=0.7, elinewidth=0.6)
        ax.axhline(0, ls=":", color="grey", alpha=0.5, lw=0.7)
        ax.set_xlabel("Training frames (M)", fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_ylim(args.clip_min - 0.05, 1.05)
        ax.tick_params(axis="both", labelsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(loc="lower right", fontsize=7.5, frameon=False)

    fig.suptitle("Training dynamics: when does the encoder--memory race emerge?",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = args.out_dir / f"training_dynamics.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
