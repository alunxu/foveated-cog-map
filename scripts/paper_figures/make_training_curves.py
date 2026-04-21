"""
Training-curve figure for §4.1 Training Performance.

Reads each condition's tensorboard scalars directly from
``/scratch/izar/$USER/habitat_checkpoints/{cond}/tb/events.out.tfevents.*``
and plots SPL and success rate vs.\ frame count. This is an authoritative
view of convergence: it does not rely on parsing text logs (which may
have been rotated off /scratch).

The script runs on the cluster because the event files live there; it
then copies the resulting PNG/PDF back to the paper figures directory.

Usage (local, over ssh):

    scp scripts/paper_figures/make_training_curves.py \\
        izar:/home/wxu/cs503-project/scripts/paper_figures/
    ssh izar 'sbatch --qos=normal --time=00:15:00 --account=cs-503 \\
        --gres=gpu:0 --cpus-per-task=2 --mem=8G \\
        --output=/tmp/tcurves.out --error=/tmp/tcurves.err \\
        --wrap="source /home/wxu/cs503-project/scripts/cluster/common.sh && \\
               python /home/wxu/cs503-project/scripts/paper_figures/make_training_curves.py \\
                 --ckpt-root /scratch/izar/wxu/habitat_checkpoints \\
                 --out-dir  /home/wxu/cs503-project/docs/cs503_final/fig"'

Alternatively, any local env with tensorboard + the event files mounted
works just the same.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "tensorboard not installed; "
        "install with `pip install tensorboard`\n"
    )
    raise


COND_DISPLAY = {
    "blind_gibson": ("Blind", "#444444"),
    "uniform_gibson": ("Uniform", "#4daf4a"),
    "foveated_gibson": ("Foveated (fixed)", "#e41a1c"),
    "foveated_learned_gibson": ("Foveated (learned)", "#ff7f00"),
    "matched_gibson": ("Matched-48 (deprecated)", "#377eb8"),
    "matched128_gibson": ("Matched-128", "#377eb8"),
}
COND_ORDER = [
    "blind_gibson",
    "uniform_gibson",
    "foveated_gibson",
    "foveated_learned_gibson",
    "matched128_gibson",
]


def _read_tb_series(ckpt_root: Path, cond_dir: str, tag_suffix: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (steps, values) for the first tag whose name ends with
    ``tag_suffix``. Concatenates event files from multiple training
    resumes in mtime order; when a resume restarted the step counter
    from zero (e.g. resume from latest.pth without .habitat-resume-state),
    we shift the resumed file's steps by the previous file's max step so
    the curve remains monotonic. This prevents visual spikes where
    separate event files overlap in step range."""
    paths = sorted(
        glob.glob(str(ckpt_root / cond_dir / "tb" / "events.out.tfevents.*")),
        key=os.path.getmtime,
    )
    if not paths:
        return None
    all_steps: list[int] = []
    all_vals: list[float] = []
    offset = 0
    for p in paths:
        ea = EventAccumulator(p, size_guidance={"scalars": 1_000_000})
        try:
            ea.Reload()
        except Exception as e:
            sys.stderr.write(f"[skip] {p}: {e}\n")
            continue
        tag = next(
            (t for t in ea.Tags()["scalars"] if t.lower().endswith(tag_suffix)),
            None,
        )
        if tag is None:
            continue
        evs = ea.Scalars(tag)
        if not evs:
            continue
        file_steps = [ev.step for ev in evs]
        # If this file's min step is less than previous files' max step
        # (step-counter restart), offset it.
        if all_steps and min(file_steps) < max(all_steps):
            base_offset = max(all_steps) - min(file_steps) + 1
        else:
            base_offset = 0
        for ev in evs:
            all_steps.append(ev.step + base_offset)
            all_vals.append(ev.value)
    if not all_steps:
        return None
    order = np.argsort(all_steps)
    return np.array(all_steps)[order], np.array(all_vals)[order]


def _fig(ax, ckpt_root: Path, tag_suffix: str, ylabel: str, title: str, ylim: tuple) -> None:
    for cond in COND_ORDER:
        label, colour = COND_DISPLAY.get(cond, (cond, "#888888"))
        series = _read_tb_series(ckpt_root, cond, tag_suffix)
        if series is None:
            sys.stderr.write(f"[skip] {cond}: no {tag_suffix} scalar\n")
            continue
        steps, vals = series
        # Down-sample for readability: plot every 50th point
        stride = max(1, len(steps) // 500)
        ax.plot(
            steps[::stride] / 1e6, vals[::stride],
            "-", color=colour, label=label, linewidth=1.5, alpha=0.9,
        )
    ax.set_xlabel("training frames (M)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=7, frameon=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-root", type=Path, required=True,
                   help="directory with one {cond}/tb/ subdir per condition")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharex=True)
    _fig(axes[0], args.ckpt_root, "spl",
         ylabel="SPL", title="Success-weighted path length vs. training",
         ylim=(0, 1.0))
    _fig(axes[1], args.ckpt_root, "success",
         ylabel="success rate", title="Success rate vs. training",
         ylim=(0, 1.05))
    fig.tight_layout()

    for ext in ("pdf", "png"):
        p = args.out_dir / f"training_curves.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
