"""
Systematic convergence frame identification per condition.

Reads TensorBoard SPL scalar from each cond's training-time tb events.
Applies criterion: smallest frame where SPL is within EPSILON of running max
AND remains so for subsequent WINDOW_M million frames.

Outputs per-cond convergence frame + diagnostic plot.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

EPSILON = 0.02
WINDOW_M = 50  # 50M-frame plateau window required

CONDS = {
    "blind":     "/scratch/izar/wxu/habitat_checkpoints/blind_gibson",
    "matched":   "/scratch/izar/wxu/habitat_checkpoints/matched_gibson",
    "foveated":  "/scratch/izar/wxu/habitat_checkpoints/foveated_gibson_corrupt_job2836021",
    "uniform":   "/scratch/izar/wxu/habitat_checkpoints/uniform_gibson",
}


def load_spl(cond_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (frames_M, spl) from TB events in cond_dir/tb/."""
    tb_dir = Path(cond_dir) / "tb"
    events = sorted(tb_dir.glob("events.out.tfevents.*"))
    print(f"  {len(events)} event files")
    all_steps = []
    all_vals = []
    for ev in events:
        try:
            ea = event_accumulator.EventAccumulator(str(ev),
                size_guidance={"scalars": 0})
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            spl_tag = None
            for cand in ["metrics/spl", "spl", "eval_spl", "metrics/eval_spl"]:
                if cand in tags:
                    spl_tag = cand; break
            if spl_tag is None:
                # Pick first tag containing "spl" lowercase
                for t in tags:
                    if "spl" in t.lower() and "success" not in t.lower():
                        spl_tag = t; break
            if spl_tag is None:
                continue
            for scalar_event in ea.Scalars(spl_tag):
                all_steps.append(scalar_event.step)
                all_vals.append(scalar_event.value)
        except Exception as e:
            print(f"    skip {ev.name}: {e}")
    if not all_steps:
        return np.array([]), np.array([])
    arr = np.array(sorted(zip(all_steps, all_vals)))
    return arr[:, 0] / 1e6, arr[:, 1]  # frames in M


def find_convergence(frames_M: np.ndarray, spl: np.ndarray,
                     epsilon: float = EPSILON, window_M: int = WINDOW_M):
    """Find smallest frame where SPL is within epsilon of max for next window_M frames."""
    if len(spl) < 10:
        return None, None
    spl_max = float(np.max(spl))
    threshold = spl_max - epsilon
    for i, f_i in enumerate(frames_M):
        # Check all subsequent frames in [f_i, f_i + window_M]
        future_mask = (frames_M >= f_i) & (frames_M <= f_i + window_M)
        if not future_mask.any():
            continue
        future_spl = spl[future_mask]
        if (future_spl >= threshold).all() and len(future_spl) >= 5:
            return float(f_i), spl_max
    return None, spl_max


def main():
    out = {}
    for cond, ckpt_dir in CONDS.items():
        print(f"\n=== {cond} ===")
        if not Path(ckpt_dir).exists():
            print(f"  SKIP: {ckpt_dir} missing")
            continue
        frames_M, spl = load_spl(ckpt_dir)
        if len(frames_M) < 10:
            print(f"  SKIP: not enough scalar data")
            continue
        # smooth with 5-window average
        spl_smooth = np.convolve(spl, np.ones(5)/5, mode="same")
        conv_frame_M, spl_max = find_convergence(frames_M, spl_smooth)
        last_frame_M = float(frames_M[-1])
        out[cond] = {
            "max_spl_smooth": spl_max,
            "convergence_frame_M": conv_frame_M,
            "last_frame_M": last_frame_M,
            "n_scalar_points": int(len(spl)),
            "criterion": f"SPL within ±{EPSILON} of max for {WINDOW_M}M frames",
        }
        if conv_frame_M is not None:
            print(f"  convergence at {conv_frame_M:.0f}M (max SPL={spl_max:.3f}); last frame={last_frame_M:.0f}M")
        else:
            print(f"  did NOT converge by criterion; last frame={last_frame_M:.0f}M, max SPL={spl_max:.3f}")

    Path("/scratch/izar/wxu/probing_results/convergence_frames.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nwrote /scratch/izar/wxu/probing_results/convergence_frames.json")


if __name__ == "__main__":
    main()
