"""Aggregate a 2x2 encoder-scale x sensor-constraint Memory-Maze pilot.

Expected layout:
    <probe_root>/<encoder>/<condition>.json

Each JSON is produced by 04_probe.py. The aggregate reports the raw cells and
the comparison that matters for the broader-implication claim:

    small constrained encoder  vs.  larger unconstrained encoder

This is deliberately framed as a pilot decision table. It should not be read as
the final Habitat result; it tells us whether the stronger crossed experiment is
worth running at full cost.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_cell(probe_root: Path, encoder: str, condition: str):
    path = probe_root / encoder / f"{condition}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def metric(cell, name: str):
    if cell is None:
        return None
    if name == "static_linear_eval_r2":
        return (cell.get("static") or {}).get("linear_eval_r2")
    if name == "static_mlp_eval_r2":
        return (cell.get("static") or {}).get("mlp_eval_r2")
    return cell.get(name)


def fmt(x):
    if x is None:
        return "NA"
    return f"{x:.4f}"


def verdict(ok):
    if ok is None:
        return "unknown"
    return "yes" if ok else "no"


def diff(a, b):
    if a is None or b is None:
        return None
    return a - b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_root", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--small_encoder", default="dinov2_vits14")
    ap.add_argument("--large_encoder", default="dinov2_vitb14")
    ap.add_argument("--constrained_condition", default="foveated")
    ap.add_argument("--unconstrained_condition", default="uniform")
    ap.add_argument("--margin", type=float, default=0.02,
                    help="Treat values within this R2 margin as a match.")
    args = ap.parse_args()

    probe_root = Path(args.probe_root)
    encoders = [args.small_encoder, args.large_encoder]
    conditions = [args.constrained_condition, args.unconstrained_condition]

    cells = {
        enc: {cond: load_cell(probe_root, enc, cond) for cond in conditions}
        for enc in encoders
    }

    metrics = [
        "linear_eval_r2",
        "mlp_eval_r2",
        "mlp_minus_linear_gap",
        "static_linear_eval_r2",
        "static_mlp_eval_r2",
    ]

    small_constrained = cells[args.small_encoder][args.constrained_condition]
    large_unconstrained = cells[args.large_encoder][args.unconstrained_condition]

    comparison = {}
    for m in metrics:
        a = metric(small_constrained, m)
        b = metric(large_unconstrained, m)
        comparison[m] = {
            "small_constrained": a,
            "large_unconstrained": b,
            "delta": diff(a, b),
            "matches_or_beats": None if a is None or b is None else a + args.margin >= b,
        }

    sensor_effects = {}
    for enc in encoders:
        sensor_effects[enc] = {}
        for m in metrics:
            c = metric(cells[enc][args.constrained_condition], m)
            u = metric(cells[enc][args.unconstrained_condition], m)
            sensor_effects[enc][m] = diff(c, u)

    scale_effects = {}
    for cond in conditions:
        scale_effects[cond] = {}
        for m in metrics:
            s = metric(cells[args.small_encoder][cond], m)
            l = metric(cells[args.large_encoder][cond], m)
            scale_effects[cond][m] = diff(l, s)

    out = {
        "probe_root": str(probe_root),
        "encoders": encoders,
        "conditions": conditions,
        "cells": cells,
        "small_constrained_vs_large_unconstrained": comparison,
        "sensor_effect_constrained_minus_unconstrained": sensor_effects,
        "scale_effect_large_minus_small": scale_effects,
        "decision_rule": {
            "primary": (
                f"{args.small_encoder}/{args.constrained_condition} matches or beats "
                f"{args.large_encoder}/{args.unconstrained_condition} on linear_eval_r2 "
                f"within margin {args.margin}"
            ),
            "secondary": "MLP R2 should remain comparable, so the constraint is not merely deleting spatial information.",
        },
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))

    lines = []
    lines.append("# Encoder scale x sensor constraint pilot")
    lines.append("")
    lines.append("| encoder | sensor | linear R2 | MLP R2 | gap | static linear | static MLP |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for enc in encoders:
        for cond in conditions:
            cell = cells[enc][cond]
            lines.append(
                f"| {enc} | {cond} | {fmt(metric(cell, 'linear_eval_r2'))} | "
                f"{fmt(metric(cell, 'mlp_eval_r2'))} | {fmt(metric(cell, 'mlp_minus_linear_gap'))} | "
                f"{fmt(metric(cell, 'static_linear_eval_r2'))} | {fmt(metric(cell, 'static_mlp_eval_r2'))} |"
            )

    lines.append("")
    lines.append("## Key comparison")
    lines.append("")
    lines.append(
        f"`{args.small_encoder}/{args.constrained_condition}` vs "
        f"`{args.large_encoder}/{args.unconstrained_condition}`"
    )
    lines.append("")
    lines.append("| metric | small constrained | large unconstrained | delta | match/beat |")
    lines.append("|---|---:|---:|---:|---:|")
    for m in metrics:
        row = comparison[m]
        lines.append(
            f"| {m} | {fmt(row['small_constrained'])} | {fmt(row['large_unconstrained'])} | "
            f"{fmt(row['delta'])} | {row['matches_or_beats']} |"
        )

    lines.append("")
    primary_ok = comparison["linear_eval_r2"]["matches_or_beats"]
    secondary_ok = comparison["mlp_eval_r2"]["matches_or_beats"]
    lines.append(f"Primary linear-R2 match/beat: {verdict(primary_ok)}.")
    lines.append(f"Secondary MLP-R2 comparable: {verdict(secondary_ok)}.")
    if primary_ok is False:
        lines.append("Interpretation: this pilot does not support the primary scale-vs-sensor claim on the linear probe.")
    elif primary_ok is True and secondary_ok is True:
        lines.append("Interpretation: this pilot supports the primary comparison and passes the MLP sanity check.")
    else:
        lines.append("Interpretation: this pilot is inconclusive under the current decision rule.")
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out_json}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
