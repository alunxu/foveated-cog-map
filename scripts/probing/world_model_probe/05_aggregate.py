"""Aggregate per-condition probe results + emit a Markdown summary table.

Reads /tmp/wmprobe_results/<condition>.json files; writes:
  - /tmp/wmprobe_results/summary.json
  - /tmp/wmprobe_results/summary.md  (paste-ready table)

Decision against pre-registered hypothesis is also computed:
  - is linear R^2 monotone-non-decreasing across (blind, coarse, uniform)?
  - is gap (MLP - linear) at coarse > 2x gap at uniform?
"""
from __future__ import annotations

import argparse
import json
import os

CONDITIONS = ["blind", "coarse", "foveated", "uniform", "foveated_logpolar"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_dir", required=True)
    ap.add_argument("--out_summary", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    rows = {}
    for cond in CONDITIONS:
        path = os.path.join(args.probe_dir, f"{cond}.json")
        if os.path.exists(path):
            rows[cond] = json.load(open(path))

    summary = {"per_condition": rows, "ordering": CONDITIONS}

    # Decision logic for pre-reg
    if all(c in rows for c in ["blind", "coarse", "uniform"]):
        bl = rows["blind"]["linear_eval_r2"]
        co = rows["coarse"]["linear_eval_r2"]
        un = rows["uniform"]["linear_eval_r2"]
        summary["pre_reg"] = {
            "monotone_blind_le_coarse_le_uniform": bl <= co + 1e-6 and co <= un + 1e-6,
            "gap_at_coarse": rows["coarse"]["mlp_minus_linear_gap"],
            "gap_at_uniform": rows["uniform"]["mlp_minus_linear_gap"],
            "gap_coarse_gt_2x_uniform": (
                rows["coarse"]["mlp_minus_linear_gap"]
                > 2 * rows["uniform"]["mlp_minus_linear_gap"]
            ),
        }

    json.dump(summary, open(args.out_summary, "w"), indent=2)
    print(f"wrote {args.out_summary}")

    # Markdown table
    md = ["| condition | linear R² (eval) | MLP R² (eval) | gap | static-linear | static-MLP |"]
    md.append("|-----------|-----------------:|--------------:|----:|--------------:|-----------:|")
    for cond in CONDITIONS:
        r = rows.get(cond)
        if r is None:
            md.append(f"| {cond} | — | — | — | — | — |")
            continue
        lin = r.get("linear_eval_r2", float("nan"))
        mlp = r.get("mlp_eval_r2", float("nan"))
        gap = r.get("mlp_minus_linear_gap", float("nan"))
        st_lin = (r.get("static") or {}).get("linear_eval_r2", float("nan"))
        st_mlp = (r.get("static") or {}).get("mlp_eval_r2", float("nan"))
        md.append(
            f"| {cond} | {lin:.4f} | {mlp:.4f} | {gap:+.4f} | {st_lin if st_lin is None else f'{st_lin:.4f}'} | {st_mlp if st_mlp is None else f'{st_mlp:.4f}'} |"
        )

    if "pre_reg" in summary:
        md.append("\n### Pre-registered tests")
        md.append(f"- Monotone (blind ≤ coarse ≤ uniform): **{summary['pre_reg']['monotone_blind_le_coarse_le_uniform']}**")
        md.append(f"- gap@coarse = {summary['pre_reg']['gap_at_coarse']:+.4f}, "
                   f"gap@uniform = {summary['pre_reg']['gap_at_uniform']:+.4f}")
        md.append(f"- gap@coarse > 2× gap@uniform: **{summary['pre_reg']['gap_coarse_gt_2x_uniform']}**")

    with open(args.out_md, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"wrote {args.out_md}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
