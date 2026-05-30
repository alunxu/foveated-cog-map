"""Combine per-condition eval JSONs into a single eval_5cond.json table.

Run after all 5 runai jobs complete.  Reads from
/scratch/wxu/habitat_checkpoints_rcp/eval_5cond/{cond}.json and writes
eval_5cond.json + a printable LaTeX-row preview to stdout.
"""
import json
import sys
from pathlib import Path

OUT_DIR = Path("/scratch/wxu/habitat_checkpoints_rcp/eval_5cond")
CONDS = ["blind", "coarse", "foveated", "uniform", "foveated_logpolar"]

combined = {}
for c in CONDS:
    p = OUT_DIR / f"{c}.json"
    if not p.exists():
        print(f"  MISSING: {p}", file=sys.stderr)
        continue
    j = json.loads(p.read_text())
    combined[c] = j["summary"]

out_path = OUT_DIR / "eval_5cond.json"
out_path.write_text(json.dumps(combined, indent=2))
print(f"Wrote {out_path}\n")

# Pretty table
print("Cond                | n    | SPL    | Succ   | steps  | path")
print("--------------------|------|--------|--------|--------|------")
for c in CONDS:
    s = combined.get(c)
    if s is None:
        print(f"{c:20s}| MISSING")
        continue
    print(f"{c:20s}| {s['n']:4d} | {s['mean_spl']:.4f} | "
          f"{s['success_rate']:.4f} | {s['mean_steps']:6.1f} | {s['mean_path_length']:.2f}")

print("\nLaTeX rows (SPL ± SEM, Succ as %):")
for c in CONDS:
    s = combined.get(c)
    if s is None:
        continue
    name = c.replace("_", "-")
    print(f"  {name:20s} & ${s['mean_spl']:.3f} \\pm {s['sem_spl']:.3f}$ & ${s['success_rate']*100:.1f}\\%$ \\\\")
