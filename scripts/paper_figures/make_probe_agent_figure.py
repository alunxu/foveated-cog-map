"""
WJ-B: probe-agent SPL bar chart (Wijmans 2023 Fig 3B replication).

For each condition, two bars:
- agent: mean SPL of standard zero-init agent run (= AllZeroMemory).
- probe: mean SPL when probe is initialised with the trained agent's
  final memory $(h_T, c_T)$ and tasked with the SAME episode S->T.

If probe SPL > agent SPL, the agent's memory carries policy-relevant
information about how to navigate the episode (the agent took some
exploratory excursions; the probe skips them). For rich-encoder
conditions where linear probes cannot read GPS from the hidden state,
probe > agent is direct behavioural evidence that the memory is
USEFUL, sharpening the §4.5 candidate "probe-readable vs policy-used"
dissociation.

Reads:  --results-dir <dir>/{cond}.json (output of probe_agent.py)
Writes: <out-dir>/fig5b_probe_agent.pdf
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
    # (filename_key, display_label, colour)
    ("blind",      "Blind",          "#444444"),
    ("matched128", "Coarse (1$\\times$1)", "#377eb8"),
    ("uniform",    "Uniform",        "#4daf4a"),
    ("foveated",   "Foveated (fix)", "#e41a1c"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, label, colour in CONDS:
        p = args.results_dir / f"{key}.json"
        if not p.exists():
            rows.append({"key": key, "label": label, "colour": colour,
                         "agent_spl": None, "probe_spl": None,
                         "agent_succ": None, "probe_succ": None,
                         "delta": None, "n": 0})
            continue
        d = json.loads(p.read_text())
        rows.append({
            "key": key, "label": label, "colour": colour,
            "agent_spl":  d["agent"]["mean_spl"],
            "probe_spl":  d["probe_trained"]["mean_spl"],
            "agent_succ": d["agent"]["success_rate"],
            "probe_succ": d["probe_trained"]["success_rate"],
            "delta":      d["delta"]["mean_spl"],
            "n":          d["n_episodes"],
        })

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    n_cond = len(rows)
    x = np.arange(n_cond)
    bar_w = 0.36

    for i, r in enumerate(rows):
        if r["agent_spl"] is None:
            ax.text(i, 0.05, "n/a", ha="center", va="bottom",
                    fontsize=9, color="#888", style="italic")
            continue
        # Agent bar (lighter / patterned)
        ax.bar(x[i] - bar_w / 2, r["agent_spl"], bar_w,
               color=r["colour"], alpha=0.40, edgecolor="black",
               linewidth=0.6, label="Agent (zero-init)" if i == 0 else None)
        # Probe bar (full colour)
        ax.bar(x[i] + bar_w / 2, r["probe_spl"], bar_w,
               color=r["colour"], alpha=1.00, edgecolor="black",
               linewidth=0.6, label="Probe (trained-mem init)" if i == 0 else None)
        # Delta annotation above the higher bar
        higher = max(r["agent_spl"], r["probe_spl"])
        ax.text(x[i], higher + 0.03,
                f"$\\Delta{{=}}{r['delta']:+.2f}$",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=r["colour"])

    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows])
    ax.set_ylabel("SPL")
    ax.set_ylim(0, 1.05)
    ax.axhline(0, color="black", lw=0.4)
    ax.set_title("Probe-agent: trained memory vs zero memory, same task",
                 loc="left")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.3)

    plt.tight_layout()
    out = args.out_dir / "fig5b_probe_agent.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    print("\n=== Summary ===")
    for r in rows:
        if r["agent_spl"] is None: continue
        print(f"  {r['label']:20s}  agent={r['agent_spl']:.3f}  "
              f"probe={r['probe_spl']:.3f}  Δ={r['delta']:+.3f}  "
              f"(n={r['n']})")


if __name__ == "__main__":
    main()
