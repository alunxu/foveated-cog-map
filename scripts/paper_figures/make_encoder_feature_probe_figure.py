"""
Encoder feature-map probe figure.

Probes the encoder's output feature map (post-ResNet-18, pre-LSTM)
directly for GPS / compass.  The diagnostic question this answers:
where does world-frame position information sit within the encoder
→ LSTM pipeline?

Three outcomes are interpretable:
  (i)~Encoder probes high R² for uniform / foveated but low for
      matched-compute → confirms matched's $1{\times}1$ collapse is
      where the encoder bottleneck happens.
  (ii)~Encoder probes high R² for all rich-encoder conditions
      including foveated → foveated has position in the encoder, the
      LSTM just doesn't promote it to the top layer.  H1 is then
      downstream of where the foveated bottleneck is.
  (iii)~Encoder probes high R² for matched too → matched's bottleneck
      framing needs revision; channel info alone might carry usable
      position even after spatial collapse.

Reads:  <results-dir>/<cond>_encoder_features_det.json (per condition,
         from analyze_encoder_features.py)
Writes: <out-dir>/encoder_feature_probe.{pdf,png}

Usage:
    python scripts/paper_figures/make_encoder_feature_probe_figure.py \\
        --results-dir /tmp/probing_results_local \\
        --out-dir docs/NeurIPS_2026/fig
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Encoder feature probe is meaningful only for sighted conditions
# (blind has no encoder).
CONDS = [
    ("matched",           "Matched (1×1)",    "#377eb8"),
    ("uniform",           "Uniform",          "#4daf4a"),
    ("foveated",          "Foveated (fix)",   "#e41a1c"),
    ("foveated_learned",  "Foveated (learned)",      "#ff7f00"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--clip-min", type=float, default=-0.5)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Try a single combined JSON first (matches analyze_encoder_features
    # default when --conditions is given), fall back to per-condition.
    rows = []
    combined = list(args.results_dir.glob("*encoder_features_det.json"))
    if len(combined) == 1:
        d = json.loads(combined[0].read_text())
        for cond_key, label, colour in CONDS:
            if cond_key not in d:
                continue
            entry = d[cond_key]
            rows.append({
                "label": label, "colour": colour,
                "gps_r2": entry.get("encoder_features_gps_r2_mean"),
                "gps_std": entry.get("encoder_features_gps_r2_std", 0.0),
                "comp_r2": entry.get("encoder_features_compass_r2_mean"),
                "comp_std": entry.get("encoder_features_compass_r2_std", 0.0),
                "enc_dim": entry.get("encoder_dim"),
            })
    else:
        for cond_key, label, colour in CONDS:
            p = args.results_dir / f"{cond_key}_encoder_features_det.json"
            if not p.exists():
                continue
            entry = json.loads(p.read_text()).get(cond_key, {})
            rows.append({
                "label": label, "colour": colour,
                "gps_r2": entry.get("encoder_features_gps_r2_mean"),
                "gps_std": entry.get("encoder_features_gps_r2_std", 0.0),
                "comp_r2": entry.get("encoder_features_compass_r2_mean"),
                "comp_std": entry.get("encoder_features_compass_r2_std", 0.0),
                "enc_dim": entry.get("encoder_dim"),
            })

    if not rows:
        raise RuntimeError(f"No encoder-feature JSONs in {args.results_dir}")

    # Plot: 2-panel grouped bar (GPS / Compass) with encoder dim
    # annotated under each condition label.
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=True,
                             gridspec_kw={"wspace": 0.18})
    for ax, key, std_key, title in [
        (axes[0], "gps_r2", "gps_std", "Encoder $\\to$ GPS"),
        (axes[1], "comp_r2", "comp_std", "Encoder $\\to$ Compass"),
    ]:
        x = np.arange(len(rows))
        means = np.array([(r[key] if r[key] is not None else np.nan)
                          for r in rows])
        stds  = np.array([r[std_key] for r in rows])
        m_clip = np.clip(means, args.clip_min, 1.05)
        ax.bar(x, m_clip, yerr=stds, color=[r["colour"] for r in rows],
               edgecolor="black", linewidth=0.5, capsize=2.5,
               error_kw={"linewidth": 0.6})
        for i, m in enumerate(means):
            if m < args.clip_min:
                ax.annotate(f"{m:.2f}", (i, args.clip_min + 0.06),
                            ha="center", fontsize=6.5, color="darkred")
        ax.axhline(0, color="black", linewidth=0.4)
        ax.set_ylim(args.clip_min - 0.05, 1.08)
        labels = [r["label"] + (f"\n($d={r['enc_dim']}$)"
                                if r["enc_dim"] is not None else "")
                  for r in rows]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("$R^2$ (5-fold CV)", fontsize=9)
        ax.set_title(title, fontsize=9.5)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    fig.suptitle("Encoder feature-map probe: where world-frame info sits",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = args.out_dir / f"encoder_feature_probe.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
