"""
Shared matplotlib style for all paper figures.

Import + call ``apply_paper_style()`` at the top of every figure script:

    from _style import apply_paper_style
    apply_paper_style()

This standardises font family, weight conventions, math typesetting,
and PDF font embedding so every figure has a consistent, professional
look.

Conventions (also used by individual scripts where overriden):
- font.family: Helvetica (with Arial / DejaVu Sans fallbacks)
- title / axis-label: 12 pt bold (set per-figure via fontweight='bold')
- tick label: 10-11 pt regular
- legend: 10 pt regular
- annotations on data: 10 pt bold for emphasis
- mathtext: Computer Modern (matches LaTeX paper body)
- PDF fonttype: 42 (TrueType, embeds correctly in NeurIPS submissions)
"""
from __future__ import annotations

import matplotlib as mpl


def apply_paper_style() -> None:
    """Configure matplotlib rcParams for paper-quality figures."""
    mpl.rcParams.update({
        # Font family — match the NeurIPS body text (Times Roman) directly.
        # Putting Times New Roman first (the desktop equivalent of LaTeX
        # Times) ensures figure text shares the exact same letterforms as
        # the paper body. STIX fallbacks remain for math/glyphs that Times
        # doesn't provide.
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "STIX Two Text",
            "STIXGeneral",
            "DejaVu Serif",
        ],
        "font.size": 11,
        # Bold weight remains controlled per-element via fontweight='bold'
        # on set_title / set_xlabel / set_ylabel; rcParams defaults below
        # ensure those fire automatically. Tick labels, legend, body
        # annotations stay regular weight unless individual scripts request
        # bold for emphasis (e.g., clipped-value annotations).

        # Title / label weights — bold by default for emphasis.
        # Individual scripts can override via set_title(..., fontweight=...).
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 12,

        # Tick label sizes — slightly smaller, regular weight.
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,

        # Legend.
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,

        # Math typesetting — STIX integrates with STIX Two Text body
        # so $R^2$ and "R" in surrounding text share the same letterforms.
        "mathtext.fontset": "stix",
        "mathtext.default": "regular",

        # PDF / PS font embedding — TrueType (fonttype=42) so fonts
        # embed correctly when published. Default is Type 3 which some
        # conferences (incl. NeurIPS) reject for non-embedability.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Slight aesthetic tweaks.
        "axes.grid": False,  # individual scripts opt in
        "axes.spines.top": False,
        "axes.spines.right": False,
    })