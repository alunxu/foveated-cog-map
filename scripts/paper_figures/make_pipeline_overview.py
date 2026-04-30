"""
Paper-level pipeline overview (Figure 1, single self-contained schematic).

Two-row composition:

    Row 1 -- Architecture
        4 condition thumbnails (Blind / Coarse / Foveated / Uniform), stacked
        vertically, each rasterised from `fig1_setup_*.pdf` via PyMuPDF and
        framed in its palette colour, with a one-word italic Times label.
            -> Coarse / Foveated / Uniform converge into ResNet-18
               (5-block tapered stack with 2 skip arcs)
            -> per-condition encoder-output stickers (1x1 / 8x8 / 8x8)
            -> sensor stack (GPS / compass / target / action)
            -> LSTM (3 cells L0/L1/L2 with recurrent self-loops; L0 with
               simplified sigma/sigma/tanh gate cluster)
            -> policy head (pi)
            -> h_2 vector (column of squares, prominent right-end anchor)
        Blind bypasses ResNet-18 with a separate arc that lands on the
        sensor-stack/L0 join region.
        train/eval dataset annotations sit above and below this row.

    Row 2 -- Three method-finding columns, each a self-contained story:
        Linear probes           Geometry                Intervention
        (regression-line glyph) (3x3 mini heatmap)      (donor/recipient swatches)
            v                       v                       v
        H1 bar chart            H2 4x4 heatmap          probe-vs-policy 2x2

A single down-arrow from row 1's h_2 vector lands on a horizontal h_2 strip
that distributes to all 3 row-2 columns.

Aesthetic inherits from `_style.py` (Times serif, STIX math, tight weights)
and the four-condition colour palette
    blind=#444444, coarse=#377eb8, foveated=#e41a1c, uniform=#4daf4a
already used by every other paper figure.

Writes: docs/manuscript/fig/fig_pipeline_overview.{pdf}

Run from the project root:
    python scripts/paper_figures/make_pipeline_overview.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import fitz  # PyMuPDF: rasterise condition thumbnail PDFs at >= 300 dpi.
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)

sys.path.insert(0, str(Path(__file__).parent))
from _style import apply_paper_style  # noqa: E402

apply_paper_style()

# Reaffirm Times-family serif so figure text matches the manuscript body
# (manuscript.sty implicitly loads `times`). STIX provides math glyphs.
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIX Two Text", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "mathtext.rm": "serif",
    "mathtext.it": "serif:italic",
    "mathtext.bf": "serif:bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT_DIR = Path("docs/manuscript/fig")

# Per-condition thumbnail PDFs (rasterised via PyMuPDF for the left column).
COND_THUMB_PDFS = {
    "Blind":    OUT_DIR / "fig1_setup_blind.pdf",
    "Coarse":   OUT_DIR / "fig1_setup_matched.pdf",
    "Foveated": OUT_DIR / "fig1_setup_foveated_fix.pdf",
    "Uniform":  OUT_DIR / "fig1_setup_uniform.pdf",
}

# Per-condition palette (matches every other paper figure).
COND_COLORS = {
    "Blind":    "#444444",
    "Coarse":   "#377eb8",
    "Foveated": "#e41a1c",
    "Uniform":  "#4daf4a",
}
COND_ORDER = ["Blind", "Coarse", "Foveated", "Uniform"]

# Light versions for backgrounds / face colours that should not over-saturate.
COND_LIGHT = {
    "Blind":    "#bdbdbd",
    "Coarse":   "#cfe1f2",
    "Foveated": "#f7c2c2",
    "Uniform":  "#cce8c4",
}

# Encoder spatial-output sticker text per VISUAL condition (Blind has no
# encoder pass, so it is omitted).
ENC_OUT = {
    "Coarse":   r"$1{\times}1$",
    "Foveated": r"$8{\times}8$",
    "Uniform":  r"$8{\times}8$",
}

# Generic greys / accents.
GREY_LINE = "#555555"
GREY_FILL = "#ececec"
TEXT_COL = "#1a1a1a"
HIGHLIGHT = "#fff4d4"  # h_2 box / findings highlight tint
DARK = "#222222"

# Method-family accent colours (row 2 column titles + finding panel borders).
METHOD_ACCENTS = {
    "probes":       "#1f6dad",
    "geometry":     "#7d3a9f",
    "intervention": "#a14a1f",
}

# Grid debug toggle: if True, draw faint red verticals at unit multiples.
DEBUG_GRID = False
# Grid base unit (inches): stage centres land on integer multiples of this.
UNIT = 0.5

# Subtle drop shadow for major groupings (row 1 architecture, row 2 columns).
SHADOW = pe.SimplePatchShadow(
    offset=(0.5, -0.5), shadow_rgbFace="lightgrey", alpha=0.25,
)


# ────────────────────────── helpers ──────────────────────────────────────
def rounded_box(ax, x, y, w, h, text, *,
                facecolor="white", edgecolor=DARK, lw=0.6,
                fontsize=8.5, weight="normal", rad=0.04, text_color=None,
                zorder=2, shadow=False):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={rad}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
        zorder=zorder,
    )
    if shadow:
        patch.set_path_effects([SHADOW, pe.Normal()])
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center",
                fontsize=fontsize, fontweight=weight,
                color=text_color or TEXT_COL, zorder=zorder + 1)


def arrow(ax, p_from, p_to, *, lw=0.45, color=GREY_LINE,
          style="-|>", connectionstyle=None, mutation_scale=8, zorder=4):
    a = FancyArrowPatch(
        p_from, p_to,
        arrowstyle=style, mutation_scale=mutation_scale,
        color=color, lw=lw, shrinkA=2, shrinkB=2,
        connectionstyle=connectionstyle,
        zorder=zorder,
    )
    ax.add_patch(a)


# ────────────────────────── thumbnail rasterisation ──────────────────────
def _rasterise_pdf(pdf_path: Path, dpi: int = 320) -> np.ndarray:
    """Open a single-page PDF with PyMuPDF and return an RGB(A) ndarray.

    Used for the 4 condition thumbnails on the left edge of row 1.
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = arr.reshape(pix.height, pix.width, pix.n)
    doc.close()
    return arr


def draw_condition_column(ax, x_left, y_center, *, col_w=0.78, col_h=4.40):
    """Stack of 4 thumbnail axes (Blind / Coarse / Foveated / Uniform) at
    the LEFT edge of row 1. Each thumbnail gets a coloured frame and an
    italic Times label below it.

    Returns dict mapping condition -> (right_anchor_x, right_anchor_y) for
    arrow takeoff.
    """
    n = 4
    # Thumbnail sticker geometry (inches).
    thumb_w = 0.65
    thumb_h = 0.65
    label_h = 0.18  # vertical room reserved below each thumbnail for label
    cell_h = thumb_h + label_h
    gap = (col_h - n * cell_h) / (n - 1)
    if gap < 0.05:
        gap = 0.05

    top_y = y_center + col_h / 2  # top of the column (where Blind starts)
    cx = x_left + col_w / 2

    anchors: dict[str, tuple[float, float]] = {}
    order = ["Blind", "Coarse", "Foveated", "Uniform"]
    for i, cond in enumerate(order):
        thumb_top = top_y - i * (cell_h + gap)
        thumb_bot = thumb_top - thumb_h
        thumb_left = cx - thumb_w / 2
        thumb_right = cx + thumb_w / 2

        # Rasterise thumbnail and embed via imshow on an inset axes-like
        # rectangle. We use a transient `ax.imshow` with `extent=` since
        # the parent `ax` is in inch-coordinates (set_xlim/set_ylim equal
        # to FIG_W/FIG_H + aspect=equal), which lets us drop the image
        # directly without creating a child Axes.
        img = _rasterise_pdf(COND_THUMB_PDFS[cond], dpi=320)
        ax.imshow(
            img,
            extent=(thumb_left, thumb_right, thumb_bot, thumb_top),
            interpolation="bilinear",
            zorder=4,
            aspect="auto",
        )
        # Coloured frame around the thumbnail (4 - 8 pt stroke -> 0.55 lw).
        frame = Rectangle(
            (thumb_left, thumb_bot), thumb_w, thumb_h,
            facecolor="none",
            edgecolor=COND_COLORS[cond],
            linewidth=0.85,
            zorder=5,
        )
        frame.set_path_effects([SHADOW, pe.Normal()])
        ax.add_patch(frame)

        # Italic Times label just below in the condition's colour.
        ax.text(
            cx, thumb_bot - 0.05, cond,
            ha="center", va="top",
            fontsize=8.5, color=COND_COLORS[cond],
            style="italic", fontweight="bold",
            zorder=6,
        )

        anchors[cond] = (thumb_right, (thumb_top + thumb_bot) / 2)

    return anchors


# ────────────────────────── glyph primitives ─────────────────────────────
def glyph_location_pin(ax, cx, cy, size, color="#444"):
    """Tiny location-pin (teardrop + dot)."""
    r = size * 0.45
    head = Circle((cx, cy + size * 0.15), r,
                  facecolor=color, edgecolor=DARK, linewidth=0.4, zorder=3)
    ax.add_patch(head)
    tail = Polygon(
        [(cx - r * 0.55, cy + size * 0.05),
         (cx + r * 0.55, cy + size * 0.05),
         (cx, cy - size * 0.45)],
        closed=True, facecolor=color, edgecolor=DARK, linewidth=0.4,
        zorder=3,
    )
    ax.add_patch(tail)
    inner = Circle((cx, cy + size * 0.18), r * 0.35,
                   facecolor="white", edgecolor=color, linewidth=0.3,
                   zorder=4)
    ax.add_patch(inner)


def glyph_compass_rose(ax, cx, cy, size, color="#444"):
    """4-arm compass rose (N/E/S/W diamond points)."""
    r = size * 0.5
    ring = Circle((cx, cy), r * 1.05,
                  facecolor="white", edgecolor=color, linewidth=0.4,
                  zorder=3)
    ax.add_patch(ring)
    for dx, dy in [(0, r), (r, 0), (0, -r), (-r, 0)]:
        flank = r * 0.18
        if dx == 0:
            poly = Polygon(
                [(cx, cy), (cx - flank, cy + dy * 0.35),
                 (cx + dx, cy + dy), (cx + flank, cy + dy * 0.35)],
                closed=True, facecolor=color, edgecolor=DARK,
                linewidth=0.3, zorder=4,
            )
        else:
            poly = Polygon(
                [(cx, cy), (cx + dx * 0.35, cy - flank),
                 (cx + dx, cy + dy), (cx + dx * 0.35, cy + flank)],
                closed=True, facecolor=color, edgecolor=DARK,
                linewidth=0.3, zorder=4,
            )
        ax.add_patch(poly)
    ax.add_patch(Circle((cx, cy), r * 0.10, facecolor=DARK, zorder=5))


def glyph_target(ax, cx, cy, size, color="#444"):
    """Concentric-rings target glyph."""
    r = size * 0.5
    for k, frac in enumerate([1.0, 0.65, 0.30]):
        ax.add_patch(Circle(
            (cx, cy), r * frac,
            facecolor="white" if k % 2 == 0 else color,
            edgecolor=color, linewidth=0.4, zorder=3 + k,
        ))
    ax.add_patch(Circle((cx, cy), r * 0.10, facecolor=color, zorder=6))


def glyph_action(ax, cx, cy, size, color="#444"):
    """Square with an arrow inside (a = previous action)."""
    s = size * 0.85
    half = s / 2
    sq = Rectangle(
        (cx - half, cy - half), s, s,
        facecolor="white", edgecolor=color, linewidth=0.5, zorder=3,
    )
    ax.add_patch(sq)
    a = FancyArrowPatch(
        (cx - s * 0.30, cy), (cx + s * 0.30, cy),
        arrowstyle="-|>", mutation_scale=6, lw=0.7, color=color, zorder=4,
    )
    ax.add_patch(a)


def _iso_prism(ax, x, y, w, h, *, depth, face_color, top_color, side_color,
               lw=0.5, edge_color=DARK, zorder=3):
    """Draw a 3D-isometric rectangular prism: front face + top face + right
    side face, offset by ``depth`` (in inches) up-and-to-the-right.

    PlotNeuralNet / NN-SVG-style: front face is rectangular; top and right
    faces are parallelograms tilted at the same depth offset. Returns the
    front-face geometry as (x, y, w, h) for downstream anchoring.
    """
    dx = depth
    dy = depth * 0.55  # isometric tilt ratio (~30deg)
    # Top face: parallelogram from (x, y+h) to (x+dx, y+h+dy) etc.
    top_face = Polygon(
        [(x, y + h), (x + dx, y + h + dy),
         (x + w + dx, y + h + dy), (x + w, y + h)],
        closed=True, facecolor=top_color, edgecolor=edge_color,
        linewidth=lw, zorder=zorder,
    )
    ax.add_patch(top_face)
    # Right face.
    right_face = Polygon(
        [(x + w, y), (x + w + dx, y + dy),
         (x + w + dx, y + h + dy), (x + w, y + h)],
        closed=True, facecolor=side_color, edgecolor=edge_color,
        linewidth=lw, zorder=zorder,
    )
    ax.add_patch(right_face)
    # Front face.
    front_face = Rectangle(
        (x, y), w, h,
        facecolor=face_color, edgecolor=edge_color,
        linewidth=lw, zorder=zorder + 1,
    )
    ax.add_patch(front_face)
    return (x, y, w, h)


def resnet18_stack(ax, x, y, w, h, *, lw=0.55, shadow=False):
    """Morphologically accurate ResNet-18 (He et al. 2016, Table 1).

    Renders:
      - conv1 stem (7x7, 64ch, 112x112 -> 56x56 after maxpool)
      - 4 conv stages (conv2_x..conv5_x), each with 2 BasicBlocks
        (channels 64/128/256/512, spatial 56/28/14/7)
      - Each stage drawn as an isometric 3D prism whose width tapers
        with spatial dim and whose face-darkness tracks channel count
      - 2 skip-connection arcs PER STAGE (2 BasicBlocks each), arching
        over the prism (the canonical ResNet residual signature)
      - Tiny "C / S^2" labels under each stage
      - "ResNet-18" title label below the bottom-most label row

    Returns right-edge midpoint of conv5_x (input anchor for FC head).
    """
    # Title above the stack.
    ax.text(
        x + w / 2, y + h + 0.16,
        "ResNet-18 visual encoder",
        ha="center", va="bottom",
        fontsize=8.5, fontweight="bold", color=TEXT_COL,
    )

    # 5 stages: stem (conv1), then conv2_x..conv5_x.
    stage_names = ["conv1", "conv2_x", "conv3_x", "conv4_x", "conv5_x"]
    stage_labels = [
        r"64 / $112^{2}$",
        r"64 / $56^{2}$",
        r"128 / $28^{2}$",
        r"256 / $14^{2}$",
        r"512 / $7^{2}$",
    ]
    n_blocks_per_stage = [0, 2, 2, 2, 2]  # conv1 stem has 0 residual pairs
    n_stages = 5

    # Front-face heights taper with spatial dimension (PlotNeuralNet style:
    # spatial 112->56->28->14->7 -> heights ~ log-scale).
    spatial_h_norm = np.array([1.00, 0.78, 0.62, 0.48, 0.36])
    face_h_max = h * 0.62
    face_heights = spatial_h_norm * face_h_max

    # Front-face widths grow with channel count (64->64->128->256->512):
    # use sqrt to balance figure budget.
    chan = np.array([64, 64, 128, 256, 512], dtype=float)
    chan_w_norm = np.sqrt(chan / chan.max())  # 0.35..1.00
    face_w_max = (w * 0.94) / (n_stages * 1.08)  # leave room for gaps
    face_widths = chan_w_norm * face_w_max
    # The stem (conv1) is visually narrower than conv2_x to suggest it's
    # NOT a residual stage (no skip arcs).
    face_widths[0] *= 0.85

    # Face colours: progressively deeper steel-blue with channel count.
    face_palette = [
        ("#eef2f7", "#f5f8fc", "#dde4ee"),
        ("#dde7f3", "#e8eff8", "#c6d3e3"),
        ("#c6d6ec", "#d6e1f1", "#a8bcd4"),
        ("#a8c0e0", "#bcd0e8", "#8aa6c2"),
        ("#85a5d0", "#a0bcd9", "#6685b0"),
    ]

    # Layout: lay prisms horizontally, baseline aligned to the row1 centre.
    # Centre the ENSEMBLE within (x, x+w).
    inter_gap = h * 0.05  # small separator
    total_widths_with_depth = []
    depth_per_stage = []
    for i in range(n_stages):
        d = max(0.05, face_heights[i] * 0.18)  # depth proportional to height
        depth_per_stage.append(d)
        total_widths_with_depth.append(face_widths[i] + d)
    total_used = sum(total_widths_with_depth) + inter_gap * (n_stages - 1)
    bx = x + (w - total_used) / 2

    # Vertical baseline: centre prisms vertically within (y, y+h*0.62).
    band_top = y + h * 0.86
    centres_for_arcs = []
    fronts = []  # store (x, y, w, h) of each front face for skip arcs

    for i in range(n_stages):
        fw = face_widths[i]
        fh = face_heights[i]
        d = depth_per_stage[i]
        fy = band_top - fh - 0.05  # bottom-align prisms near top of canvas
        face_col, top_col, side_col = face_palette[i]
        # ResNet stem (i=0): single prism.
        # Residual stages (i>=1): single prism with internal "block" hint.
        front = _iso_prism(
            ax, bx, fy, fw, fh,
            depth=d,
            face_color=face_col, top_color=top_col, side_color=side_col,
            lw=lw, edge_color=DARK, zorder=3,
        )
        if shadow and i == 0:
            # Shadow on the stem only (don't double up).
            sh = Rectangle((bx, fy), fw, fh,
                           facecolor="none", edgecolor="none")
            sh.set_path_effects([SHADOW, pe.Normal()])
            ax.add_patch(sh)
        fronts.append(front)

        # For residual stages (conv2_x..conv5_x), draw 2 inner block hints
        # (thin horizontal dividers) to suggest the 2 BasicBlocks inside.
        if i >= 1:
            n_b = n_blocks_per_stage[i]
            for k in range(1, n_b):
                yk = fy + fh * (k / n_b)
                ax.plot(
                    [bx, bx + fw], [yk, yk],
                    color=DARK, lw=0.35, alpha=0.55, zorder=4.5,
                )

        # Channel/spatial label under each prism (small, italic Times).
        ax.text(
            bx + fw / 2 + d / 2, fy - 0.06, stage_labels[i],
            ha="center", va="top",
            fontsize=6.5, color="#444", style="italic",
        )
        # Stage name (conv1 etc.) further below.
        ax.text(
            bx + fw / 2 + d / 2, fy - 0.18, stage_names[i],
            ha="center", va="top",
            fontsize=6.0, color="#888",
        )

        centres_for_arcs.append((bx, bx + fw, fy, fy + fh, d))
        bx += fw + d + inter_gap

    # Skip-connection arcs: TWO per residual stage (2 BasicBlocks each).
    # Drawn as prominent curved arcs that arch above each prism's TOP face.
    # The arc is the canonical visual signature of a residual network and
    # MUST be readable.
    for i in range(1, n_stages):
        bx0, bx1, fy0, fy1, d = centres_for_arcs[i]
        # Two arcs over each stage. Each arc spans roughly the prism width.
        # Place them at 1/4 and 3/4 of the prism (one per BasicBlock).
        prism_w = bx1 - bx0
        for k in range(2):
            # arc start/end points: spans roughly half the prism width
            mid = (k + 0.5) / 2.0  # 0.25 or 0.75
            frac0 = mid - 0.18
            frac1 = mid + 0.18
            x_lo = bx0 + frac0 * prism_w
            x_hi = bx0 + frac1 * prism_w
            arc = FancyArrowPatch(
                (x_lo, fy1 - 0.005), (x_hi, fy1 - 0.005),
                arrowstyle="-|>", mutation_scale=6.0,
                color="#1f3d6a", lw=0.95,
                connectionstyle=f"arc3,rad=-0.95",
                zorder=8, shrinkA=0.5, shrinkB=0.5,
            )
            ax.add_patch(arc)

    # Right anchor: right-edge midpoint of conv5_x prism.
    last = centres_for_arcs[-1]
    right_anchor = (last[1] + last[4], (last[2] + last[3]) / 2)
    return right_anchor


LSTM_YELLOW = "#f7d35d"  # Olah NN-block yellow
LSTM_PINK   = "#f6c7c7"  # Olah pointwise-op pink
LSTM_LINE   = "#222222"  # cell-state line dark grey/black
LSTM_BG     = "#fffbed"  # cell background (very pale yellow)


def _draw_nn_block(ax, cx, cy, w, h, sym, *, lw=0.45, zorder=8):
    """A yellow rounded rectangle = a learned-NN block (gate or candidate).

    Olah's convention: yellow for sigma / tanh blocks. Symbol is rendered
    in the centre at small fontsize.
    """
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.012",
        facecolor=LSTM_YELLOW, edgecolor=DARK, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, sym, ha="center", va="center",
            fontsize=5.8, color="#1a1a1a", zorder=zorder + 1)


def _draw_pointwise(ax, cx, cy, r, sym, *, lw=0.4, zorder=8):
    """Pink circle = a pointwise operation (×, +, tanh on cell state).

    Olah's convention: pink/coral circles for pointwise ops.
    """
    ax.add_patch(Circle(
        (cx, cy), r,
        facecolor=LSTM_PINK, edgecolor=DARK, linewidth=lw, zorder=zorder,
    ))
    ax.text(cx, cy, sym, ha="center", va="center",
            fontsize=6.5, color="#1a1a1a", zorder=zorder + 1)


def _line(ax, p_from, p_to, *, lw=0.55, color=LSTM_LINE, zorder=7):
    """Plain line segment (no arrowhead) for cell-state / hidden-state
    routing within the LSTM cell."""
    ax.plot([p_from[0], p_to[0]], [p_from[1], p_to[1]],
            color=color, lw=lw, zorder=zorder, solid_capstyle="round")


def _arrowed_line(ax, p_from, p_to, *, lw=0.55, color=LSTM_LINE, zorder=7,
                  mutation_scale=5):
    """Line with a small arrowhead at p_to."""
    a = FancyArrowPatch(
        p_from, p_to,
        arrowstyle="-|>", mutation_scale=mutation_scale,
        color=color, lw=lw, shrinkA=0.5, shrinkB=0.5,
        zorder=zorder,
    )
    ax.add_patch(a)


def lstm_olah_cell(ax, x, y, w, h, *, lw=0.55, shadow=False):
    """Morphologically accurate Olah-style LSTM cell.

    Layout (cell at bottom of x..x+w, y..y+h):

      ┌─────────────────────────────────────────────────┐  C-line (top)
      │  C_{t-1} ──────[×_f]──────[+]───────[tanh]──┬── C_t
      │              ↑              ↑              ↓
      │           [σ_f]         [×_i]          [×_o]
      │            ↑           ↑   ↑              ↑
      │       (concat)─→[σ_i] [tanh_c]         [σ_o]
      │            ↑                              │
      │  h_{t-1} ──────────────────────────────────┴── h_t
      └─────────────────────────────────────────────────┘

    Returns dict with anchor points used by the surrounding figure:
      - input_left:  (x, h-line y) -- where x_t and h_{t-1} enter
      - cell_right:  (x+w, C-line y) -- where C_t exits
      - hidden_right: (x+w, h-line y) -- where h_t exits
      - top_y, bot_y, x_left, x_right
    """
    # Cell boundary (rounded).
    boundary = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.06",
        facecolor=LSTM_BG, edgecolor=DARK, linewidth=lw, zorder=3,
    )
    if shadow:
        boundary.set_path_effects([SHADOW, pe.Normal()])
    ax.add_patch(boundary)

    # Geometry.
    pad = 0.10
    # C-line at top (~80% up); h-line at bottom (~18% up).
    c_y = y + h * 0.80
    h_y = y + h * 0.20

    # Four NN blocks across the cell (left-to-right): σ_f, σ_i, tanh̃, σ_o.
    nn_w = 0.18
    nn_h = 0.16
    nn_y = y + h * 0.46  # mid-height
    cell_inner_left = x + pad + 0.18  # leave room for left "concat" hub
    cell_inner_right = x + w - pad - 0.18
    nn_xs = np.linspace(cell_inner_left, cell_inner_right, 4)
    nn_specs = [
        (nn_xs[0], r"$\sigma_{f}$"),  # forget gate
        (nn_xs[1], r"$\sigma_{i}$"),  # input gate (sigmoid)
        (nn_xs[2], r"$\tilde{C}$"),    # candidate (tanh)
        (nn_xs[3], r"$\sigma_{o}$"),  # output gate
    ]
    for nx, sym in nn_specs:
        _draw_nn_block(ax, nx, nn_y, nn_w, nn_h, sym)

    # Pointwise operations on the C-line.
    pw_r = 0.055
    pw_x_f = nn_xs[0]              # × forget (above σ_f)
    pw_x_plus = (nn_xs[1] + nn_xs[2]) / 2  # + (between input and candidate, above)
    pw_x_i = (nn_xs[1] + nn_xs[2]) / 2 - 0.10  # × input gate (below the +)
    pw_x_o = nn_xs[3]              # × output gate (above σ_o)

    # × forget: on the C-line, directly above σ_f.
    _draw_pointwise(ax, pw_x_f, c_y, pw_r, r"$\times$")
    # × input: BELOW the C-line (on a diagonal between σ_i and tanh̃).
    pw_input_y = nn_y + nn_h / 2 + 0.05
    _draw_pointwise(ax, pw_x_i + 0.05, pw_input_y, pw_r, r"$\times$")
    # + on the C-line: combines forget output with input contribution.
    _draw_pointwise(ax, pw_x_plus, c_y, pw_r, r"$+$")
    # × output: on the lower path (h-line), at column nn_xs[3].
    pw_out_y = h_y + 0.10
    _draw_pointwise(ax, pw_x_o, pw_out_y, pw_r, r"$\times$")
    # tanh on cell-state output: between + and × output, ON the C-line,
    # branching down into × output.
    pw_tanh_x = (nn_xs[2] + nn_xs[3]) / 2 + 0.04
    _draw_pointwise(ax, pw_tanh_x, c_y, pw_r, r"$\tanh$", lw=0.4)

    # ── Cell-state line: C_{t-1} -> [×_f] -> [+] -> [tanh] -> C_t ─────
    x_left = x + pad * 0.4
    x_right = x + w - pad * 0.4
    # left segment to ×_f
    _line(ax, (x_left, c_y), (pw_x_f - pw_r, c_y))
    # ×_f to +
    _line(ax, (pw_x_f + pw_r, c_y), (pw_x_plus - pw_r, c_y))
    # + to tanh-on-cell-state
    _line(ax, (pw_x_plus + pw_r, c_y), (pw_tanh_x - pw_r, c_y))
    # tanh out to C_t exit
    _line(ax, (pw_tanh_x + pw_r, c_y), (x_right, c_y))

    # ── Hidden-state line: h_{t-1} -> ... -> h_t  (and forking into gates) ──
    _line(ax, (x_left, h_y), (pw_x_o, h_y))   # h-line stub
    # Continue h-line to right exit (after × output joins from above).
    _line(ax, (pw_x_o, h_y), (x_right, h_y))

    # ── Routing / forks ─────────────────────────────────────────────────
    # Concatenation hub at left: x_t (from below) and h_{t-1} (from left)
    # converge at (x + pad, h_y) and feed UP into all 4 NN blocks.
    hub_x = x + pad + 0.04
    hub_y = h_y
    # Hub fork dot (small filled black circle, Olah style).
    ax.add_patch(Circle((hub_x, hub_y), 0.018,
                        facecolor=DARK, edgecolor=DARK, zorder=8))
    # Up-bus from hub to a horizontal rail at nn_y - nn_h / 2 - 0.04.
    rail_y = nn_y - nn_h / 2 - 0.025
    _line(ax, (hub_x, hub_y), (hub_x, rail_y))
    _line(ax, (hub_x, rail_y), (nn_xs[3] + 0.02, rail_y))
    # Each NN block taps the rail from below.
    for nx, _ in nn_specs:
        _arrowed_line(ax,
                      (nx, rail_y),
                      (nx, nn_y - nn_h / 2 - 0.005),
                      lw=0.45, mutation_scale=4)

    # σ_f output rises into ×_f on the C-line.
    _arrowed_line(ax,
                  (nn_xs[0], nn_y + nn_h / 2 + 0.005),
                  (pw_x_f, c_y - pw_r - 0.002),
                  lw=0.45, mutation_scale=4)
    # σ_i × tanh̃ -> ×_input -> + on C-line.
    # σ_i upward to ×_input.
    _arrowed_line(ax,
                  (nn_xs[1], nn_y + nn_h / 2 + 0.005),
                  (pw_x_i + 0.05 - pw_r * 0.6, pw_input_y - pw_r * 0.6),
                  lw=0.45, mutation_scale=4)
    # tanh̃ upward to ×_input.
    _arrowed_line(ax,
                  (nn_xs[2], nn_y + nn_h / 2 + 0.005),
                  (pw_x_i + 0.05 + pw_r * 0.6, pw_input_y - pw_r * 0.6),
                  lw=0.45, mutation_scale=4)
    # ×_input upward to + on C-line.
    _arrowed_line(ax,
                  (pw_x_i + 0.05, pw_input_y + pw_r),
                  (pw_x_plus, c_y - pw_r - 0.002),
                  lw=0.45, mutation_scale=4)
    # σ_o upward to ×_output (on the lower line, near h_y).
    # Actually σ_o feeds into ×_output which sits between σ_o and the h-line.
    _arrowed_line(ax,
                  (nn_xs[3], nn_y - nn_h / 2 - 0.005),
                  (pw_x_o, pw_out_y + pw_r + 0.002),
                  lw=0.45, mutation_scale=4)
    # tanh-on-cell-state branches DOWN into ×_output.
    _arrowed_line(ax,
                  (pw_tanh_x, c_y - pw_r - 0.002),
                  (pw_x_o + (pw_tanh_x - pw_x_o) * 0.0,
                   pw_out_y + pw_r + 0.002),
                  lw=0.45, mutation_scale=4)
    # ×_output downward to the h-line (joining hub-to-exit segment).
    _arrowed_line(ax,
                  (pw_x_o, pw_out_y - pw_r - 0.002),
                  (pw_x_o, h_y + 0.005),
                  lw=0.45, mutation_scale=4)

    # x_t entry from below the cell (a stub arrow from outside).
    # We draw a short stub coming up from below the cell into the hub.
    _arrowed_line(ax,
                  (hub_x, y - 0.05),
                  (hub_x, h_y - 0.005),
                  lw=0.55, mutation_scale=5)

    # Tiny labels on the inputs and outputs.
    label_fs = 5.5
    ax.text(x_left - 0.02, c_y + 0.04, r"$C_{t-1}$",
            ha="right", va="bottom", fontsize=label_fs, color="#444",
            style="italic", zorder=9)
    ax.text(x_left - 0.02, h_y - 0.04, r"$h_{t-1}$",
            ha="right", va="top", fontsize=label_fs, color="#444",
            style="italic", zorder=9)
    ax.text(x_right + 0.02, c_y + 0.04, r"$C_{t}$",
            ha="left", va="bottom", fontsize=label_fs, color="#444",
            style="italic", zorder=9)
    ax.text(x_right + 0.02, h_y - 0.04, r"$h_{t}$",
            ha="left", va="top", fontsize=label_fs, color="#444",
            style="italic", zorder=9)
    ax.text(hub_x, y - 0.06, r"$x_{t}$",
            ha="center", va="top", fontsize=label_fs, color="#444",
            style="italic", zorder=9)

    return {
        "x_left": x_left,
        "x_right": x_right,
        "c_y": c_y,
        "h_y": h_y,
        "input_x_t": (hub_x, y - 0.05),
        "input_h_left": (x_left, h_y),
        "input_C_left": (x_left, c_y),
        "out_C_right": (x_right, c_y),
        "out_h_right": (x_right, h_y),
        "top_y": y + h,
        "bot_y": y,
    }


def lstm_simple_cell(ax, x, y, w, h, label, *, facecolor=LSTM_BG,
                     edgecolor=DARK, lw=0.55, fontsize=8.5, shadow=False):
    """Compact LSTM cell representation for L1 / L2 (no internal gates).

    Olah's diagrams typically show ONE detailed cell and keep stacked /
    repeated cells abstract. We follow this convention: only L0 has the
    full gate complement; L1 and L2 are simpler boxes with the same
    cell-state line and recurrent loop signature.
    """
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=3,
    )
    if shadow:
        patch.set_path_effects([SHADOW, pe.Normal()])
    ax.add_patch(patch)

    # Mini "cell-state line + tanh + ×" hint inside (very compact).
    c_y = y + h * 0.72
    h_y = y + h * 0.28
    _line(ax, (x + 0.05, c_y), (x + w - 0.05, c_y),
          lw=0.45, color=LSTM_LINE)
    _line(ax, (x + 0.05, h_y), (x + w - 0.05, h_y),
          lw=0.45, color=LSTM_LINE)
    # one mini × on each line as a hint.
    _draw_pointwise(ax, x + w * 0.45, c_y, 0.030, r"$\times$",
                    lw=0.3, zorder=8)
    # Label centred.
    ax.text(x + w / 2, y + h * 0.48, label,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=TEXT_COL, zorder=9)
    return {
        "x_left": x + 0.05,
        "x_right": x + w - 0.05,
        "c_y": c_y,
        "h_y": h_y,
        "top_y": y + h,
        "bot_y": y,
    }


# ────────────────────────── ROW 1: architecture ───────────────────────────
def draw_row1_architecture(ax, x_left, y_center, *, total_w):
    """Build the entire row-1 architecture, sized to fit `total_w` inches.

    Layout (left to right):
        ResNet-18 stack | encoder-output stickers | sensor stack | LSTM(3) | pi | h_2

    Returns the bottom-centre of the h_2 box -- the takeoff point for the
    'freeze + rollout' arrow that bridges to row 2.
    """
    # --- ResNet-18 encoder (taller for row 1) ---
    enc_w, enc_h = 3.10, 1.95
    enc_x = x_left
    enc_y = y_center - enc_h / 2
    resnet18_stack(ax, enc_x, enc_y, enc_w, enc_h, lw=0.55, shadow=True)

    # --- Encoder-output stickers (3 visual conditions) ---
    sticker_w, sticker_h = 0.36, 0.32
    sticker_x = enc_x + enc_w + 0.12
    visual_conds = ["Coarse", "Foveated", "Uniform"]
    n = len(visual_conds)
    total_sticker_h = n * sticker_h + (n - 1) * 0.06
    sy = y_center + total_sticker_h / 2 - sticker_h
    for cond in visual_conds:
        rounded_box(
            ax, sticker_x, sy, sticker_w, sticker_h,
            ENC_OUT[cond],
            facecolor=COND_LIGHT[cond],
            edgecolor=COND_COLORS[cond],
            lw=0.6, fontsize=7.5, rad=0.02,
        )
        sy -= (sticker_h + 0.06)
    # Tiny caption below the sticker column.
    ax.text(
        sticker_x + sticker_w / 2,
        y_center - total_sticker_h / 2 - 0.05,
        "encoder out",
        ha="center", va="top",
        fontsize=6.5, color="#666", style="italic",
    )

    # --- Sensor stack (4 icons, ~30% larger than v3) ---
    sensor_x = sticker_x + sticker_w + 0.55
    sensor_w = 0.42
    sensor_h = 0.42
    n_sensors = 4
    sensor_total_h = n_sensors * sensor_h + (n_sensors - 1) * 0.08
    sensor_y_top = y_center + sensor_total_h / 2 - sensor_h
    sensor_specs = [
        ("gps",     "GPS"),
        ("compass", "Cmp"),
        ("goal",    "Goal"),
        ("action",  "Act"),
    ]
    sensor_centers = []
    for i, (kind, lab) in enumerate(sensor_specs):
        sx = sensor_x
        sy = sensor_y_top - i * (sensor_h + 0.08)
        rounded_box(
            ax, sx, sy, sensor_w, sensor_h, "",
            facecolor="white", edgecolor="#888", lw=0.45, rad=0.03,
        )
        cx = sx + sensor_w / 2
        cy = sy + sensor_h / 2 + 0.03
        glyph_size = sensor_h * 0.55
        if kind == "gps":
            glyph_location_pin(ax, cx, cy, glyph_size, color="#c9701a")
        elif kind == "compass":
            glyph_compass_rose(ax, cx, cy, glyph_size, color="#5a4ea3")
        elif kind == "goal":
            glyph_target(ax, cx, cy, glyph_size, color="#2f8a3f")
        elif kind == "action":
            glyph_action(ax, cx, cy, glyph_size, color="#2766a8")
        ax.text(cx, sy + 0.03, lab,
                ha="center", va="bottom", fontsize=6.5, color="#333")
        sensor_centers.append((cx, cy))
    # Sensor-stack caption: just below the bottom-most sensor box.
    bottom_sensor_y = sensor_y_top - (n_sensors - 1) * (sensor_h + 0.08)
    ax.text(
        sensor_x + sensor_w / 2,
        bottom_sensor_y - 0.07,
        "sensors",
        ha="center", va="top", fontsize=7.5, color="#555", style="italic",
    )

    # --- LSTM stack: ONE detailed Olah-style cell at L0 + 2 simpler cells
    # (L1, L2) above. L0 width is generous (Olah cell needs space for the
    # 4 NN blocks + 4 pointwise ops + cell-state and hidden-state lines).
    lstm_x = sensor_x + sensor_w + 0.65
    lstm_w = 2.65           # L0 detailed cell width (more room for gates)
    lstm_h_main = 1.00      # L0 detailed cell height (more vertical room)
    lstm_h_simple = 0.30    # L1, L2 compact cell height
    lstm_simple_w = lstm_w * 0.82  # L1, L2 narrower than L0 (Olah hierarchy)

    layer_labels = ["L2", "L1", "L0"]
    inter_layer_gap = 0.10

    # Position: stack is L0 (detailed, bottom) → L1 → L2 (top), centred so
    # the WHOLE stack vertically straddles y_center.
    stack_total_h = lstm_h_main + 2 * lstm_h_simple + 2 * inter_layer_gap
    stack_bot = y_center - stack_total_h / 2
    l0_y = stack_bot
    l0_top = l0_y + lstm_h_main
    l1_y = l0_top + inter_layer_gap
    l1_top = l1_y + lstm_h_simple
    l2_y = l1_top + inter_layer_gap
    l2_top = l2_y + lstm_h_simple

    # Detailed Olah cell at L0.
    l0_anchors = lstm_olah_cell(
        ax, lstm_x, l0_y, lstm_w, lstm_h_main, lw=0.6, shadow=True,
    )
    # Simple cells at L1 and L2 (centred horizontally over L0).
    simple_x = lstm_x + (lstm_w - lstm_simple_w) / 2
    l1_anchors = lstm_simple_cell(
        ax, simple_x, l1_y, lstm_simple_w, lstm_h_simple, "L1",
        facecolor=LSTM_BG, lw=0.5, fontsize=7.5, shadow=True,
    )
    l2_anchors = lstm_simple_cell(
        ax, simple_x, l2_y, lstm_simple_w, lstm_h_simple, "L2",
        facecolor=LSTM_BG, lw=0.5, fontsize=7.5, shadow=True,
    )

    # Recurrent self-loops (Olah convention: small arc above each cell).
    def _recurrent_loop(cell_x, cell_top, loop_w=0.18):
        x_a = cell_x + loop_w * 0.5
        x_b = cell_x + loop_w * 1.5
        loop = FancyArrowPatch(
            (x_b, cell_top + 0.005),
            (x_a, cell_top + 0.005),
            arrowstyle="-|>", mutation_scale=5,
            color="#a6803c", lw=0.65,
            connectionstyle="arc3,rad=-1.30",
            zorder=10, shrinkA=0.5, shrinkB=0.5,
        )
        ax.add_patch(loop)

    _recurrent_loop(lstm_x + 0.08, l0_top)
    _recurrent_loop(simple_x + 0.05, l1_top)
    _recurrent_loop(simple_x + 0.05, l2_top)

    # Inter-layer arrows: L0 -> L1 -> L2 (cell-state and hidden-state both).
    # Use vertical lines on the right edge: cell-state at upper part, hidden
    # at lower part.
    bridge_x = simple_x + lstm_simple_w * 0.78
    # L0_top (right edge area) up to L1 bottom.
    _arrowed_line(ax,
                  (bridge_x, l0_top),
                  (bridge_x, l1_y - 0.005),
                  lw=0.5, color="#a6803c", mutation_scale=6, zorder=8)
    _arrowed_line(ax,
                  (bridge_x, l1_top),
                  (bridge_x, l2_y - 0.005),
                  lw=0.5, color="#a6803c", mutation_scale=6, zorder=8)

    # LSTM caption above L2.
    ax.text(lstm_x + lstm_w / 2, l2_top + 0.10,
            "LSTM (3 layers, 512-d)  --  Olah-style cell, ×3 stacked",
            ha="center", va="bottom", fontsize=7.5, color="#555",
            style="italic")
    # "L0" / "L1" / "L2" badge below the L0 cell title (the detailed cell
    # already has its inputs labelled internally; we add a small "L0" tag
    # in the upper-left corner for parity with the simple cells above).
    ax.text(lstm_x + 0.04, l0_top - 0.05, "L0",
            ha="left", va="top",
            fontsize=7.5, fontweight="bold", color=TEXT_COL, zorder=11)

    # Convenience anchors for downstream wiring.
    layer_y = [l2_y + lstm_h_simple / 2,  # L2 mid
               l1_y + lstm_h_simple / 2,  # L1 mid
               l0_y + lstm_h_main / 2]    # L0 mid
    layer_y_top = [l2_top, l1_top, l0_top]
    layer_y_bot = [l2_y, l1_y, l0_y]

    # Encoder + sensor inputs feed into L0's left side on the h-line
    # (Olah's h_{t-1} input position).
    l0_h_in = (lstm_x, l0_anchors["h_y"])

    # encoder-output -> L0 left side (h-line input). Encoder anchor at
    # right-edge midpoint of the sticker column.
    enc_out_anchor = (sticker_x + sticker_w + 0.04,
                      (sticker_y_top := y_center))
    arrow(ax, (enc_out_anchor[0], y_center),
          (l0_h_in[0] - 0.01, l0_h_in[1]),
          lw=0.55, mutation_scale=8,
          connectionstyle="arc3,rad=+0.10")

    # Sensor stack -> L0 left side (curve in from upper-right of L0)
    sensor_mid = (
        sensor_x + sensor_w + 0.04,
        (sensor_centers[0][1] + sensor_centers[-1][1]) / 2,
    )
    arrow(ax, sensor_mid, (l0_h_in[0] - 0.01, l0_h_in[1]),
          lw=0.55, mutation_scale=8,
          connectionstyle="arc3,rad=+0.20")

    # --- Policy head (triangle) right of LSTM stack ---
    # Take off from the right-edge of the L0 cell-state line (output C_t,
    # which the policy head reads from the top-most layer L2 in practice;
    # we keep visual simplicity by drawing pi off the right edge of the
    # stack, anchored at L2 mid).
    pol_x = lstm_x + lstm_w + 0.40
    pol_y = layer_y[0]  # L2 mid (top of stack)
    pol_size = 0.30
    poly = Polygon(
        [(pol_x, pol_y - pol_size),
         (pol_x + pol_size * 1.6, pol_y),
         (pol_x, pol_y + pol_size)],
        closed=True,
        facecolor="#e6f5e6", edgecolor=DARK, linewidth=0.7, zorder=4,
    )
    ax.add_patch(poly)
    ax.text(pol_x + pol_size * 0.55, pol_y, r"$\pi$",
            ha="center", va="center", fontsize=11, fontweight="bold",
            zorder=5)
    # Arrow from L2 right edge to policy head input.
    l2_right_x = simple_x + lstm_simple_w
    arrow(ax, (l2_right_x + 0.02, pol_y), (pol_x - 0.02, pol_y),
          lw=0.55, mutation_scale=8)

    # --- h_2 vector: prominent column-of-squares, right end of row 1 ---
    h2_w, h2_h = 1.05, 1.85
    h2_x = pol_x + pol_size * 1.6 + 0.55
    h2_y = y_center - h2_h / 2

    rounded_box(
        ax, h2_x, h2_y, h2_w, h2_h, "",
        facecolor=HIGHLIGHT, edgecolor="#7a5a1f", lw=1.1, rad=0.05,
        shadow=True,
    )
    n_sq = 12
    sq_size = 0.13
    column_h = n_sq * sq_size
    grid_x = h2_x + h2_w / 2 - sq_size / 2
    grid_y_top = h2_y + (h2_h - column_h) / 2 + (n_sq - 1) * sq_size
    cmap = plt.get_cmap("YlOrBr")
    rng = np.random.default_rng(7)
    vals = rng.uniform(0.20, 0.85, size=n_sq)
    for i in range(n_sq):
        sy = grid_y_top - i * sq_size
        rect = Rectangle(
            (grid_x, sy), sq_size, sq_size,
            facecolor=cmap(vals[i]), edgecolor=DARK, linewidth=0.35,
            zorder=4,
        )
        ax.add_patch(rect)
    # Math label above.
    ax.text(h2_x + h2_w / 2, h2_y + h2_h + 0.08,
            r"$\mathbf{h}_2 \in \mathbb{R}^{512}$",
            ha="center", va="bottom",
            fontsize=10.0, fontweight="bold", zorder=5)

    # L2 -> h_2: tap from the TOP edge of L2 cell (distinct from the
    # right-mid edge which goes to pi), curving over the policy head into
    # the top-left of the h_2 box. Semantically: the same activation that
    # feeds pi is ALSO collected as the probe target h_2.
    l2_top_x = simple_x + lstm_simple_w * 0.80
    l2_top_y = layer_y_top[0]
    l2_branch_src = (l2_top_x, l2_top_y - 0.005)
    h2_top_left = (h2_x + 0.20, h2_y + h2_h - 0.05)
    arrow(ax, l2_branch_src, h2_top_left, lw=0.8, mutation_scale=10,
          connectionstyle="arc3,rad=-0.30",
          color="#7a5a1f", zorder=5)

    # Right-edge bbox of row 1 (used to size the canvas if needed).
    row1_right = h2_x + h2_w
    row1_left = enc_x

    return {
        "h2_bottom_center": (h2_x + h2_w / 2, h2_y),
        "row1_left": row1_left,
        "row1_right": row1_right,
    }


# ────────────────────────── ROW 2: method-finding columns ─────────────────
def _draw_h1_minibar(ax, x, y, w, h, accent):
    """Bottom panel of column 1: 4-bar GPS R^2 (B/C/F/U)."""
    rounded_box(ax, x, y, w, h, "",
                facecolor="white", edgecolor=accent, lw=0.5, rad=0.04,
                zorder=2, shadow=True)
    ax.text(x + w / 2, y + h - 0.08, "H1: encoder-memory race",
            ha="center", va="top", fontsize=8.5,
            fontweight="bold", color=accent, zorder=5)
    inner_x = x + 0.40
    inner_w = w - 0.55
    plot_top = y + h - 0.30
    plot_bot = y + 0.25
    inner_h = plot_top - plot_bot
    midline_y = plot_bot + 0.42 * inner_h
    raw_vals = [0.95, 0.78, 0.06, -0.31]
    vmax = 1.0
    bar_w = inner_w / (len(raw_vals) * 1.7)
    gap = bar_w * 0.7
    ax.plot([inner_x, inner_x + inner_w],
            [midline_y, midline_y], color="#888", lw=0.45, zorder=3)
    for i, (cond, v) in enumerate(zip(COND_ORDER, raw_vals)):
        bx = inner_x + i * (bar_w + gap) + gap / 2
        max_up = plot_top - midline_y
        max_down = midline_y - plot_bot
        if v >= 0:
            bh = max_up * (v / vmax)
            ax.add_patch(Rectangle(
                (bx, midline_y), bar_w, bh,
                facecolor=COND_COLORS[cond], edgecolor=DARK, lw=0.4,
                zorder=4))
        else:
            bh = max_down * (abs(v) / vmax)
            ax.add_patch(Rectangle(
                (bx, midline_y - bh), bar_w, bh,
                facecolor=COND_COLORS[cond], edgecolor=DARK, lw=0.4,
                zorder=4))
        ax.text(bx + bar_w / 2, plot_bot - 0.04, cond[0],
                ha="center", va="top", fontsize=7.0, color="#444",
                zorder=5)
    ax.text(inner_x - 0.06, midline_y, r"GPS $R^2$",
            ha="right", va="center", fontsize=7.0, color="#444",
            zorder=5)


def _draw_h2_heatmap(ax, x, y, w, h, accent):
    """Bottom panel of column 2: 4x4 donor/recipient heatmap."""
    rounded_box(ax, x, y, w, h, "",
                facecolor="white", edgecolor=accent, lw=0.5, rad=0.04,
                zorder=2, shadow=True)
    ax.text(x + w / 2, y + h - 0.08, "H2: disjoint subspaces",
            ha="center", va="top", fontsize=8.5,
            fontweight="bold", color=accent, zorder=5)

    inner_top = y + h - 0.32
    inner_bot = y + 0.30
    side = inner_top - inner_bot
    n = 4
    cell = side / n
    inner_x = x + 0.50
    M = np.array([
        [0.90, 0.40, 0.30, 0.20],
        [0.55, 0.85, 0.35, 0.25],
        [0.20, 0.25, 0.85, 0.55],
        [0.15, 0.20, 0.50, 0.85],
    ])
    cmap = plt.get_cmap("RdPu")
    for i in range(n):
        for j in range(n):
            rx = inner_x + j * cell
            ry = inner_bot + (n - 1 - i) * cell
            ax.add_patch(Rectangle(
                (rx, ry), cell, cell,
                facecolor=cmap(M[i, j]), edgecolor="white", lw=0.4,
                zorder=4))
    initials = ["B", "C", "F", "U"]
    for j in range(n):
        ax.text(inner_x + (j + 0.5) * cell, inner_bot - 0.03,
                initials[j],
                ha="center", va="top", fontsize=7.0, color="#333",
                zorder=5)
    for i in range(n):
        ax.text(inner_x - 0.03, inner_bot + (n - 1 - i + 0.5) * cell,
                initials[i],
                ha="right", va="center", fontsize=7.0, color="#333",
                zorder=5)
    ax.text(inner_x + n * cell / 2, inner_bot - 0.16,
            "recipient",
            ha="center", va="top", fontsize=7.0, color="#333",
            style="italic", zorder=5)
    ax.text(inner_x - 0.22, inner_bot + n * cell / 2,
            "donor",
            ha="center", va="center", fontsize=7.0, color="#333",
            style="italic", rotation=90, zorder=5)


def _draw_dissociation(ax, x, y, w, h, accent):
    """Bottom panel of column 3: 2x2 probe-vs-policy dissociation."""
    rounded_box(ax, x, y, w, h, "",
                facecolor="white", edgecolor=accent, lw=0.5, rad=0.04,
                zorder=2, shadow=True)
    ax.text(x + w / 2, y + h - 0.08, "probe vs policy memory",
            ha="center", va="top", fontsize=8.5,
            fontweight="bold", color=accent, zorder=5)

    inner_top = y + h - 0.32
    inner_bot = y + 0.30
    inner_x = x + 0.55
    inner_w = w - 0.75
    inner_h = inner_top - inner_bot

    ax.add_patch(Rectangle(
        (inner_x, inner_bot), inner_w, inner_h,
        facecolor="#fafafa", edgecolor="#888", lw=0.5, zorder=3))
    midx = inner_x + inner_w / 2
    midy = inner_bot + inner_h / 2
    ax.plot([midx, midx], [inner_bot, inner_bot + inner_h],
            color="#aaa", lw=0.45, ls="--", zorder=4)
    ax.plot([inner_x, inner_x + inner_w], [midy, midy],
            color="#aaa", lw=0.45, ls="--", zorder=4)
    # Off-diagonal markers: Coarse top-right (probe-readable, low-policy-use),
    # Uniform bottom-left (low-probe, high-policy-use).
    ax.scatter([inner_x + 0.78 * inner_w], [inner_bot + 0.78 * inner_h],
               s=80, marker="s", color=COND_COLORS["Coarse"],
               edgecolor=DARK, linewidth=0.5, zorder=5)
    ax.scatter([inner_x + 0.22 * inner_w], [inner_bot + 0.22 * inner_h],
               s=80, marker="^", color=COND_COLORS["Uniform"],
               edgecolor=DARK, linewidth=0.5, zorder=5)
    ax.text(midx, inner_bot - 0.05,
            "probe-readable",
            ha="center", va="top", fontsize=7.0, color="#333",
            style="italic", zorder=5)
    ax.text(inner_x - 0.06, midy,
            "policy-uses",
            ha="right", va="center", fontsize=7.0, color="#333",
            style="italic", rotation=90, zorder=5)
    ax.text(inner_x + 0.04, inner_bot - 0.02, "lo",
            ha="left", va="top", fontsize=6.0, color="#666", zorder=5)
    ax.text(inner_x + inner_w - 0.04, inner_bot - 0.02, "hi",
            ha="right", va="top", fontsize=6.0, color="#666", zorder=5)


def _draw_method_top_probes(ax, x, y, w, h, accent):
    """Top-of-column method icon for 'Linear probes'."""
    rounded_box(ax, x, y, w, h, "",
                facecolor="#fafafa", edgecolor=accent, lw=0.55, rad=0.04,
                zorder=2, shadow=True)
    ax.text(x + w / 2, y + h - 0.08, "Linear probes",
            ha="center", va="top", fontsize=9.0, fontweight="bold",
            color=accent, zorder=5)
    # Mini regression-line glyph centred horizontally, slightly below title.
    icon_cx = x + w / 2
    icon_cy = y + h * 0.45
    xs = np.linspace(icon_cx - 0.30, icon_cx + 0.30, 7)
    ys = icon_cy + 0.30 * (xs - icon_cx) / 0.30 \
        + np.array([-0.06, 0.03, -0.04, 0.05, -0.02, 0.04, -0.05])
    ax.scatter(xs, ys, s=14, color=accent, zorder=4)
    xline = np.linspace(icon_cx - 0.32, icon_cx + 0.32, 30)
    yline = icon_cy + 0.30 * (xline - icon_cx) / 0.30
    ax.plot(xline, yline, color=accent, lw=1.0, zorder=3)
    ax.text(x + w / 2, y + 0.10, r"Ridge $\to R^{2}$",
            ha="center", va="bottom", fontsize=8.0, color="#333",
            style="italic", zorder=5)


def _draw_method_top_geometry(ax, x, y, w, h, accent):
    """Top-of-column method icon for 'Geometry'."""
    rounded_box(ax, x, y, w, h, "",
                facecolor="#fafafa", edgecolor=accent, lw=0.55, rad=0.04,
                zorder=2, shadow=True)
    ax.text(x + w / 2, y + h - 0.08, "Geometry",
            ha="center", va="top", fontsize=9.0, fontweight="bold",
            color=accent, zorder=5)
    # 4x4 mini heatmap, centred.
    n = 4
    sz = 0.10
    icon_cx = x + w / 2
    icon_cy = y + h * 0.45
    top = icon_cy + (n / 2) * sz
    left = icon_cx - (n / 2) * sz
    cmap = plt.get_cmap("Purples")
    grid = np.array([
        [0.85, 0.45, 0.30, 0.20],
        [0.45, 0.85, 0.40, 0.30],
        [0.30, 0.40, 0.85, 0.45],
        [0.20, 0.30, 0.45, 0.85],
    ])
    for i in range(n):
        for j in range(n):
            rx = left + j * sz
            ry = top - (i + 1) * sz
            ax.add_patch(Rectangle(
                (rx, ry), sz, sz,
                facecolor=cmap(grid[i, j]), edgecolor="#444",
                linewidth=0.3, zorder=4,
            ))
    ax.text(x + w / 2, y + 0.10, "CKA / shape / ID",
            ha="center", va="bottom", fontsize=8.0, color="#333",
            style="italic", zorder=5)


def _draw_method_top_intervention(ax, x, y, w, h, accent):
    """Top-of-column method icon for 'Intervention' (donor->recipient swatches)."""
    rounded_box(ax, x, y, w, h, "",
                facecolor="#fafafa", edgecolor=accent, lw=0.55, rad=0.04,
                zorder=2, shadow=True)
    ax.text(x + w / 2, y + h - 0.08, "Intervention",
            ha="center", va="top", fontsize=9.0, fontweight="bold",
            color=accent, zorder=5)
    # Two h_2-vector swatches, donor (Coarse) on the left, recipient
    # (Foveated) on the right, connected by a curved arrow.
    icon_cx = x + w / 2
    icon_cy = y + h * 0.45
    rect_w, rect_h = 0.18, 0.40
    swatch_specs = [
        (icon_cx - 0.32, COND_COLORS["Coarse"], "Blues", 13),
        (icon_cx + 0.32, COND_COLORS["Foveated"], "Reds", 17),
    ]
    for cx, col, cmap_name, seed in swatch_specs:
        ax.add_patch(Rectangle(
            (cx - rect_w / 2, icon_cy - rect_h / 2),
            rect_w, rect_h,
            facecolor="white", edgecolor=col, linewidth=0.6, zorder=3,
        ))
        cell_h = rect_h / 6
        cmap_col = plt.get_cmap(cmap_name)
        cell_rng = np.random.default_rng(seed)
        cell_vals = cell_rng.uniform(0.30, 0.85, size=6)
        for k in range(6):
            ax.add_patch(Rectangle(
                (cx - rect_w / 2,
                 icon_cy - rect_h / 2 + k * cell_h),
                rect_w, cell_h,
                facecolor=cmap_col(cell_vals[k]),
                edgecolor="none", zorder=4,
            ))
        ax.add_patch(Rectangle(
            (cx - rect_w / 2, icon_cy - rect_h / 2),
            rect_w, rect_h,
            facecolor="none", edgecolor=col, linewidth=0.6, zorder=5,
        ))
    arrow(ax,
          (icon_cx - 0.32 + rect_w / 2 + 0.01, icon_cy),
          (icon_cx + 0.32 - rect_w / 2 - 0.01, icon_cy),
          lw=0.7, color=accent,
          connectionstyle="arc3,rad=-0.35",
          mutation_scale=8, zorder=4)
    ax.text(x + w / 2, y + 0.10, "Transplant / shortcut",
            ha="center", va="bottom", fontsize=8.0, color="#333",
            style="italic", zorder=5)


def draw_row2_columns(ax, h2_strip_y, x_left, total_w):
    """3 method-finding columns spanning `total_w` inches starting at `x_left`.

    Each column has a method-icon panel at top, an arrow down, a finding
    panel at bottom. Returns a list of (col_top_x, col_top_y) anchors for
    the down-arrow distribution from the h_2 strip.
    """
    n_cols = 3
    inter_col_gap = 0.50
    col_w = (total_w - (n_cols - 1) * inter_col_gap) / n_cols

    method_top_h = 1.10
    arrow_band_h = 0.30
    finding_h = 1.30

    column_total_h = method_top_h + arrow_band_h + finding_h
    # Place columns so their tops sit just below the h_2 strip.
    top_y = h2_strip_y - 0.20
    method_top_y_top = top_y
    method_top_y_bot = top_y - method_top_h
    finding_y_top = method_top_y_bot - arrow_band_h
    finding_y_bot = finding_y_top - finding_h

    method_drawers = [
        _draw_method_top_probes,
        _draw_method_top_geometry,
        _draw_method_top_intervention,
    ]
    finding_drawers = [_draw_h1_minibar, _draw_h2_heatmap, _draw_dissociation]
    accent_keys = ["probes", "geometry", "intervention"]

    col_top_centres = []
    for k in range(n_cols):
        cx = x_left + k * (col_w + inter_col_gap)
        accent = METHOD_ACCENTS[accent_keys[k]]

        method_drawers[k](
            ax, cx, method_top_y_bot, col_w, method_top_h, accent,
        )
        finding_drawers[k](
            ax, cx, finding_y_bot, col_w, finding_h, accent,
        )

        # Down-arrow from method panel bottom to finding panel top.
        arrow(
            ax,
            (cx + col_w / 2, method_top_y_bot - 0.02),
            (cx + col_w / 2, finding_y_top + 0.02),
            lw=0.6, color=accent, mutation_scale=9,
        )

        col_top_centres.append((cx + col_w / 2, method_top_y_top))

    return col_top_centres, finding_y_bot


# ────────────────────────── main ──────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Figure dimensions: 14.0" x 7". When LaTeX scales to \linewidth
    # (~6.5"), all elements remain readable. Width was bumped from 12.5"
    # to 14.0" to give morphologically-accurate ResNet-18 (~3.1") and
    # Olah-style LSTM cell (~2.5") proper space without crushing other
    # elements.
    FIG_W, FIG_H = 14.0, 7.0
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Row 1: architecture (centred horizontally) ───────────────────────
    # Layout (left to right):
    #   condition column | ResNet-18 | encoder stickers | sensors | LSTM | pi | h_2
    row1_y = 5.30           # centre of row 1
    cond_col_x = 0.40        # left edge of the 4-thumbnail column
    cond_col_w = 0.78
    row1_x_left = cond_col_x + cond_col_w + 0.65  # ResNet-18 left edge

    # 4 condition thumbnails (Blind / Coarse / Foveated / Uniform).
    # col_h kept <= 3.6" so the column fits within row 1's vertical band
    # (row1_y +/- 1.85 + small label margin) and the Blind bypass apex
    # never escapes above FIG_H.
    cond_anchors = draw_condition_column(
        ax, cond_col_x, row1_y, col_w=cond_col_w, col_h=3.55,
    )

    # Train annotation (top of row 1). Shifted RIGHT of ResNet-18 so it
    # does not collide with the Blind bypass arc that arches over the
    # encoder.
    ax.text(
        row1_x_left + 3.30, row1_y + 1.40,
        "Train: Gibson (411) + MP3D-train (61)",
        ha="left", va="bottom",
        fontsize=8.5, color="#666666", style="italic",
    )

    row1 = draw_row1_architecture(
        ax, row1_x_left, row1_y, total_w=FIG_W - row1_x_left - 0.85,
    )

    # Eval annotation: placed just below row 1 on the LEFT side, well
    # away from the freeze/rollout arrow which sits centre-right. Aligned
    # under the ResNet-18 block.
    ax.text(
        row1_x_left + 0.30, row1_y - 1.55,
        "Eval: Gibson held-out + MP3D-test (18)",
        ha="left", va="top",
        fontsize=8.5, color="#666666", style="italic",
    )

    # ── Arrows from condition thumbnails -> ResNet-18 / LSTM ─────────────
    # Coarse / Foveated / Uniform converge on the LEFT edge of ResNet-18.
    # ResNet-18 is now ~3.1" wide; its visible left edge is approximately
    # at row1_x_left + small centering pad.
    enc_w = 3.10
    # The new resnet18_stack centres the 5-stage row within (x, x+w);
    # its left visible edge lies a bit inside x. Use a 0.05" pad.
    enc_left_visible_x = row1_x_left + 0.10
    enc_mid_y = row1_y
    for cond in ("Coarse", "Foveated", "Uniform"):
        src = cond_anchors[cond]
        # Aim slightly inside the encoder stack's left mid-band, with a
        # gentle curve for visual separation.
        dst = (enc_left_visible_x - 0.03, enc_mid_y)
        rad = {
            "Coarse":   -0.18,
            "Foveated": +0.00,
            "Uniform":  +0.18,
        }[cond]
        arrow(
            ax, src, dst,
            lw=0.55, color=GREY_LINE,
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=8, zorder=3,
        )

    # Blind bypass arc: routes ABOVE the ResNet-18 stack and lands on the
    # sensor-stack/L0 join region (i.e. at the L0 input on the LSTM column).
    # The L0 input lives at lstm_x = sensor_x + sensor_w + 0.60.
    blind_src = cond_anchors["Blind"]
    # Compute approximate sensor-stack -> L0 join point in figure coords.
    # From draw_row1_architecture: enc_x + enc_w + 0.12
    #   + sticker_w(0.36) + 0.55 + sensor_w(0.42) + 0.04.
    sticker_x = row1_x_left + enc_w + 0.12
    sticker_w = 0.36
    sensor_x = sticker_x + sticker_w + 0.55
    sensor_w = 0.42
    sensor_join_x = sensor_x + sensor_w + 0.04
    # Aim the bypass at the LSTM L0 input region. Land high enough that
    # the arc clearly arches OVER ResNet-18 without colliding with the
    # encoder-out stickers or sensors below.
    blind_dst = (sensor_join_x - 0.02, row1_y + 0.55)
    arrow(
        ax, blind_src, blind_dst,
        lw=0.6, color=COND_COLORS["Blind"],
        connectionstyle="arc3,rad=-0.30",
        mutation_scale=9, zorder=7,
    )
    # "bypass" caption placed inline along the top of the arc (kept well
    # below FIG_H so it doesn't escape the bounding box).
    apex_x = (blind_src[0] + blind_dst[0]) / 2 - 0.10
    apex_y = min(FIG_H - 0.25, max(blind_src[1], blind_dst[1]) + 0.35)
    ax.text(
        apex_x, apex_y, "bypass",
        ha="center", va="center",
        fontsize=6.5, color=COND_COLORS["Blind"],
        style="italic", zorder=7,
    )

    # ── Row 2: 3 method-finding columns ──────────────────────────────────
    # h_2 strip y: where the down-arrow lands and distributes horizontally.
    h2_strip_y = 3.10
    h2_strip_left = 0.80
    h2_strip_right = FIG_W - 0.80
    h2_strip_h = 0.20

    # The h_2 distribution strip: a thin highlighted bar with 'h_2' label
    # in the centre. It bridges row 1's h_2 box (top) to row 2 columns.
    rounded_box(
        ax,
        h2_strip_left, h2_strip_y - h2_strip_h / 2,
        h2_strip_right - h2_strip_left, h2_strip_h,
        "",
        facecolor=HIGHLIGHT, edgecolor="#7a5a1f", lw=0.8, rad=0.04,
        zorder=3,
    )
    ax.text(
        FIG_W / 2, h2_strip_y,
        r"frozen $\mathbf{h}_2$ from rollouts",
        ha="center", va="center",
        fontsize=9.0, fontweight="bold", color="#7a5a1f", zorder=5,
    )

    # Down arrow from row 1 h_2 box to the centre of the h_2 strip.
    h2_bot_x, h2_bot_y = row1["h2_bottom_center"]
    h2_strip_top = h2_strip_y + h2_strip_h / 2
    arrow(
        ax,
        (h2_bot_x, h2_bot_y - 0.05),
        (FIG_W / 2, h2_strip_top + 0.02),
        lw=0.95, color="#7a5a1f", mutation_scale=14,
        connectionstyle="arc3,rad=-0.10",
    )
    # 'freeze + rollout' annotation, placed RIGHT of the curved down-arrow
    # (between h_2 box and strip), well clear of the Eval text on the left.
    ax.text(
        h2_bot_x + 0.32, (h2_bot_y + h2_strip_top) / 2,
        "freeze\n+ rollout",
        ha="left", va="center",
        fontsize=8.0, color="#7a5a1f", style="italic", zorder=5,
    )

    # Row 2 columns.
    row2_x_left = 0.80
    row2_total_w = FIG_W - 1.60
    col_top_centres, _ = draw_row2_columns(
        ax, h2_strip_y - h2_strip_h / 2, row2_x_left, row2_total_w,
    )

    # Distribute arrows from the h_2 strip down to each column-top centre.
    for (cx, cy_top) in col_top_centres:
        arrow(
            ax,
            (cx, h2_strip_y - h2_strip_h / 2 - 0.02),
            (cx, cy_top + 0.02),
            lw=0.6, color="#7a5a1f", mutation_scale=8,
        )

    # Optional grid debug overlay.
    if DEBUG_GRID:
        for k in range(int(FIG_W / UNIT) + 1):
            ax.axvline(k * UNIT, lw=0.2, color="red", alpha=0.3)
        for k in range(int(FIG_H / UNIT) + 1):
            ax.axhline(k * UNIT, lw=0.2, color="red", alpha=0.3)

    out = OUT_DIR / "fig_pipeline_overview.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
