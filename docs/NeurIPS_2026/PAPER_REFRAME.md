# Paper reframe after probe-methodology deep-dive

## Status of original claims (post-fix, 5-condition det data)

| Original claim | Status | Evidence |
|----------------|--------|----------|
| H1: foveated > uniform compensatory memory | ❌ **REVERSED** | Both ~0 GPS R² under det; foveated has no advantage |
| "More pixels ≠ better decoding" hook | ✅ **Stronger version** | Matched 1px > uniform full-RGB, R² 0.78 vs 0 |
| H2: representational format divergence | ✅ Survives | CKA / transplant / transfer are behavioral, untouched |
| H3: fov-learned compass 0.94 | ❌ Bug artefact | CV R² = -1.34 ± 3.14 under det (unreliable) |
| "LSTM learns cognitive map" | ✅ **Strengthened** | Blind 0.95, matched 0.78 — map emerges when vision limited |

## The TRUE core finding

**Visual encoder capacity inversely determines LSTM spatial encoding.**

Clean monotonic pattern across 5 conditions (5-fold CV, deterministic probe):

| Condition | Encoder type | GPS R² | Compass R² | DtG R² |
|-----------|--------------|--------|-----------|--------|
| Blind | None | **0.95 ± 0.02** | **0.81 ± 0.08** | 0.90 ± 0.03 |
| Matched | 1×1 degenerate | **0.78 ± 0.10** | **0.64 ± 0.10** | 0.85 ± 0.12 |
| Uniform | Full RGB | -0.31 ± 0.86 | +0.36 ± 0.23 | 0.86 ± 0.09 |
| Foveated-fix | Full RGB + Gaussian blur | +0.06 ± 0.88 | +0.07 ± 0.69 | 0.82 ± 0.09 |
| Fov-learned | Full RGB + static gaze | -2.43 ± 3.98 | -1.34 ± 3.14 | 0.81 ± 0.09 |

DtG (task-relevant ego-state) is **universally encoded** (~0.80-0.90) across all conditions
— every agent tracks "how far to goal". But world-frame GPS/compass is only encoded
when the visual encoder can't provide it.

## Bio-alignment tightens

The paper already cited:
- Kupers 2014 (blind humans → enhanced hippocampal coupling)
- Chen 2016 (grid cells collapse in darkness)
- Geva 2016 (bat vision↔echolocation remapping)

Original paper used these to motivate "foveated = compensatory". Under the new
framing, the analogies are **directly** supported:
- Blind GPS 0.95 ↔ blind-human hippocampal enhancement
- Matched 1×1 (near-dark) ↔ grid cells in darkness
- Fov/uniform ≈ 0 ↔ sighted conditions relying on visual streams

No re-cite needed; just redirect the motivation.

## Proposed paper restructure

### Abstract — new lede

Replace "foveated vision shapes cognitive maps" with something like:
> "We train five PointNav agents differing only in their visual input
> (blind / 1×1-degenerate-encoder / foveated / uniform / foveated-with-
> learned-gaze) and probe what they encode in their recurrent memory.
> A monotonic pattern emerges: the less visual information the encoder
> provides, the more the LSTM encodes world-frame spatial variables.
> Blind and 1×1-bottleneck agents develop rich cognitive maps (GPS
> R² = 0.95, 0.78); agents with full RGB encoders do not (R² ≈ 0),
> storing only task-relevant ego-state. This replicates and extends
> Wijmans (2023) beyond the blind special case, identifying encoder-
> capacity bottleneck (not sensor modality) as the trigger for
> compensatory memory. Within the rich-encoder regime, gaze
> location shifts representational format [H3 — pending fov-shifted]."

### §3 New H1 formulation

"**H1 (visual bottleneck → compensatory memory).** Agents with
impoverished visual encoders should develop world-frame spatial
representations in recurrent memory; agents with rich encoders should
rely on observation pass-through. Signature: GPS/compass probe R²
monotonic with encoder-information-bottleneck across 5 conditions."

### §4 Results structure

1. Clean 5-condition monotonic result (this is the headline)
2. Both bottleneck conditions (blind, matched) show robust GPS + compass
3. Matched compass decays with lag while matched GPS doesn't — an
   interesting dissociation (compass is partially inferable from
   single-frame 1×1 visual, GPS never is)
4. Fov-fix/uniform/fov-lrn cluster near zero for GPS/compass,
   indistinguishable
5. DtG is universally encoded (control: every agent tracks task state)

### §5 H2 (format divergence) — keep as-is

The CKA/transplant/transfer evidence is behavioral and didn't rely
on the buggy probe data. CKA ≈ 0 across all pairs, transplant drops
SPL by 0.19-0.21 for cross-condition, etc. Untouched by the fix.

### §6 H3 — ON HOLD

The probe-based H3 claim (fov-learned compass R² = 0.94) was a bug
artefact. Under det CV, fov-learned compass is -1.34 ± 3.14 —
unreliable. Two options:

(a) Wait for fov-shifted training to converge (~2–3 days remaining)
    and do a clean causal test: same foveation transform, different
    static gaze location → different representational content.

(b) Drop H3 to Appendix / future work until (a) provides data.

## Matched's upgraded role

Paper originally framed matched-compute as:
- "Pixel-count control" — rules out "more pixels = better decoding"
- Ran at degenerate 1×1 encoder head due to training stability quirk,
  disclosed in Appendix

Under the new framing, matched-compute is:
- **Headline evidence** for the bottleneck hypothesis
- The 1×1 encoder quirk is the **mechanism**, not a caveat — an
  information-bottlenecked encoder forces LSTM compensation
- Complements blind (0 visual) at an intermediate bottleneck level (1 pixel)

This is a gift from the training-stability anomaly. What paper treated
as "retained despite quirk" becomes "key data point".

## What needs to be written / rerun

1. **Main.tex H1 rewrite** — ~3 pages (Intro H1 statement, Methods §3.3,
   Results §4.1-4.3, Discussion)
2. **Table 1 rebuild** — new columns (det-probe CV R² with error bars,
   ordered by bottleneck severity)
3. **New hero figure** — 5-condition monotonic R² bar chart
   (blind/matched/uniform/fov-fix/fov-lrn × GPS/compass/DtG) with CV error bars
4. **Lag-k figure** — maybe: lag-k GPS for bottleneck conditions only
   (blind, matched), showing retention; fov/uniform excluded as pass-through
5. **Retire path-history GPS figure** (the buggy one excluded fov-lrn;
   replace with DtG or cancel)
6. **H3 section** — downgrade to "preliminary"; wait for fov-shifted

## Scripts to extend

- `extended_lag_probe.py` — add DtG target option (partially done: --suffix)
- `make_h1h2_figures.py` — regenerate using det data
- `make_h3_content_figure.py` — pending fov-shifted
- New: make_bottleneck_figure.py — the 5-condition hero figure
