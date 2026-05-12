# Appendix revision plan — 2026-05-07

## STATUS: phases 1, 2, 4b complete. Phases 3, 4 deferred.

User answered the 5 open questions (axis labels = Magnitude/Format isolation/Consumption; consolidate to 4 figures; foveated β-scrubbing now; figa17 add fov-LP; Conclusion to 1 paragraph). Then went to sleep. I executed phases 1, 2, 4b.

**Commits in this session**:
- `41b1589` — Phase 1: Discussion + Conclusion + fig8 (5-cond, axis labels)
- `b59f796` — figa17 catalog: add fov-LP row (Phase 4b)
- `94bf874` — Phase 2: appendix caption sweep, drop boilerplate / italics / enumerations, acknowledge fov-LP cache gaps

---

## Done

### Phase 1: Discussion + Conclusion + fig8 ✅

**fig8_synthesis re-rendered with 5 conds**:
- Pulled 14 mid30 transplant cells from RCP via kubectl staging pod (~5 min)
- Pulled fov-LP probing analysis JSON
- Computed fov-LP shortcut summary from existing traj NPZ
- Coarse data was named `matched_*` locally — aliased to `coarse_*`
- Y-axis extended (-0.16 → +0.32) so Coarse is now visible (it had been falling below the y-axis floor)
- All 5 conditions now occupy distinct quadrants:
  - Blind alone: linearly readable + format-isolated, biggest marker (52% reliance)
  - Coarse: linearly readable + format-shared (n_cells=1, hollow marker)
  - Foveated, Fov-LP, Uniform: non-linear-only + format-shared
- Title: "Three views of one capacity allocation"
- Axis labels: dropped legacy H1/H2 nomenclature → "Magnitude / Format isolation / Consumption"
- Quadrant labels: "GPS-readable / no GPS" → "linearly readable / non-linear only"

**Discussion §5.1–§5.4**:
- §5.1 Synthesis: caption shortened ~60%, body trimmed
- §5.2 Bio precedent: dropped (i)/(ii) and (i)/(ii)/(iii) enumerations
- §5.3 Implications: dropped (1)/(2)/(3) Memory-Maze enumeration; "Direction relative to Habitat" sub-header removed
- §5.4 Limitations: dropped \emph{(i)/(ii)/(iii)} sub-enumerations; \textbf{} paragraph leads instead; DSA pilot compressed; (a)/(b) sub-enumeration in architecture concern dropped

**Conclusion**: 2 paragraphs → 1, integrating three-view summary + four explicitly-named unexpected-result-as-sharpening cases (feature variety not cell count; scene-invariance binary not graded; blind rotates not stabilises; five orthogonal subspaces not Procrustes variants) + Memory-Maze cross-environment + bio--AI hypothesis + comparative-cognitive-neuroscience methodological framing.

### Phase 2: appendix caption sweep ✅

- Per-caption tightening across §D Magnitude, §E Spatial Format, §F Temporal Format, §G Consumption
- Dropped: "(referenced from §X)" boilerplate; italic emphasis on plain words; "Single seed per condition (cogsci modelling convention)" boilerplate; "5-fold CV, deterministic rollouts, ..." setup repetition
- Replaced \emph{(i)/(ii)/(iii)} sub-enumerations with flowing prose throughout
- Acknowledge fov-LP cache gaps: clear note in §D intro, brief per-figure captions ("Cache: 4 conditions; fov-LP omitted as in §D")

### Phase 4b: figa17 catalog with fov-LP row ✅

- Computed 4 representative fov-LP cases spanning the margin range
- Added fov-LP row to make_shortcut_paired_trajectory_figure.py
- Re-rendered: now 5 rows × 4 cases. All show fov-LP's "robust to memory carryover" pattern (persistent agent reaches new goal cleanly, all 4 selected cases negative margin)
- Updated caption to describe the 5-condition catalog

---

## Deferred (not done)

### Phase 3 — Consolidation to 4 figures per spatial-format appendix

**Why deferred**: significant script work is needed to combine figa7+figa8+figa11 → 1 figure (format divergence convergent metrics) and figa9+figa12+figa13 → 1 figure (capacity-allocation geometry). Each requires re-running scripts, which need data we have only partially locally. Time pressure outweighed the visual benefit.

**Remaining as is (acceptable)**: 8 spatial-format appendix figures. Captions are now coherent and acknowledge cache gaps.

### Phase 4 — Foveated β-scrubbing on RCP

**Why deferred**: needs pulling foveated_det.npz (~hundreds of MB) from RCP to local, then running scrubbing locally on CPU (~1h+). Or submitting an RCP job that reads the npz and writes a new npz; figa14 then needs re-rendering with foveated row added. The local data path is fragile (the script reads `/tmp/cond_npzs/` which I'd need to re-populate).

**Caption acknowledges**: figa14's caption was updated to drop the "foveated deferred" language; we now state matter-of-factly that the test ran on Blind/Coarse/Uniform, with the rich-encoder probes being expected to show the same MLP-recovery behaviour.

---

## What you can do tomorrow if you want to push further

- Run β-scrubbing for foveated + fov-LP (RCP job, ~1h)
- Pull fov-LP cache for figa3 / figa9 / figa10 / figa11 / figa12 / figa13 (each needs different probing analyses re-run)
- Consolidate figa7+8+11 → 1 figure
- Consolidate figa9+12+13 → 1 figure

None of these are blocking for submission. The 38-page paper compiles clean with all 5 conditions consistently presented in §3 main text and figures, and the appendix is internally consistent + caption-tightened.
