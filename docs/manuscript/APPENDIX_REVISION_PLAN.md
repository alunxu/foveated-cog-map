# Appendix revision plan — 2026-05-07

Comprehensive plan for revising appendix figures + text using the same revision strategies established for §3 (trim, flow, fix stale data, flag negative results positively, consolidate, kill clichés/enumerations).

**Status**: Plan only — no execution yet. User is asleep, will wake to review.

---

## Current appendix structure (audit)

| § | Topic | Figures | Status |
|---|-------|---------|--------|
| A | Reproducibility | figa1 (training curves) | ✅ clean |
| B | Probing protocol | (no figs) | ✅ clean |
| C | Alternative-account rebuttals | (no figs) | ⚠ may have stale refs |
| D | Magnitude diagnostics | figa2a, figa3, figa4a/b, figa5, figa6 | ⚠ several use 4-cond data |
| E | Spatial Format | figa7, figa8, figa9, figa10, figa11, figa12, figa13, figa14, figa15 | ⚠ multi-issue (see below) |
| F | Temporal Format | figa_autocorr, figa_persistence_betti, figa_persistence_wasserstein | ✅ recently updated |
| G | Consumption | fig7a/b (boundary), figa17 (catalog), figa18 (transplant sweep) | ⚠ figa17 4-cond, fig7a/b 5-cond? |
| H | Foveation status | (no figs in current scan) | ⚠ check |

---

## Critical issues to fix

### MAIN-TEXT DISCUSSION ISSUES (carryover from §3 audit)

**fig8_synthesis (Discussion §5.1) — CRITICAL: Coarse is missing**
- Currently shows 4 conditions (Blind, Foveated, Fov-LP, Uniform). Coarse is data-missing.
- Local transplant data (`data/transplant/`) only has `foveated_to_*` and `foveated_learned_to_foveated_*` — no `coarse_to_*` or `*_to_coarse`.
- The script (`make_synthesis_figure.py`) reads from `/tmp/transplant_local/<donor>_to_<recipient>_mid30.json` which is the 5-cond cache that lives on RCP.
- **Fix**: pull 5×5 mid30 transplant cache from RCP (same kubectl approach we used for fov-LP shortcut), re-render fig8 with all 5 conds.
- **Plus**: drop "H1 magnitude / H2 format isolation" axis labels — the H1/H2/H3 nomenclature was retired earlier in the paper. Use "Magnitude / Format isolation / Consumption" instead.

**Discussion text (§5.1–§5.4)**
- §5.1 Synthesis: 1 dense paragraph; trim, flow.
- §5.2 Bio precedent: has `\emph{(i)} \emph{(ii)} \emph{(iii)}` enumerations in two places ("Two specific bio-AI bridges land cleanly..." + "Three falsifiable predictions follow"). Drop enumerations; flow as connected prose.
- §5.3 Implications: has `\emph{(1)} \emph{(2)} \emph{(3)}` Memory-Maze validation enumerations. Same fix.
- §5.4 Limitations: has `\emph{(i)/(ii)/(iii)}` enumerations. Compress into 2 flowing paragraphs.
- §5 Conclusion: tighten, add positive framing of unexpected findings if not already.

---

### APPENDIX FIGURES — STALE DATA AUDIT

| Fig | Status | Issue | Fix |
|-----|--------|-------|-----|
| **figa1** training curves | ✅ | none | keep |
| **figa2a** H1 mega-panel | ⚠ | 5-panel composite of magnitude diagnostics; check fov-LP coverage | re-render with 5-cond cache |
| **figa3** per-layer probe | ❌ | caption says "all 4 conditions" — stale | re-render + caption to 5 conds |
| **figa4a** layerwise decay | ⚠ | check fov-LP | re-render |
| **figa4b** MP3D generalisation | ⚠ | check fov-LP | re-render |
| **figa5** substitution dynamics | ⚠ | check fov-LP | re-render |
| **figa6** MINE capacity | ✅ | already 5-cond per caption (Blind 6.01, Coarse 4.55, Fov 4.30, Fov-LP 4.66, Uniform 4.47) | keep |
| **figa7** 5×5 transplant matrix | ⚠ | should be 5×5 — check it really is | verify, re-render if needed |
| **figa8** CKA heatmap | ⚠ | check 4 vs 5 conds | re-render |
| **figa9** population coding | ⚠ | check fov-LP | re-render |
| **figa10** goal-vector probe | ⚠ | check fov-LP | re-render |
| **figa11** t-SNE | ⚠ | "4 conditions" probably stale | re-render |
| **figa12** position-axis | ⚠ | "all four conditions" — stale | re-render with 5 conds |
| **figa13** PC cumulative | ⚠ | check fov-LP | re-render |
| **figa14** β-scrubbing | ⚠ | "foveated deferred" — incomplete | run foveated; verify others |
| **figa15** eigenspectrum | ⚠ | check fov-LP | re-render |
| **figa_autocorr** Murray τ | ✅ | recently created with 5 conds | keep |
| **figa_persistence_betti** | ⚠ | check 5 conds | verify |
| **figa_persistence_wasserstein** | ⚠ | check 5 conds | verify |
| **fig7a** memory-init transplant | ⚠ | now in App G; check fov-LP | re-render |
| **fig7b** GPS-compass ablation | ⚠ | now in App G; check fov-LP | re-render |
| **figa17** shortcut catalog | ❌ | "4 cases per condition" but caption says 4 conds — fov-LP missing | now we have fov-LP shortcut data; can add 5th row |
| **figa18** transplant sweep | ✅ | 2 representative pairs only (intentional) | keep |

---

## Consolidation opportunities (info density ↑, fragmentation ↓)

### Proposal A: Magnitude appendix
- **figa2a is already a 5-panel mega**. Good.
- **figa3, figa4a, figa5** all show layerwise/training probe diagnostics → could consolidate into one 4-panel mega: "Magnitude diagnostics: layer-wise + training-trajectory + MP3D generalisation".
- **Pros**: fewer figures, more coherent narrative.
- **Cons**: requires script work.
- **Recommendation**: low priority; current structure is acceptable.

### Proposal B: Spatial Format appendix
- **figa7** (5×5 transplant) + **figa8** (CKA) + **figa11** (t-SNE) all measure cross-condition format divergence by different metrics → **consolidate into single figure "Format divergence: 4 metrics"** with a subgrid.
- **figa9** (population coding) + **figa12** (position axis) + **figa13** (PC cumulative) → **consolidate into "Capacity-allocation geometry"** (3 panels).
- **figa10** (goal vector) — standalone, fine.
- **figa14** (β-scrubbing) — niche; can stay standalone or fold into capacity-allocation panel.
- **figa15** (eigenspectrum) — important; standalone OK.
- **Recommendation**: do the consolidation. Reduces figa7-figa14 (8 figures) to ~4 figures with denser content.

### Proposal C: Consumption appendix
- **fig7a + fig7b** (boundary checks) already 2-panel. Keep.
- **figa17** (shortcut catalog) — extends main fig:consumption(b). Keep but add fov-LP row (5 conditions).
- **figa18** (transplant sweep) — defends t=30 choice. Keep.
- **Recommendation**: minimal restructure; just add fov-LP to figa17.

---

## Experiments needed

### Local-only (no RCP):
1. **Re-render any figure** whose data is already in local cache. Most legacy 4-cond figures probably need new data; need to check `results/cogneuro_data/` and similar local caches.
2. **figa14 foveated β-scrubbing**: if h2 data for foveated is local, can compute. Check.

### Need RCP jobs / data pulls:
1. **5×5 transplant mid30 cache**: pull from RCP `/scratch/wxu/.../transplant_*` (15min, kubectl staging pod).
2. **Possibly: any 5-cond probing/h2 caches** if local copies are stale. Likely already local under `/tmp/rcp_analysis_v3/` per recent work.
3. **figa3 per-layer probe** for 5 conds: probe at h0/h1/h2 + c0/c1/c2 for each cond. If data not cached, ~30min job.

### Long experiments to skip (per user instruction "unless particularly long"):
- Re-collecting all probing data from scratch (would need re-running probing pipeline, hours per condition)
- Multi-seed grid (mentioned as future work)
- Architecture-comparison (transformer)

---

## Text + caption revision strategies (apply throughout appendix)

### Drop:
- `\emph{(i)} ... \emph{(ii)} ... \emph{(iii)} ...` enumerations → flowing prose
- Italics for emphasis on plain words
- "single seed per condition" boilerplate (state once globally in §5.4 limitations)
- "(referenced from §X)" if context already makes the reference clear

### Tighten:
- Long captions describing both methodology and result → split: methodology in 1 sentence, then result.
- Repetitive setup descriptions ("5-fold CV, deterministic rollouts, ..." stated 5+ times) → state once in App B (probing protocol), refer to it.
- Numeric details that don't affect interpretation can move to caption-only (out of body).

### Add (positively framed):
- For each unexpected finding (sub-chance R² in rich-encoder, blind's spatial-info distribution being flatter, etc.), write 1 sentence on what it would have predicted vs what we found, framing the divergence as a sharpening.

### Caption ↔ figure consistency check (per figure):
- Number of conditions in figure must match caption.
- Numbers cited in caption must match what the data actually shows.
- Subpanel labels (a)/(b)/(c) must match the panel order.

---

## Recommended execution order (when starting work)

### Phase 1 — Critical (must do)
1. Pull 5×5 mid30 transplant cache from RCP (kubectl staging pod, ~10 min).
2. Re-render fig8_synthesis with 5 conditions, fix axis labels (drop H1/H2 nomenclature).
3. Trim Discussion §5.1–§5.4: kill enumerations, flow as prose, in-place positive framings.
4. Tighten Conclusion.

### Phase 2 — High value (should do)
5. Audit each appendix figure: render at high-res, count conditions, compare to caption.
6. Re-render any figure whose data caches now have 5-cond entries (likely most of figa3-figa15).
7. Tighten captions: drop boilerplate, drop enumerations, ensure caption ↔ figure consistency.

### Phase 3 — Consolidation (nice to have, time-permitting)
8. Consolidate figa7+figa8+figa11 into "Format divergence: convergent metrics" (3-panel).
9. Consolidate figa9+figa12+figa13 into "Capacity-allocation geometry" (3-panel).
10. Add fov-LP row to figa17 shortcut catalog (now that we have fov-LP shortcut data).

### Phase 4 — Experiments (only if data-gap critical)
11. figa14 foveated β-scrubbing if local h2 is available.
12. Any figa-* whose data we can compute locally from existing 5-cond NPZs.

---

## Open questions for user (when awake)

1. **Synthesis figure axis nomenclature**: replace "H1 magnitude / H2 format isolation" with what? Suggest "Magnitude / Format isolation / Consumption" or "Probe-readability / Transplant cost / Policy reliance"?
2. **Consolidation aggressiveness**: collapse 8 spatial-format figures into 4? Or keep current granularity and just tighten?
3. **figa14 foveated β-scrubbing**: currently "deferred". Add it now if data available, or leave deferred?
4. **figa17 shortcut catalog**: currently 4 conds × 4 cases. Add fov-LP row → 5 conds × 4 cases. Or replace catalog with something more focused?
5. **Conclusion length**: current is 2 paragraphs. Trim to 1 paragraph? Or expand with explicit list of unexpected-but-supportive findings?

---

## Constraints noted from session memory
- Storage policy: RCP only. Don't copy large checkpoints/datasets to local. Pull only the small cached results needed.
- Work autonomously per user preference. Just submit jobs; don't ask permission.
- Single seed per condition is the convention (per Wijmans 2023 et al). Don't try to add multi-seed unless data exists.
- Deadline 2026-05-06 has passed; user is iterating on submission/revision.

