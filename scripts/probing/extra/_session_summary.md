# 2026-05-06 deadline session — cogneuro extension summary

## What we set out to do

The user asked: "introduce more cognitive neuroscience traditional methods/
frameworks into DL [...] would significantly increase contribution / impact"

Two parallel tracks tonight:
- **E (world-model probe)**: architecture-agnostic falsifier of our Habitat
  bandwidth-allocation prediction, on a different env (Memory Maze) and
  encoder family (DINOv2-Base + small LSTM).
- **Cogneuro frameworks**: literature-search + implementation of cogneuro
  analyses that map onto LSTM PointNav agents.

## Cogneuro deliverables

**12-candidate ranked literature shortlist** from background research agent
(`docs/manuscript/sample/cogneuro_frameworks/`), top-3 picks:
1. CCGP (Bernardi+2020 Cell)
2. TGM (King-Dehaene 2014 TICS)
3. MFTMA (Chung+2018, MEDIUM risk per HP-shopping)

**Implemented & verified (this session):**

| Method | Code | Result | Paper status |
|--------|------|--------|--------------|
| **TGM** | `temporal_generalisation.py` | **STRONG** — blind diagonal (transient code), coarse block (sustained code), sighted-rich graded between. Code half-life: coarse > uniform > fov-LP > foveated > blind. | **§3.5 + Fig fig:tgm** |
| **Murray timescales** | `intrinsic_timescales.py` | **STRONG** — blind median tau=19.3 steps, sighted ~7.7-9.2 (2.1-2.5x shorter). Mirrors cortical-hierarchy fast-sensor / slow-integrator. | **§3.5 + Fig fig:timescales** |
| **Tangling Q** | `tangling.py` | Modest — coarse outlier (Q=0.08), other 4 conds at 0.045-0.058. Consistent with PR/ID coarse outlier. | Supplement (committed) |
| **CCGP** | `ccgp_abstraction.py` | Noisy — blind highest WCV for pos_x_bin (consistent with paper's "blind compensates"), coarse highest abstraction index across multiple targets, but ordering doesn't match pre-reg. | Supplement (committed) |

**Pre-registration discipline:**
- Each method has a markdown spec under `cogneuro_frameworks/` filed BEFORE
  code ran, including predicted result shape and decision rules.
- All four methods produced results that PARTIALLY differ from pre-reg.
- Per pre-reg policy, we report what we found, not iterate on HPs.

## E (world-model probe) status

- Pipeline: 7 scripts in `scripts/probing/world_model_probe/`
- Pre-registration: `PRE_REGISTRATION.md`
- Smoke validated at N=50 traj
- Main run (600 train + 200 eval traj × 5 sensor-resolution conditions on
  DINOv2-Base + small LSTM): in progress at LSTM training stage as of last
  check.
- ETA: ~30-60 min more to LSTM convergence + probe.

## Bib additions

5 references added to `literature.bib`:
- King & Dehaene 2014 TICS (TGM)
- Bernardi et al. 2020 Cell (CCGP)
- Pasukonis et al. 2023 ICLR (Memory Maze)
- Murray et al. 2014 Nat Neurosci (timescales)
- Russo et al. 2018 Neuron (tangling)
- Oquab et al. 2024 (DINOv2)

## Paper changes

- **§3.5 [NEW]**: Format axis (temporal) — TGM paragraph + Fig fig:tgm,
  Timescales paragraph + Fig fig:timescales (39 pages total, was 37).
- **§5.2**: Added two bio-AI bridge sentences referencing King-Dehaene MEG
  decoding and Murray cortical hierarchy.
- Pre-existing Δ unicode bug fixed at line 270.
- amssymb package added (was missing; needed for $\lesssim$).

## What's NOT in the paper (reasons)

- **Tangling Q**: pattern noisy (only coarse separates, others all similar);
  duplicates information in the existing PR/ID/Procrustes geometry table.
- **CCGP**: WCV finding duplicates magnitude-axis claim; AI ordering
  contradicts pre-reg, requires careful interpretation; saved as supplement.

## Files committed

```
docs/manuscript/main.tex                 # +§3.5 + §5.2 update
docs/manuscript/literature.bib           # +6 references
docs/manuscript/fig/cogneuro/            # 7 figs/JSONs
docs/manuscript/sample/cogneuro_frameworks/  # 14 lit-search docs
docs/manuscript/sample/world_model_probe/    # 10 lit-search docs

scripts/probing/extra/
  ccgp_abstraction.py + plot_ccgp.py
  temporal_generalisation.py + temporal_generalisation_v2.py + plot_tgm.py
  intrinsic_timescales.py + plot_timescales.py
  tangling.py + plot_tangling.py
  _tgm_paragraph.md + _session_summary.md (this file)

scripts/probing/world_model_probe/       # 8 scripts + PRE_REGISTRATION
```
