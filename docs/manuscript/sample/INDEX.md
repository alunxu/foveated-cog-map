# Sample literature index

23 papers, organised by bucket. All entries are *not* currently in `docs/manuscript/literature.bib` (gap-fill only). Each entry below points to its `.md` metadata file in this folder.

---

## Bucket A. Emergent maps in deep-RL navigation agents

- `schaeffer2022nofreelunch.md` -- **No Free Lunch from Deep Learning in Neuroscience** (Schaeffer, Khona, Fiete; NeurIPS 2022). 11k networks, <10% yield grid-like units; emergent grids are post-hoc. Stronger version of `schaeffer2023fiction` already cited.
- `shai2024beliefgeometry.md` -- **Transformers Represent Belief State Geometry in their Residual Stream** (Shai et al.; NeurIPS 2024). Linear probes recover full belief-state geometry in transformers; precedent for our linear-probing of integration code.
- `bar2024navigationworldmodels.md` -- **Navigation World Models** (Bar, Zhou, Tran, Darrell, LeCun; CVPR 2025). 1B-param diffusion-transformer NWM that puts the navigation world model in the *encoder* side; direct contrast to LSTM PointGoal.

## Bucket B. Encoder-memory pipelines / capacity in modern embodied AI

- `orozco2025emergentopenvla.md` -- **Emergent World Representations in OpenVLA** (Orozco et al.; arXiv 2509.24559). Probes OpenVLA, finds linear-readable world model in middle layers, framed via Koopman theory + SAE. The closest contemporary mirror of our methodology.
- `assran2025vjepa2.md` -- **V-JEPA 2** (Assran, Bardes, Fan et al.; Meta FAIR 2025). Latent-prediction video model + zero-shot Franka planning; cleanest 2025 JEPA encoder-side world model.
- `samsami2024recall2imagine.md` -- **Mastering Memory Tasks with World Models** (Samsami, Zholus, Rajendran, Chandar; ICLR 2024 oral). DreamerV3 + S3M; deliberate memory-side scaling, useful counterpart to our LSTM analysis.
- `bar2024navigationworldmodels.md` -- (also Bucket A; spans both)
- `cetin2024spatiallyaware.md` -- **Spatially-Aware Transformer for Embodied Agents** (Cho et al.; arXiv 2402.15160, 2024). Place-anchored episodic memory in transformers; architectural test of our hypothesis.
- `duan2024statelens.md` -- **Do LLMs Build World Representations? Probing Through the Lens of State Abstraction** (Hu et al.; NeurIPS 2024). Goal-relevant abstraction dominates; aligns with our magnitude-vs-format split.

## Bucket C. Spatial reasoning failures in VLMs / probing of visual encoders

- `zhao2024physbench.md` -- **PhysBench** (Chow et al.; ICLR 2025). 10k-entry physical-world VLM benchmark, frontier VLMs fail; encoder-bottleneck argument. Newer + broader than VSI-Bench.

(Three further VLM/encoder papers are already in our bib and not duplicated here: `tong2024cambrian`, `tong2024eyeswideshut`, `fu2024blink`, `cheng2024spatialrgpt`, `elbanani2024probing3d`, `man2024lexicon3d`, `majumdar2023vc1`, `chen2024spatialvlm`, `yang2024thinkinginspace`, `kamath2023whatsup`, `liu2023vsr`, `ramakrishnan2025space`. PhysBench is the standout missing piece.)

## Bucket D. Cognitive maps / hippocampus / hidden-state inference / comparative cognition

- `whittington2022transformerhippocampus.md` -- **Relating transformers to models and neural representations of the hippocampal formation** (Whittington, Warren, Behrens; ICLR 2022). TEM-transformer; only ICLR work directly equating transformers with hippocampal grid/place cells.
- `whittington2025composition.md` -- **Constructing future behavior in the hippocampal formation through composition and replay** (Whittington et al.; Nature Neuroscience 2025). Successor to TEM; compositional state spaces + replay-induced firing fields.
- `dorrell2024taleoftwo.md` -- **A tale of two algorithms: Structured slots explain prefrontal sequence memory and are unified with hippocampal cognitive maps** (Dorrell, El-Gaby, Behrens, Whittington; Neuron 2024). Activity-slot vs synaptic-storage duality -- the canonical 2024 mathematical scaffold for treating LSTM h_2 as a PFC-style cognitive map.
- `jian2024incontextepisodicmemory.md` -- **Linking In-context Learning in Transformers to Human Episodic Memory** (Ji-An et al.; NeurIPS 2024). CMR-like induction heads in LLMs causally support ICL; bridges our work and transformer-as-hippocampus tradition.
- `katayama2024beliefinference.md` -- **Belief inference for hierarchical hidden states in spatial navigation** (Katayama et al.; Communications Biology 2024). Hierarchical Bayesian model + fMRI of human Tiger maze navigation; complements Sanders et al. 2020.
- `khona2022attractor.md` -- **Attractor and integrator networks in the brain** (Khona, Fiete; Nature Reviews Neuroscience 2022). Authoritative review framing LSTM h_2 as an integrator; one-stop biology citation.
- `khona2025peakselection.md` -- **Global modules robustly emerge from local interactions and smooth gradients** (Khona, Chandra, Fiete; Nature 2025). Peak-selection principle for grid-cell module formation; emergent spatial coding requires structural constraints, not just task pressure.
- `garrett2024shortcutting.md` -- **Shortcutting from self-motion signals reveals a cognitive map in mice** (eLife 2024). Mice without landmarks build a cognitive map from self-motion alone; biological motivation for our shortcut-discovery (Tolman) consumption-axis test.

## Bucket E. Foveation / bandwidth-constrained vision in deep learning

- `schmidt2025segmentthisthing.md` -- **Segment This Thing: Foveated Tokenization for Efficient Point-Prompted Segmentation** (Schmidt et al.; CVPR 2025). 24x token reduction via biologically-inspired foveated tokeniser.
- `shi2024transnext.md` -- **TransNeXt: Robust Foveal Visual Perception for Vision Transformers** (Shi; CVPR 2024). Aggregated pixel-focused attention with foveal prior, competitive with global ViT attention.

(Already in bib: `deza2020foveated`, `min2022pervit`, `mnih2014attention`, `atanov2024photoreceptor`, `shang2023sugarl`, `pourrahimi2025foveated`, `rosenholtz2016peripheral`, `strasburger2011peripheral`. TransNeXt + SegmentThisThing are the freshest 2024-2025 entries we lack.)

## Bucket F. Probing methodology + representational geometry

- `geiger2023iia.md` -- **Causal Abstraction for Faithful Model Interpretation** (Geiger, Potts, Icard; theory companion to NeurIPS 2023 Boundless DAS). Unifies activation patching, DAS, causal mediation, scrubbing, SAE, steering as interchange interventions; supports our memory-transplant + ablation framing.
- `birdal2021idphpersistent.md` -- **Intrinsic Dimension, Persistent Homology and Generalization in Neural Networks** (Birdal, Lou, Guibas, Şimşekli; NeurIPS 2021). PH-dimension as a more robust ID estimator than TwoNN; the methodological caveat reviewers will press us on.
- `ostrow2023dsa.md` -- **Beyond Geometry: Dynamical Similarity Analysis** (Ostrow, Eisen, Kozachkov, Fiete; NeurIPS 2023). DSA as a dynamics-aware Procrustes; the canonical 2023 NeurIPS rebuttal to geometry-only similarity measures.
- `cloos2024evaluatingrepsim.md` -- **Evaluating Representational Similarity Measures from the Lens of Functional Correspondence** (Cloos et al.; arXiv 2024). Empirical benchmark of 8 similarity metrics; linear CKA + Procrustes win, justifying our choices.

---

## Summary statistics

- **Total**: 23 metadata files (no duplicates of entries already in `literature.bib`).
- **Year mix**: 1 from 2021, 2 from 2022, 1 from 2023 (+ implicit Geiger 2023), 11 from 2024, 6 from 2025.
- **Venue mix**: NeurIPS (8), ICLR (3), CVPR (3), Nature/NN/Cell-family (5), eLife (1), Communications Biology (1), arXiv-only preprint (3 -- but all are NeurIPS/ICLR-route VLA / world-model preprints).
- **Bucket coverage**: A=3, B=5, C=1 (plus 11 already in bib), D=8, E=2, F=4.

## Notes for future use

- Several entries (`bar2024navigationworldmodels`, `orozco2025emergentopenvla`, `dorrell2024taleoftwo`, `shai2024beliefgeometry`, `khona2025peakselection`) are likely the *highest-leverage* additions. They are recent, top-venue, and structurally aligned with our magnitude-format-consumption framing.
- We did not download PDFs (the user said it was OK to skip). Each metadata file has the canonical URL; PDFs can be fetched on demand.
- Bucket C looks "thin" in the sample because our existing bib already contains 11 of the most-relevant VLM-spatial-failure papers (Tong, Fu, Cheng, etc.). PhysBench is the principal addition.
