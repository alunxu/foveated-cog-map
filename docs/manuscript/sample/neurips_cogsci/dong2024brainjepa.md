# Brain-JEPA: Brain Dynamics Foundation Model with Gradient Positioning and Spatiotemporal Masking

- **Authors**: Zijian Dong, Ruilin Li, Yilei Wu, Thuan Tinh Nguyen, Joanna Su Xian Chong, Fang Ji, Nathanael Ren Jie Tong, Christopher Li Hsian Chen, Juan Helen Zhou
- **Venue**: NeurIPS 2024 (Spotlight)
- **Year**: 2024
- **arXiv**: https://arxiv.org/abs/2409.19407
- **NeurIPS PDF**: https://proceedings.neurips.cc/paper_files/paper/2024/file/9c3828adf1500f5de3c56f6550dfe43c-Paper-Conference.pdf
- **GitHub**: https://github.com/Eric-LRL/Brain-JEPA

## Abstract (paraphrased)
Adapts the JEPA (joint-embedding predictive architecture) self-supervised paradigm to fMRI brain-dynamics modelling. Two innovations: (i) Brain Gradient Positioning, a functional-coordinate-system positional embedding for ROIs based on Mesulam's hierarchy, (ii) Spatiotemporal Masking tailored to fMRI's heterogeneous time-series patches. Achieves SOTA on demographic prediction, disease diagnosis/prognosis, and trait prediction with strong cross-ethnic generalisation.

## Structure analysis
- **§2 Related Work**: **Yes, standalone**, two subsections ("Task-specific models for fMRI" and "fMRI Foundation Model"). ~4 paragraphs.
- **Intro**: ~10 paragraphs (long), **motivation-first** with cogsci framing (fMRI for understanding cognition → existing limitations → BrainLM gap → Brain-JEPA). **No explicit numbered contribution list** — contributions integrated narratively as "To address these gaps, here we introduce Brain-JEPA..."
- **Where prior work appears**: Concentrated in §2; inline in §3 (each technical choice — gradient positioning, masking — justified with neuroscience citations: Mesulam hierarchy, fMRI coherence literature); occasional in interpretation §4.7.
- **Bio-AI bridge style**: **Inline within methods** (gradient positioning literally encodes neuroanatomical hierarchy). Interpretation subsection §4.7 maps attention patterns to known cognitive networks (DMN, CN, SAN). No dedicated bio-bridge subsection; the architecture *is* the bridge.
- **Section ratios** (estimate): Intro ~15%; Related Work ~8%; Methods (§3) ~25%; Experiments (§4 with 7 subsections) ~40%; Conclusion/Limitations ~7%; appendices large (~50 pages).
- **Methods presentation**: **Standalone §3** with full pipeline figure, then subsections (gradient positioning, spatiotemporal masking, training).
- **Results vs Discussion**: **Merged** within §4 — results subsections include interpretation (e.g., §4.7 "Interpretation" analyses attention vs known brain networks). No separate Discussion section; Conclusion is short.
