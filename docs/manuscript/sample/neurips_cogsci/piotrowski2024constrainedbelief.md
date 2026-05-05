# Constrained belief updates explain geometric structures in transformer representations

- **Authors**: Mateusz Piotrowski, Paul M Riechers, Daniel Filan, Adam S Shai
- **Venue**: NeurIPS 2024 NeuroAI workshop (oral); also ICML 2025
- **Year**: 2024 (workshop), 2025 (full)
- **arXiv**: https://arxiv.org/abs/2502.01954
- **NeurIPS workshop talk**: https://neurips.cc/virtual/2024/101437

## Abstract (paraphrased)
Extends Shai et al. 2024 (belief geometry in residual streams). Shows that transformers implement *constrained* Bayesian belief updating — a parallelised approximation shaped by attention's architectural constraints — and predicts in detail the attention patterns, OV-vectors, and embedding geometry from a modified Bayesian-prediction equation. Tightly couples computational-mechanics theory with mechanistic-interpretability evidence.

## Structure analysis
- **§II.1 Related Work**: **Yes, sub-numbered** standalone (within §II Background), three thematic subsections (~3 paragraphs each): "Features as directions," "From features to circuits," "Belief state geometry and computational mechanics." Total ~1000 words.
- **Intro**: ~3-4 paragraphs, **claim-first** opening with a question ("What computational structures emerge in transformers trained on next-token prediction?"). **3 explicit numbered contributions**.
- **Where prior work appears**: Concentrated §II.1; inline citations in intro and methods; minimal in Discussion.
- **Bio-AI bridge style**: **Dedicated subsection §II.2 "Optimal Prediction and Belief State Geometry"** — self-contained mathematical exposition of HMM/computational-mechanics framework before connecting to neural networks. The bridge is theoretical, not empirical-bio.
- **Section ratios**: Intro ~8%; Background+Related ~15%; Methodology §III ~4%; Results §IV (5 subsections) ~55%; Discussion §V ~10%; appendices ~8%. Heavy results.
- **Methods presentation**: **Inline at point-of-use** — primary mechanistic procedures embedded inside §IV.2-IV.4 results, with theoretical derivations introduced as each result demands them. Brief §III "Methodology" lists data-generation only.
- **Results vs Discussion**: **Merged in spirit** — each results subsection (IV.1–IV.5) contains both empirical observation and theoretical prediction. §V Discussion is short and forward-looking, not a separate findings synthesis.
