# Temporal-Difference Learning Using Distributed Error Signals

- **Authors**: Jonas Guan, Shon Eduard Verch, Claas Voelcker, Ethan C. Jackson, Nicolas Papernot, William A. Cunningham
- **Venue**: NeurIPS 2024 (poster)
- **Year**: 2024
- **arXiv**: https://arxiv.org/abs/2411.03604
- **OpenReview**: https://openreview.net/forum?id=8moTQjfqAV
- **GitHub**: https://github.com/social-ai-uoft/ad-paper

## Abstract (paraphrased)
Asks whether the spatially homogeneous distribution of dopamine in the nucleus accumbens (NAc) — which seems to preclude back-prop-style credit assignment — could nevertheless support complex non-linear reward learning. Introduces "Artificial Dopamine" (AD), a deep-Q-learning algorithm with synchronously distributed per-layer TD error signals (no back-prop through layers), and shows it matches standard deep-RL on MinAtar, DMC, and classic control benchmarks.

## Structure analysis
- **§2 Related Work**: **No standalone Related Work.** §2 is titled "**Background**" with three subsections (2.1 Reward-Based Learning in NAc, 2.2 TD Learning, 2.3 Forward-Forward) that fold prior work into mechanism explanation.
- **Intro**: ~5 paragraphs, **motivation-first** with biological puzzle framing (dopamine homogeneity vs back-prop credit assignment). Ends with **3 numbered/bulleted contributions**.
- **Where prior work appears**: Heavy in intro (biological context); §2 Background (technical foundations); minimal in results; competing biological theories returned to in §6 Limitations/Discussion.
- **Bio-AI bridge style**: **Hybrid** — *dedicated subsection* (§2.1 simplifies NAc reward learning); *inline parentheticals* throughout methods ("AD cell mirrors local homogeneous dopamine distribution"); *footnote* hedges acknowledging competing accounts.
- **Section ratios**: Intro ~10%; Background §2 (with bio sub-subsection) ~12%; Methods §3 (AD architecture) ~15%; Results §5 ~25%; **Limitations §6 ~12%** (unusually long for NeurIPS, foregrounded as standalone section, not appendix); appendices ~rest.
- **Methods presentation**: **Standalone §3 "Artificial Dopamine"** (3.1 cell internals, 3.2 network connections), presented before experiments.
- **Results vs Discussion**: **Strictly separate.** §5 Results is empirical; §6 is a standalone, substantial Limitations / Discussion section that explicitly addresses biological plausibility caveats — unusual for an ML conference, signals neuroscience-cognizant scholarship.
