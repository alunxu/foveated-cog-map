# Predictive Attractor Models

- **Authors**: Ramy Mounir, Sudeep Sarkar
- **Venue**: NeurIPS 2024 (poster)
- **Year**: 2024
- **arXiv**: https://arxiv.org/abs/2410.02430
- **NeurIPS PDF**: https://proceedings.neurips.cc/paper_files/paper/2024/file/5df4313ecd4875931fbdacc486cc1fcf-Paper-Conference.pdf
- **GitHub**: https://github.com/ramyamounir/pam

## Abstract (paraphrased)
Introduces PAM, a sequence memory architecture inspired by HTM/predictive-coding theories of cortex: it learns sequences online through Hebbian local rules, avoids catastrophic forgetting via lateral inhibition in cortical mini-columns, and produces multi-modal future predictions through an attractor model trained alongside a predictor. Empirically demonstrates competitive single-pass online sequence learning with biologically plausible learning signals.

## Structure analysis
- **§2 Related Work**: **Yes, standalone**, ~10 paragraphs across three subsections (Predictive Coding ~4 par, Fixed-Point Attractor Dynamics ~3 par, Predictive Learning ~3 par).
- **Intro**: ~5 paragraphs, **motivation-first** (opens with the cognitive function of sequential memory), then a 5-item bulleted **desiderata** list, and ends with **3 numbered contributions**.
- **Where prior work appears**: Concentrated in §2, with selective inline citations in intro (motivation) and methods (when grounding mechanism in HTM). Few citations in results.
- **Bio-AI bridge style**: Distributed — *inline parentheticals* in intro and methods preliminaries; *dedicated mini-subsections* in §3.2 (SDR) and §3.3 (lateral inhibition) tying components to neurobiology. **No standalone Discussion**.
- **Section ratios** (paragraph-rough): Intro ~5 par (~8%); Background ~10 par (~12%); Methods/PAM ~25 par (~35%); Experiments ~15 par (~25%); Conclusion ~2 par (~3%); Limitations in **Appendix A only** (not in main text).
- **Methods presentation**: Standalone §3 with subsections (SSM formulation, SDR, sequence learning, sequence generation), algorithms presented before any experiment.
- **Results vs Discussion**: **Strict separation, but no Discussion section** — §4 is pure Experiments/Results, §5 is a 2-paragraph Conclusion that paraphrases the abstract; broader interpretation absent. Limitations relegated to appendix.
