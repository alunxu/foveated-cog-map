# Evaluating alignment between humans and neural network representations in image-based learning tasks

- **Authors**: Can Demircan, Tankred Saanum, Leonardo Pettini, Marcel Binz, Blazej M Baczkowski, Christian F Doeller, Mona M Garvert, Eric Schulz
- **Venue**: NeurIPS 2024 (poster)
- **Year**: 2024
- **arXiv**: https://arxiv.org/abs/2306.09377
- **NeurIPS PDF**: https://proceedings.neurips.cc/paper_files/paper/2024/file/dd37fdb24a4e1cfa3ed5c247217a7394-Paper-Conference.pdf

## Abstract (paraphrased)
Tests how well representations from 86 pretrained vision/multimodal NNs predict human learning trajectories on continuous-relation and category tasks built from THINGS-database natural images. Finds that training-data scale and contrastive multimodal pretraining (CLIP-style) are the dominant drivers of human-like generalisation, with multiple contributing factors (model size, class separation, embedding similarity to the task generative process).

## Structure analysis
- **§2 Related Work**: **No standalone Related Work.** Related work appears as a subsection inside §5 Discussion (5.1), ~2 paragraphs, comparing to prior alignment literature.
- **Intro**: ~4 paragraphs, **motivation-first** (representational alignment is a hot topic, but existing paradigms have a complexity gap). Ends with **3 explicitly numbered bulleted contributions**.
- **Where prior work appears**: Intro (context-setting), Methods (THINGS dataset citation), Discussion 5.1 (synthesis with prior alignment work). Cite-piles in brackets throughout.
- **Bio-AI bridge style**: **Discussion-heavy** — bio framing concentrated in intro and §5; methods are largely human-experiment-mechanics + ML evaluation. Cognitive-science vocabulary (generalisation, learning trajectories) used inline rather than via dedicated subsections.
- **Section ratios**: Intro ~15%; Methods (§2 Experiments) ~15%; Results (§3-§4) ~45%; Discussion §5 ~20% (incl. Limitations §5.2, ~1.5 pages of the 12-page main).
- **Methods presentation**: **Hybrid** — core methods §2; further model-fitting details inline in Discussion/Results when first needed; full specs in Appendix A.
- **Results vs Discussion**: **Soft separation** — results subsections include interpretation ("Why are CLIP models substantially better aligned?" is itself a results subsection header). Discussion provides higher-level synthesis rather than independent analysis.
