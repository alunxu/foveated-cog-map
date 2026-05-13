# fMRI predictors based on language models of increasing complexity recover brain left lateralization

- **Authors**: Laurent Bonnasse-Gahot, Christophe Pallier
- **Venue**: NeurIPS 2024 (poster)
- **Year**: 2024
- **OpenReview**: https://openreview.net/forum?id=XF1jpo5k6l
- **NeurIPS PDF**: https://proceedings.neurips.cc/paper_files/paper/2024/hash/e28e19d00b23fe0265f433fa05a96b06-Abstract-Conference.html
- **GitHub**: https://github.com/l-bg/llms_brain_lateralization

## Abstract (paraphrased)
Tests 28 pretrained language models (124M–14.2B params, 8 families) as encoding models for fMRI activity during natural-language listening. Finds a clean scaling law: brain-prediction accuracy rises linearly with log(parameter count), and the *left-right* asymmetry in this fit also scales with model size — providing a computational handle on language-network lateralisation.

## Structure analysis
- **§2 Related Work**: **No standalone section.** Related work folded into the intro and into §2 "Materials and Methods" sub-section that lists the language-model families benchmarked.
- **Intro**: ~4 short paragraphs, **motivation-first** (language model encoding-of-fMRI is a thriving paradigm; left-lateralisation is a long-standing neurolinguistics puzzle; here we test scaling). Concise narrative-style contributions, no numbered bullets.
- **Where prior work appears**: Intro (embedded scaffolding); inline parentheticals in methods (specific encoding-model precedents like Caucheteux, Schrimpf, Hosseini); occasional in discussion when comparing scaling-law slope across studies.
- **Bio-AI bridge style**: **Heavy inline parentheticals** — every brain-area claim cites a neurolinguistics paper. The paper is fundamentally a bridge paper (LLM ↔ brain) so bio framing is woven throughout rather than ghettoised.
- **Section ratios** (10-page NeurIPS): Intro ~12%; Methods ~25%; Results ~40%; Discussion+Conclusion ~15%; Limitations ~5%; refs.
- **Methods presentation**: **Standalone Methods section** with subsections (fMRI dataset, language models, encoding model fitting, statistical test).
- **Results vs Discussion**: **Strictly separate** — Results subsections each report one scaling-law analysis (within-region, then asymmetry index, then layer effect); Discussion synthesises against the lateralisation literature.
