# Not so griddy: Internal representations of RNNs path integrating more than one agent

- **Authors**: William T Redman, Francisco Acosta, Santiago Acosta-Mendoza, Nina Miolane
- **Venue**: NeurIPS 2024 (NeuroAI workshop track) — also at NeurReps; bioRxiv 2024
- **Year**: 2024
- **OpenReview**: https://openreview.net/forum?id=HX0e4xDsg9
- **NeurIPS PDF**: https://papers.nips.cc/paper_files/paper/2024/file/285b06e0dd856f20591b0a5beb954151-Paper-Conference.pdf
- **bioRxiv**: https://www.biorxiv.org/content/10.1101/2024.05.29.596500v1

## Abstract (paraphrased)
Trains the canonical Sorscher-Cueva path-integration RNN on dual-agent path integration and probes the resulting representations. Dual-agent training produces weaker grid responses, stronger border responses, and tuning to relative inter-agent position; population analyses (TDA, dynamics) find no continuous-toroidal-attractor signature, contrasting with single-agent RNNs. Predicts and matches recent human neurophysiology suggesting MEC grid strength inversely correlates with multi-agent task performance.

## Structure analysis
- **§2 Related Work**: **No standalone Related Work.** §2 is "RNN model" — a methods section. All prior-work scaffolding done inline in intro.
- **Intro**: ~7 paragraphs, **motivation-first** with brain-first framing (path integration → MEC → existing experimental gap → multi-agent setting). Closes with a one-paragraph "**Contributions.**" labelled mini-paragraph (not numbered/bulleted but explicit).
- **Where prior work appears**: Intro (extensive — every claim about MEC/CAN/RNN models cited); inline in Methods (§2 cites Sorscher 2019 explicitly when defining ground-truth place cells); inline in Results (each finding paired with the experimental paper it bears on, e.g., [9, 10] for human MEC negative correlation).
- **Bio-AI bridge style**: **Dense inline parentheticals throughout** — the paper *is* the bridge: every results paragraph cites the corresponding rodent or human neurophysiology paper (Park et al., human fMRI studies). Discussion explicitly maps each model finding to a testable experimental prediction in a labelled "Predictions." paragraph.
- **Section ratios** (9-page workshop format): Intro ~3 pages (~30%, unusually long); §2 RNN model (methods) ~1.5 pages; §3 Results ~3 pages with 4 verdict-titled subsections (3.1 not optimal for dual; 3.2 weaker grid stronger border; 3.3 relative-space tuning; 3.4 no toroidal attractor); §4 Discussion ~2 pages with labelled mini-paragraphs (Discussion, **Limitations**, **Predictions**, **Future directions**, **Outlook**).
- **Methods presentation**: §2 is a standalone Methods, then results in §3.
- **Results vs Discussion**: **Strictly separate** §3 (Results) and §4 (Discussion). Discussion uses bolded inline labels — Limitations, Predictions, Future directions, Outlook — instead of subsections.
