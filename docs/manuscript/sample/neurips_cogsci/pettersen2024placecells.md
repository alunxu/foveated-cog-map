# Learning Place Cell Representations and Context-Dependent Remapping

- **Authors**: Markus Pettersen, Frederik Rogge, Mikkel Elle Lepperød
- **Venue**: NeurIPS 2024 (poster, main track)
- **Year**: 2024
- **OpenReview**: https://openreview.net/forum?id=7ESHFpqjNO
- **NeurIPS PDF**: https://papers.nips.cc/paper_files/paper/2024/file/0098a92f5f4e2d96c6db471e0c5507a8-Paper-Conference.pdf

## Abstract (paraphrased)
Proposes a minimal, similarity-based objective that translates proximity in physical space to proximity in network population vectors; feed-forward and recurrent networks trained on it learn place-like firing fields and, when conditioned on a scalar context input, exhibit hippocampal-like global remapping between contexts. The objective is invariant to orthogonal transformations, so rotating the trained representation generates new, distinct maps without retraining — a normative account of remapping.

## Structure analysis
- **§2 Related Work**: **No standalone section.** Prior work woven into the intro across two paragraphs: one citing experimental place-cell literature (O'Keefe, Leutgeb), one citing the recent normative-model precedents (Cueva-Wei, Banino, Sorscher, Whittington-TEM, Dorrell, Schaeffer 2023).
- **Intro**: ~4 paragraphs, **motivation-first** (cab drivers in London, place cells, remapping phenomenon), narrowing to "How place cells obtain their striking behaviors remains a matter of debate" gap statement, then one paragraph "In this work we propose..." that doubles as a numbered-style contribution preview but written as flowing prose with inline (i)/(ii)/(iii) markers (similarity-objective → joint encoding → orthogonal rotation = remapping).
- **Where prior work appears**: Intro (ground-laying), inline parentheticals throughout Results (each phenomenon paired with the experimental paper reporting it: Leutgeb 2004, Krupic 2012, Sorscher 2022, etc.), no separate Related Work section.
- **Bio-AI bridge style**: **Inline parentheticals**, dense and consistent — every model property comes with the brain finding it explains in the same sentence ("similar to Hippocampal global remapping [Leutgeb et al., 2004]"). Discussion is folded into Results.
- **Section ratios** (rough page split for the 9-page main): Intro <1 page (~10%); Results & discussion ~5 pages (~55%, merged); Limitations ~0.4 page; Conclusion ~0.3 page; Methods ~2 pages (~22%); refs.
- **Methods presentation**: **Methods are pushed to the END (§4)** — *after* Results and Conclusion. Pseudo-NeurIPS-bio-paper convention. Subsections 4.1–4.4 cover models/training/spatial-correlation/place-field analysis.
- **Results vs Discussion**: **Merged into one section "§2 Results & discussion"** with verdict-style subsection titles (e.g., "A similarity-based objective for learning representations of space and context"). Followed by a 1-paragraph Limitations and a 1-paragraph Conclusion.
