# Flexible mapping of abstract domains by grid cells via self-supervised extraction and projection of generalized velocity signals

- **Authors**: Abhiram Iyer, Sarthak Chandra, Sugandha Sharma, Ila R Fiete
- **Venue**: NeurIPS 2024 (poster)
- **Year**: 2024
- **OpenReview**: https://openreview.net/forum?id=hocAc3Qit7
- **NeurIPS PDF**: https://proceedings.neurips.cc/paper_files/paper/2024/file/9b333cc1c9eb36e479b27f8c19f0873c-Paper-Conference.pdf
- **GitHub**: https://github.com/abhi-iyer/velocity_extraction

## Abstract (paraphrased)
Proposes that grid cells in entorhinal cortex generalise spatial mapping to abstract cognitive spaces by extracting low-dimensional, state-independent velocity signals via self-supervised learning. The model uses a geometric "loop-closure" loss (closed-loop displacements sum to zero) to recover faithful velocity representations across abstract domains (auditory pitch, stretchy-blob shapes, etc.), beating standard dimensionality-reduction baselines and yielding a testable prediction: cell-cell relationships are preserved across domains.

## Structure analysis
- **§1.1 Related Work** (sub-numbered, *not* a standalone §2): ~3 paragraphs nested at the end of the introduction, organised as "neuroscience models of spatial mapping" then "dimensionality reduction work."
- **Intro**: ~9 paragraphs (long for NeurIPS), **motivation-first** (grid-cell generalisation phenomena across abstract domains, with Fig 1 panels a-d showing tone/space/bird/odor evidence), then states the gap, then proposes the velocity-extraction-first hypothesis. Ends with **5 explicit bulleted contributions** including a falsifiable experimental prediction.
- **Where prior work appears**: §1.1 (most), inline parentheticals in §2 setup paragraphs ("[16] continuous attractor models"), and in Discussion (testable predictions referencing existing experimental literature [17–23]).
- **Bio-AI bridge style**: Heavy *inline parentheticals throughout* — every method choice is justified in one clause referencing brain experiments (e.g., "this prediction aligns with the observed invariance of internal neural representations across different spatial environments and brain states [17–23]"). Discussion explicitly proposes neuroscience hypotheses.
- **Section ratios**: Intro+Related ~25%; Methods (§2 Self-supervision for velocity extraction, §3 Architecture) ~30%; Results ~30%; Discussion+Limitations+Future Work+Broader Impact ~10%; references/appendix ~rest. No standalone "Conclusion."
- **Methods presentation**: **Standalone §2 + §3** (mixed: §2 derives loss terms, §3 architecture). Each subsection introduces concepts only when needed (point-of-use within methods, but methods themselves are front-loaded).
- **Results vs Discussion**: **Separate** — §3 (architecture) flows into experiments by domain; Discussion (§4) is one paragraph synthesis + Future Work + Limitations + Broader Impact (each a labelled paragraph, not a subsection).
