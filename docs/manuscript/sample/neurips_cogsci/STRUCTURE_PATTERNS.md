# Structure patterns across 10 NeurIPS 2024 ML+cognitive-neuroscience papers

Reverse-engineered from: Mounir-Sarkar 2024 (PAM), Iyer-Fiete 2024 (grid-cells/abstract domains), Pettersen 2024 (place cells/remapping), Lee-Kwon 2024 (EFT multi-agent), Redman 2024 (dual-agent path integration), Guan 2024 (distributed TD), Dong 2024 (Brain-JEPA), Demircan 2024 (human alignment), Piotrowski 2024 (constrained belief / NeurIPS NeuroAI workshop), Bonnasse-Pallier 2024 (LLM↔fMRI lateralisation), Chua 2024 (simple SF). All NeurIPS 2024 main or workshop.

Findings on **how they're written**, ignoring contributions.

---

## A. Standalone §2 Related Work — half do, half don't

**4 of 10 have a standalone §2 Related Work**: Lee-Kwon (EFT, ~7 paragraphs), Dong (Brain-JEPA, ~4 par), Piotrowski (~3 sub-themes), Chua (~5 par typology). Add a 5th if §2 "Background" with related-work content counts: Mounir-Sarkar's §2 Background (~10 paragraphs across 3 subsections) and Guan's §2 Background (~3 subsections) functionally play that role even though the heading isn't "Related Work."

**5 of 10 have NO standalone Related Work**: Iyer-Fiete (related work tucked as §1.1 *inside* the intro), Pettersen (folded into the intro and inline-cited everywhere in Results & discussion), Redman (everything inline; no Related Work heading), Demircan (related work appears only as §5.1 *inside* Discussion), Bonnasse (related work folded into intro and methods).

**Pattern.** When the paper is a method-paper (Lee-Kwon, Chua, Brain-JEPA, PAM) it gets a standalone §2 typology. When the paper is a probing/empirical-bio paper (Iyer, Pettersen, Redman, Bonnasse) the literature is dispersed inline — every result paragraph cites the brain experiment it bears on. **The bio-AI bridge papers in this set lean toward the dispersed pattern.**

When §2 is present it averages 3-7 paragraphs (~half a page), thematically organised by sub-area or by alternative-solution-type. Never the 1.5-page "everything-relevant" lump.

---

## B. Intro length and shape

Intros are **4-10 paragraphs**, median ~5. Two broad styles:

- **Motivation-first** (8 of 10): Mounir, Iyer, Pettersen, Lee-Kwon, Redman, Guan, Dong, Bonnasse, Chua. Open with the cognitive-science or biological phenomenon, narrow to a gap, then propose. Iyer's is the longest at ~9 paragraphs (panels Fig 1a-d as visual motivation across 4 abstract domains).
- **Claim-first** (2 of 10): Piotrowski opens with a question; Demircan opens with the alignment problem. Closer to Shai/Whittington style.

**Numbered contribution preview**: present in 7 of 10. Iyer (5 bullets), Mounir (3), Lee-Kwon (3), Demircan (3), Piotrowski (3), Guan (3), Redman (one labelled "Contributions." paragraph, prose form). Pettersen, Dong, Bonnasse, Chua skip it — their contributions live inside the closing "In this work, we…" paragraph as flowing prose.

The 4-paragraph intro with ~3-bullet contribution list at the end is the modal NeurIPS shape.

---

## C. How they bridge bio and AI in writing

Three styles, in order of frequency in this sample:

1. **Inline parentheticals throughout** (Iyer, Pettersen, Redman, Bonnasse, Brain-JEPA-in-methods). Every method choice is justified in the same sentence with a brain-experiment citation: "weaker grid responses ... a result consistent with recent human neurophysiological experiments [9, 10]." This is the dominant strategy for bio-empirical papers — the bridge *is* the prose, not a designated subsection.
2. **Dedicated bio subsection inside Background or Methods** (Mounir's §2.1 Predictive Coding + §3.2 SDR; Lee-Kwon's "Episodic Future Thinking" sub-subsection inside §2; Guan's §2.1 NAc reward learning; Piotrowski's §II.2 Optimal Prediction). The bio scaffolding is fenced off so methods can cite it by name.
3. **Discussion-only bridge** (Demircan, Chua to a degree). Methods are pure ML; bio framing returns in discussion.

**Notably absent in this sample**: a separate §2 Bio-AI Bridge subsection. Nobody does it as a standalone section; it's always either dispersed (style 1), tucked into Background/Methods (style 2), or held back for Discussion (style 3).

---

## D. Section ratios

Median across the 10:
- **Intro**: 10–15% (Iyer outlier ~25% with related work nested inside)
- **Related Work / Background**: 0–12% (4 papers have ~0; 4 have ~10%)
- **Methods**: 15–30%
- **Results**: 35–55% (always the largest)
- **Discussion**: 3–10%
- **Limitations**: 1 paragraph to 1 page; usually labelled paragraph; Guan unusually has a 12% standalone Limitations section, otherwise typical is one paragraph at the end of Discussion or before Conclusion.

**Pettersen is the structural outlier**: §2 "Results & discussion" merged, *then* §3 Limitations, *then* §3 Conclusion, *then* §4 Methods at the END. This is closer to a Nature/Cell paper template than a NeurIPS paper.

---

## E. Methods inline vs standalone

- **Standalone Methods/Preliminaries** (8 of 10): Mounir §3, Iyer §2-§3, Lee-Kwon §3-§4, Redman §2, Guan §3, Dong §3, Demircan §2, Chua §3-§4.
- **Methods at the END** (1): Pettersen — methods §4 follows results, conclusion, and limitations.
- **Methods inline at point-of-use** (1): Piotrowski — primary mechanistic procedures embedded inside §IV results subsections.

The "standalone methods, before experiments" remains the NeurIPS norm even in cogsci-adjacent work.

---

## F. Results vs Discussion

- **Strictly separate** (6 of 10): Mounir, Lee-Kwon, Redman, Guan, Bonnasse, Chua.
- **Soft-merged** (Discussion is 1 short paragraph + labelled Limitations) (2): Iyer, Demircan.
- **Fully merged** (1): Pettersen — single "Results & discussion" section.
- **Merged with interpretation in results subsections** (1): Brain-JEPA — §4.7 Interpretation lives inside Experiments.

---

## G. Three to four actionable suggestions for our paper

1. **It is genuinely safe to drop the standalone §2 Related Work.** Half of this NeurIPS-2024 cogsci-adjacent sample does not have one, and the absence-style is *more* common in bio-empirical/probing papers (Iyer, Pettersen, Redman, Bonnasse) — exactly our category. Pettersen and Redman both win NeurIPS slots while citing every brain-experiment inline at the result paragraph. Use that pattern: 3 named precedents (Wijmans, Sanders, Stachenfeld) front-loaded into the intro's "the gap" paragraph; the rest cited inline at first use in Results.

2. **Keep the intro motivation-first; finish it with a numbered or bulleted 3-5-item contribution list.** 8 of 10 in this sample do this; the most ML-method-y ones (Lee-Kwon, Iyer, Mounir, Demircan, Piotrowski, Guan, Redman) all use explicit numbered contributions. Our paper is method-y enough that this convention applies; do not be tempted by Pettersen-style flowing-prose contributions — only 2 papers in this sample use that, and they're the most "biology paper" of the set.

3. **Bridge bio to AI inline at every result paragraph, NOT in a dedicated subsection.** The dominant pattern (Iyer, Pettersen, Redman, Bonnasse) is: every result sentence ends with a parenthetical cite to the brain finding it explains or predicts. This is denser and more credible than reserving bio framing for Discussion. For our paper: every "h_2 carries an integration code" claim should end with a Sanders / Stachenfeld / Wijmans citation in the same sentence, in the same paragraph as the result.

4. **Keep Limitations short and labelled.** None of these 10 papers spends >1 page on Limitations in main text (Guan is the only ~1 page exception). The norm is: a single labelled "**Limitations.**" paragraph at the start of Discussion (Iyer, Pettersen, Redman, Lee-Kwon, Bonnasse, Chua). Or relegate to appendix entirely (Mounir). Do not write a §6 Limitations subsection with sub-paragraphs.

5. **(Bonus) Consider Pettersen's "Results & discussion" merged section if the paper is short.** Pettersen pulls this off in a 9-page main; the merged §2 contains 5 verdict-titled subsections and the prose is denser as a result. Worth considering if our discussion would otherwise repeat results-paragraph synthesis. Risk: reviewers expect separation, so flag the choice in the section header.
