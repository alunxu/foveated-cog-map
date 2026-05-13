# Writing Patterns from 7 High-Leverage Precedent Papers

Reverse-engineered from: Shai 2024 (NeurIPS), Orozco 2025 (arXiv/NeurIPS-track), Schaeffer 2022 (NeurIPS), Ostrow 2023 (NeurIPS), Whittington 2022 (ICLR), Dorrell 2024 (Neuron), Bar 2024 (CVPR 2025). Findings ignored; only "how it's written" extracted.

---

## A. Structure-level patterns

**Intros.** All seven open with a single declarative claim sentence, *not* a funnel. Shai: "In this work, we present a rigorous and concrete theoretical framework that connects the structure of training data to the geometry of activations." Whittington: "In this work we 1) show that transformers (with a little twist) recapitulate spatial representations found in the brain; 2) [...]" — claim + numbered preview, all on page 1. Schaeffer is the outlier: it uses a soft funnel ("DL has underpinned nearly every success...") but pivots within one paragraph to "we examine the essential claims" and lands a numbered 6-point contribution list before §2. Bar uses a 3-paragraph claim-first: organism-level motivation, gap in current SOTA, our model. **None waste a column on a "the brain is fascinating" lede.** Average intro length: 1–1.5 pages, ending in a numbered or bulleted contribution preview.

**Related work.** Three patterns coexist:
- *No standalone §2* (Shai, Schaeffer, Whittington, Dorrell): prior work is folded into the intro and into per-result mini-contextualisations. Shai's §4 "Discussion" doubles as a related-work synthesis.
- *Short §2 Background* (Bar, Orozco): one-paragraph-per-thread, ~half a page, segmented by sub-area (Bar: "Goal-conditioned navigation," "Differently than learning a policy," "In computer vision").
- *Standalone §1.x background* (Ostrow): tucked at end of intro, not its own section.

**Methods.** All except Bar interleave: definitions where they bite. Shai introduces "mixed-state presentation" only at §2.2, immediately before its first use. Ostrow places Procrustes vs. DSA equations adjacent to the Figure-1 schematic — text and figure are mutually load-bearing. Whittington compresses TEM equations 3–7 into one column, then immediately re-derives the transformer parallel — equations earn their column inches.

**Results.** Universally **one claim per subsection**, subsection title is the verdict ("Belief state geometry is linearly represented in the residual stream"; "DSA captures dynamical similarity despite differences in geometry"; "Networks trained on path integration tasks learn to estimate position, but rarely learn grid cells"). Subsection 3.X+1 is consistently a *sharper* version of 3.X — controls, alternative data-generating processes, scaling — never a parallel new direction. Schaeffer pushes this hardest: §4→§5→§6 are "first claim," "first claim under stricter conditions," "first claim under realistic conditions." Each plot-bearing subsection closes with one verdict sentence.

**Section ratios.** Median across the 7: results 50–55%, methods 15–20%, intro 10%, discussion+limitations 15%, conclusion ≤5%. NeurIPS papers (Shai, Schaeffer, Ostrow) keep limitations under 1 page, often as a single labelled paragraph (Shai §4.4). Conclusion is consistently 1–2 short paragraphs that paraphrase the abstract in past tense plus one forward-looking sentence — never new content.

---

## B. Sentence- and paragraph-level tricks

**Inevitability framing.** Shai's signature move: state the theory's prediction *before* the experiment, then report the match. "Our framework predicts highly nontrivial fractal structure; our empirical results confirm these predictions." Same in Whittington: "These modifications are sufficient to learn spatial representations" — a forecast verb (*sufficient*, *predicts*, *should*) that converts the experiment from "we ran X, got Y" into "Y was foretold." Orozco does it with falsifiability: "Our **falsifiable hypothesis** is that f(a_t) exists ∀ LIBERO sections" (bolded in original). The sentence template is: *if [theory] holds, we expect [observable]; we observe [observable]*. This makes findings feel deductive even when the experiment is correlational.

**Hedges.** Two clean styles:
- *Footnoted caveat* (Shai, Schaeffer): the caveat lives in a numbered footnote attached to the strong claim sentence, so prose reads bold but rigour is preserved. Shai: main text says "transformers represent the belief state geometry"; footnote 4 adds "These results hold when analyzing the activations after the final layer norm as well, see Figure S1."
- *End-of-section single sentence* (Bar, Whittington, Dorrell): the strong claims march; one sentence closes the section with "However, [scope]."
None of the 7 use *inline* hedging ("we cautiously and tentatively suggest that..."). When Schaeffer wants to caveat a correlation, he says it once at paragraph end ("We caution this correlation between dimensionality and neural predictivity is not (yet) strong evidence") and never returns to it.

**Unfamiliar terminology.** Standard pattern: *italic introduction + 1-line gloss in the same sentence*. Shai: "this geometry is the *belief state geometry*" (italics + appositive). Ostrow: "the existence of a homeomorphism that maps flows of one system onto flows of the other" — definition embedded as relative clause, no separate "Definition 1" box. Heavy formalism (Ostrow §2, Bar §3.1) is fenced inside one subsection so the rest of the paper can cite it by name.

**Dense citation.** Universal: cite-piles inside square brackets at end of clauses (`[15, 3, 59, 47]`), never named in prose, *unless* the citation is a load-bearing precedent. Then the author is named ("Following Nanda et al. (2023) we investigate..."). Shai uses named attribution ~3 times in the whole paper, all for definitional precedents. The rule: pile-cite for breadth, named-cite for the paper your method directly extends.

**Methodological alternatives.** Three paragraphs, three voices: (i) acknowledge in *background* ("Geometric similarity metrics fall short in two manners..." — Ostrow), (ii) run the alternative as a baseline (Orozco's MLP-vs-linear, Schaeffer's 11,000-network sweep), (iii) defer to limitations only when the alternative isn't actionable. None apologise; the framing is always "X is great for Y, but Z requires our extension."

---

## C. Coherence tricks

**Threading results.** The strongest pattern: each subsection ends with a forward-pointing sentence that loads the next. Shai 3.1 closes "These results provide compelling evidence for our central claim... suggesting that this geometry is a fundamental aspect..." then 3.2 opens "Often, *distinct* belief states will have the *same* next-token prediction. Our framework suggests transformers will keep internal distinctions..." — the *suggestion* in 3.1 is the *prediction* tested in 3.2. Schaeffer literally numbers his bold mini-headings inside §4–§6 ("Grid period values set by hyperparameters, and multiple modules do not emerge.") so each result has a 1-sentence verdict that can be read alone — a paper-within-a-paper structure.

**Figure↔text integration.** All 7 reference figures by panel letter at the *exact* claim sentence ("Figure 5C", "Figure 7D Left"), never "see Figure X" alone. Whittington Figure 1 is captioned with the claim ("Sequence prediction in spatial navigation tasks test abstract spatial understanding...") rather than describing axes. Shai Figure 1 caption itself states the result ("Our main experimental result is that we find the fractal geometry of optimal beliefs is linearly embedded in the residual stream"). **Figures carry one bolded takeaway per caption.**

**Master metaphor.** Each paper has one and only one. Shai: "synchronizing to the hidden state." Ostrow: "topological conjugacy." Whittington: "a little twist." Dorrell: "two algorithms." Bar: "imagined trajectories." The metaphor appears in title, abstract, intro, §3 (results), discussion, conclusion — usually ≥6 occurrences. It glues otherwise-disparate experiments into one story.

---

## D. Compactness tricks

**Inline lists vs. itemize.** Itemize = contributions and >3-item enumerations (Schaeffer's 6-point findings, Orozco's 3 contributions). Inline numbered "(i)... (ii)... (iii)..." for ≤3 items inside a flowing sentence ("we 1) show that... 2) show... 3) offer a novel take..."). Itemize wastes vertical space; nobody uses it for 2-item lists.

**Anti-redundancy.** A consistent division of labour:
- *Abstract*: claim + headline empirical signature ("we find belief states are linearly represented in the residual stream, even in cases where the predicted belief state geometry has highly nontrivial fractal structure").
- *Intro*: motivation + the framework's key noun + numbered contribution preview.
- *Results*: evidence; no bio/theory speculation.
- *Discussion*: synthesis-with-bio + future work; no new evidence.
- *Conclusion*: 1–2 paragraphs paraphrasing abstract.
The same finding is *never* re-stated identically across these — each section reframes it from a different angle (signature → claim → evidence → mechanism → summary).

**Numerical evidence.** Standard: *headline number in abstract, range in body, full table in appendix*. Orozco abstract just says "statistically significant predictive ability"; body table 1 shows 123/123 successful permutation tests; appendix B holds the full R² grid. Shai gives no headline number — the figure *is* the number (a fractal). Schaeffer's "11,000 networks, fewer than 10%" is the abstract number; details deferred.

**Tables vs. prose.** Tables when ≥3 conditions × ≥2 metrics (Bar Tables 1, 2, 3; Orozco Table 1). Prose for 2-condition contrasts (`R²=0.95 vs R²=0.31`, Shai 3.2 — done in one sentence). The threshold seems to be: if you'd write more than two numbers in one sentence, table.

---

## E. 5 actionable changes for our paper

1. **Adopt Shai's inevitability framing in §sec:format-h2 (the magnitude/format/consumption results).** Currently we likely say "we observe linear R² of position-decoding in h_2 across conditions." Replace with: "*A capacity-allocation account predicts that as the encoder's metric structure improves, the linearly-readable position code in h_2 should weaken. Across the five sensor conditions, this is what we observe (Fig X, panel C: R² in h_2 monotonically decreases as encoder distance-fidelity increases).*" The forecast verb does the rhetorical work.

2. **Compress related-work to Shai/Schaeffer style (no §2).** Our current draft probably has a 1.5-page §2 Related Work. Cut it. Move 3 named precedents (Wijmans, Banino, Schaeffer 2022) into the intro paragraph that frames the gap; pile-cite the rest at first use in results. Saves ~0.8 pages — major NeurIPS-limit relief.

3. **One-claim-per-subsection results, with verdict-titles.** Rename subsections from neutral ("§sec:h1 — Position decoding from h_1") to verdict-style ("§sec:h1 — Position is decodable from h_1 only when the encoder has metric geometry"). Force each results subsection (sec:h1, sec:h2, sec:transplant, sec:probe-transfer) to end with **one** forward-pointing sentence that loads the next, Shai-style. If a subsection has two findings, split it.

4. **Adopt Schaeffer's "stricter-and-stricter" results arc for §sec:causal/sec:transplant.** Currently the causal evidence (memory transplant + cross-condition probe transfer) is probably presented as parallel experiments. Re-stage as: behavioural correlation → linear-probe correlation → cross-condition probe transfer (functional equivalence test) → memory transplant (causal). Each tightens the claim. This is also the natural pre-emption of the Ostrow/DSA reviewer concern: by the time the reader hits the transplant, geometry-only objections have already been ruled out by probe-transfer.

5. **Pick one master metaphor and use it ≥6 times.** Strongest candidate: *"capacity allocation between visual encoder and recurrent integration"* — already in your framing. Audit the abstract, intro last paragraph, each results section opener, discussion opener, conclusion. If "capacity allocation" doesn't appear in all six, insert it. Whittington and Dorrell get away with ICLR/Neuron-level claims partly because every reader leaves with one phrase ringing.

**Bonus bookkeeping (not a numbered point):** kill all inline hedges ("we tentatively suggest", "it may be the case that"). Convert each to either (a) a footnote on the strong sentence, or (b) one closing sentence at section end. None of the 7 papers hedge inline; the prose authority gap is real.
