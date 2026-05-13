# Reviewer-Objection Defense Brief

Last updated: 2026-04-28 23:00
Pre-staging likely NeurIPS reviewer concerns + our planned responses.

---

## Tier-A: Most likely strong objections

### O1. "Single seed everything — could be lucky"

**Reviewer**: All H1 R² values, 5×5 transplant cells, dissociation points,
forgetting magnitudes are single-seed. How do I know they're not 1-in-5
flukes?

**Defense (existing + in flight)**:
- §5.5 (i) explicitly flags this as #1 limitation
- Multi-seed running NOW: uniform_seed=2 + foveated_seed=2 (Izar, ~30h
  remaining); blind_seed=2 + coarse_seed=2 + foveated_learned_seed=2
  (RCP probe-19/20/21, ~7h)
- For each Tier-1 claim, paper says "multi-seed in flight"
- **Risk**: if multi-seed lands and decay-rate ordering shifts, paper
  needs to soften §4.2 ¶5

**Rebuttal sketch**: "Multi-seed replication has now landed for all 5
conditions; the H1 ordering is robust at N=2 (table in revision)."

### O2. "Causation, not just correlation"

**Reviewer**: "Encoder--memory race" implies causation. But all evidence
is from 5 conditions, an N=5 correlational design. Where's the causal
test?

**Defense (in flight)**:
- **Encoder-resolution scaling sweep** (probe-13/16/17/18 on RCP):
  K∈{32, 64, 96, 128, 192} — varies encoder capacity continuously while
  holding everything else fixed. If R² declines monotonically → causal
  evidence for encoder being the axis.
- **Visual ablation EVAL** (Izar SLURM 2861347-51): zeros out RGB+depth
  mid-rollout. If rich-encoder integrated GPS code re-emerges in h_2
  post-ablation, substitution is causally supported.
- **GPS sensor ablation** (Izar SLURM 2861353-57): zeros GPS+compass,
  measures SPL drop per condition. Bottleneck dropping more than
  rich-encoder = causal evidence for §4.5 dissociation.
- **Log-polar foveation** (probe-14): falsifiable test of mechanism.
  Pre-registered prediction R² ≥ 0.30; if it lands < 0.30, mechanism
  story needs reframing.

**Rebuttal sketch**: "Three causal tests strengthen the correlational
5-condition story: scaling sweep (App E), visual-ablation (§5.3),
GPS-ablation (§5.3 reverse). All three lined up before submission."

### O3. "Architecture / task / sim scope"

**Reviewer**: One LSTM, one task (PointGoal), one sim (Habitat), one
encoder (ResNet-18). Why should I believe this generalizes to
transformers / ObjectGoal / iGibson / ViT?

**Defense (existing)**:
- §5.5 (vi) + (vii) explicitly scope to LSTM PointNav Habitat ResNet-18
- §5.4 frames the principle as "interface-level" — "any architecture
  that maintains a persistent state across timesteps and reads it via
  a linear or near-linear policy head"
- Bio convergence (§5.2) provides cross-system support
- We do NOT claim transformer-equivalence; explicitly an open prediction

**Risk**: reviewer pushes "you should test transformers". Standard
NeurIPS dance.

**Rebuttal sketch**: "We deliberately bound scope to one architecture
for direct Wijmans-comparability. Cross-architecture replication is
proposed as future work; the embodied-RL community is positioned for it."

### O4. "Decay-rate ordering is post-hoc"

**Reviewer**: §4.2 ¶5 claims "uniform fastest > foveated > foveated_learned"
in decay rate. With 3 single-seed runs and noisy curves, this looks like
finding patterns in noise.

**Defense**:
- §4.2 ¶5 already wraps the ordering in `\uncertain{}`
- §5.5 (i) says decay-rate is "most exposed to seed variability"
- foveated_learned_seed=2 (probe-21) is precisely to test this

**Risk**: ordering may flip at multi-seed, requiring text revision.

**Rebuttal sketch**: "We agree the fine-grained ordering is uncertain;
multi-seed result will resolve. The coarse-grained binary claim
(rich-encoder loses GPS faster than bottleneck) is robust."

### O5. "WJ-F v2 forgetting splits don't really split bottleneck"

**Reviewer**: blind +0.17 / coarse +0.43 / uniform +0.31 / foveated +0.34.
The 4 conditions don't cleanly split into "bottleneck robust" and
"rich fragile" — coarse (bottleneck) is the MOST fragile. Your "bottleneck
class" framing is wrong.

**Defense (just fixed today)**:
- §4.7 ¶2 reframed today: "separates blind from the integrated-code
  regime" instead of "splits the bottleneck class"
- New text: blind = sensor passthrough (robust); coarse + rich-encoder
  = some integration (fragile)
- This actually MATCHES the data more honestly

### O6. "1-NN purity 1.000 is sample-size driven"

**Reviewer**: 5 conditions in 512-d space with 7500 samples (1500 per
cond) trivially gives 100% 1-NN purity by curse of dimensionality.

**Defense (just verified today)**:
- 50K-sample 1-NN check (10K per cond, 6.7× original): purity STILL
  1.0000 exactly
- §4.3 ¶2 paper updated with this finding
- Independent confirmation via cross-condition probe transfer (R² ≪
  -800 off-diagonal — orthogonal evidence)

**Rebuttal sketch**: "1.0000 holds at 6.7× sample size, plus an
independent linearity test (probe transfer, R² ≪ −800) confirms."

---

## Tier-B: Likely-but-weaker objections

### O7. "Bug baseline SPL=0.07 is unfair (no learning)"

**Reviewer**: Comparing learned policies (SPL 0.59-0.85) to a hand-coded
controller (SPL 0.07) is a strawman.

**Defense**: We're not claiming our agents are "best ever"; the Bug
comparison contextualizes the absolute SPL range. §4.1 reframed today
as "supports that agents are doing real navigation rather than
coincidental wall-following" (softer than "confirming").

### O8. "Probe-readable vs policy-used dissociation is N=2 single-seed"

**Reviewer**: 5 conditions × 1 seed each, with TWO conditions showing
the dissociation, is anecdotal.

**Defense**:
- §4.5 explicitly labels as "candidate dissociations rather than
  established anomalies"
- Behavioural validation via shortcut SPL drop trajectory analysis
  (uniform locks-onto-old margin +1.83m, n=46)
- Multi-seed in flight will tighten

**Risk**: dissociation could collapse to noise at multi-seed.

### O9. "Foveation transferability gap (0.71) is two single-seed numbers"

**Reviewer**: "0.71-magnitude" is foveated +0.35 minus uniform −0.36.
Both noisy single-seed.

**Defense**: §4.4 ¶2 already pendnoted single-seed; multi-seed in flight.

### O10. "What about 'ESPACE' / Ramakrishnan benchmark?"

**Reviewer**: SPACE benchmark (Ramakrishnan 2025) tests spatial cognition
in frontier vision-language models. How do your findings relate?

**Defense**:
- §2 cites Ramakrishnan and frames as parallel scale-out direction
- §4.7 (ii) cites SPACE for metric occupancy non-linear-probe finding
- We're testing a DIFFERENT system (LSTM-RL agents at training time)
  vs SPACE's frontier-VLMs at inference

---

## Holes — claims we'd struggle to defend

### H1. **No direct mid-rollout visual ablation as paper-grade evidence**

§4.2 ¶5 footnote says "A direct causal test --- ablating the visual
stream mid-rollout in rich-encoder agents to see whether the top-layer
GPS code re-emerges --- is left for future work." But we just submitted
this exact experiment as Izar SLURM 2861347-51. **Once results land,
update the footnote and integrate into §4.2**.

### H2. **No cross-condition memory transplant at multi-seed**

5×5 transplant matrix is single-seed. Multi-seed in flight covers
training but transplant analysis would need re-run. Maybe out of v1
scope.

### H3. **§4.6 H3 dynamic gaze**: probe-24 (production stoch-gaze with
fix) is still pending GPU. If it lands and shows no dynamic-gaze
contribution beyond static, paper's H3 claim shrinks to "static only".

### H4. **Encoder spatial output as the trigger axis is one
interpretation**

§5.4 says the relevant axis is "encoder spatial-feature variety per
step" but with 5 conds, can't fully separate this from "input
resolution" or "channel info". Scaling sweep (in flight) addresses
input resolution. Channel info is harder — would need encoder ablation.

### H5. **No WJ-C decoder Stage 2** (allocentric occupancy decoder)

Paper §4.7 ¶2 still has hcpending placeholder. Stage 1 (scene_occ) in
flight on RCP (probe-11), but Stage 2 not yet started. If we don't
finish Stage 2 by submission, paper has visible hcpending.

---

## Recommended rebuttal preparation by paper section

| Paper section | Most likely objection | Defense in current paper | Defense once data lands |
|---|---|---|---|
| Abstract | "single-seed; transformer untested" | "candidate" + "convergent (not conclusive)" wording | Multi-seed verified |
| §4.1 | "Bug baseline is strawman" | Softer wording today | (no new data needed) |
| §4.2 | "correlational, not causal" | App E + log-polar in flight | Causal scaling sweep + ablation tests landing |
| §4.3 | "1-NN sample-size artefact" | 50K verified today | (done) |
| §4.4 | "0.71 transferability gap noisy" | pendnote single-seed | Multi-seed |
| §4.5 | "2x2 dissociation N=2 anecdotal" | "candidate" framing | Multi-seed + GPS ablation EVAL |
| §4.6 | "H3 only static, no dynamic" | static foveated-shifted as primary | Production stoch-gaze (probe-24) |
| §4.7 | "WJ-F splits arbitrary" | reframed today; v2 metric | (done) |
| §5.4 | "transformer untested" | scope statement | (out of scope; future work) |
| §5.5 | (limitations explicit) | comprehensive (i)-(vii) | — |
