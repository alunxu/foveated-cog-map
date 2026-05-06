# Risks and Mitigations

## What went wrong with DSA (analogue we want to avoid)

The DSA experiment yielded four different "outlier" identities under four different
HP/protocol choices, making the qualitative claim non-replicable across reasonable
analyst choices. The pattern: under-determined HPs × small effect-to-noise ratio →
narrative-shopping. We need to pre-specify enough that the claim survives reviewer-
imposed HP perturbation.

## Top risks for this experiment

1. **Probe-architecture sensitivity** (the DSA-like risk). Pasukonis 2023 explicitly
   motivate 4-layer MLP because *linear is too weak* and *7-layer is no better*. If
   we report only a linear probe, an MLP probe might tell a different story (or
   vice-versa). **Mitigation:** pre-register the *pair* (linear, 4-layer MLP) and
   plot both for every condition. The story we want is "linear R² monotone in
   bandwidth, MLP R² flat" — *both must be true* for the format-shift framing.

2. **Encoder-family confound.** R3M, VC-1, and DINOv2 differ not only in bandwidth
   but in objective, data, and architecture family. A monotone trend across them
   could be confounded by training-data quality. **Mitigation:** primary axis is
   *resolution* within one encoder (DINOv2-Base at 224, 112, 56, 28). Cross-family
   axis is reported as supporting evidence only.

3. **LSTM under-training in 1.5h.** If the trained-on-top LSTM hasn't converged, the
   probe targets noise. **Mitigation:** plot probe R² vs LSTM training step; report
   only after R² stabilises. If 1.5h isn't enough, fall back to *static encoder
   probe only* (no LSTM) — this still tests the encoder-side bandwidth claim.

4. **Memory-Maze offline data is scripted-policy, not goal-directed.** The agent
   isn't trying to localize; the LSTM may not learn the same integration strategy
   our PointNav agents do. **Mitigation:** do NOT claim "this is a navigation
   agent with a localizer". Claim "this is a memory-conditioned predictor over
   memorymaze trajectories; the encoder-bandwidth → linear-readable-position
   trend is mechanism-only". This is honest; the falsifier is "did the trend
   reproduce, yes/no", not "does this agent navigate".

5. **Eval window choice.** The published protocol averages over steps 500-1000.
   Using all 1000 steps biases toward early-trajectory positions which any model
   handles. **Mitigation:** match published window exactly. Report all-step
   numbers as a sensitivity check, not the headline.

6. **Train/test split leakage.** Same maze layout in train and eval would let the
   probe memorise. **Mitigation:** Memory-Maze offline split is by *trajectory*
   not by *layout* — 29k train / 1k eval trajectories. Confirm via the official
   split file.

7. **R² scale vs RMSE scale.** Our paper reports R²; Pasukonis reports RMSE in
   grid-cell units. They are not directly comparable. **Mitigation:** report
   both; in the paper headline, use R² as in our main results, and add an RMSE
   column for cross-reference to the Memory-Maze literature.

8. **Pre-registration to prevent narrative shopping.** Before running, write down
   the predicted ordinal ordering of conditions and the predicted shape (linear
   monotone, MLP flat). If the data violates this, *report it* — do not p-hack the
   probe HPs to recover the predicted shape. This is the explicit DSA-mitigation
   policy.

## Pre-experiment commitments (write these into the EXPERIMENT_PLAN before running)

* Probe architectures: linear, 4-layer MLP (1024-1024-1024-1024 with ELU + LayerNorm),
  matching Pasukonis 2023.
* Eval window: steps 500-1000 (matches published protocol).
* Train/test: official Memory-Maze 9x9 split.
* Conditions: 4 input resolutions {64, 32, 16, 8} on DINOv2-Base, plus optional
  log-polar variant matching our Habitat condition.
* Predicted result shape: linear R² monotone-decreasing in resolution; MLP R² flat;
  the gap (MLP − linear) grows with bandwidth reduction.
* If shape is non-monotone or all-flat: report as null result with sentence
  "the encoder-bandwidth → linear-readable-position prediction does not reproduce
  in this setting; possible reasons: scripted-policy data, smaller maze, lack of
  goal-conditioning". Do not iterate on probe HPs to chase the predicted shape.
