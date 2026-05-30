# Empirical Test: Sensor Constraint vs Encoder Scale

## Question

Can a task-aligned perceptual constraint produce a more useful spatial-memory
format than simply using a larger visual encoder?

The intuition is a budget tradeoff: a richer eye can hand the memory a very
detailed scene, but that detail may encourage scene-local shortcuts. A constrained
eye can force memory to integrate movement and maintain a cleaner position code.

## Most Convincing Experiment

Run a crossed Habitat PointGoal experiment:

| factor | levels |
|---|---|
| encoder scale | ResNet-18, ResNet-50 |
| sensor structure | uniform, foveated or log-polar |

All four agents should keep the same non-visual stack, recurrent memory size,
task, data, optimizer, and training budget. The decisive comparison is:

`ResNet-18 + constrained sensor` vs. `ResNet-50 + uniform sensor`

This directly tests whether the right sensor geometry can match or beat a larger
encoder under the same navigation objective.

Primary outcomes:

- Behavior: SPL / success on MP3D-test and memory-carryover interventions.
- Memory format: linear vs MLP probe of GPS position on `h2`.
- Causality: shortcut / excursion / transplant robustness.

Success criterion:

- The constrained-small agent matches or beats the uniform-large agent on at
  least one causal behavior metric, while also showing a cleaner linearly readable
  position code.

## Fast RCP Pilot

Because the full Habitat run is expensive, the pilot uses the existing
Memory-Maze world-model probe:

| factor | levels |
|---|---|
| frozen encoder scale | DINOv2-S, DINOv2-B |
| sensor structure | foveated, uniform |

Each cell feeds frozen encoder features into the same small LSTM, trained to
predict next-frame features from current features plus action. We then probe
`agent_pos` from the recurrent state.

Decisive pilot comparison:

`DINOv2-S + foveated` vs. `DINOv2-B + uniform`

Primary pilot metric:

- linear R2 from recurrent state to `agent_pos`

Secondary sanity metric:

- MLP R2 should remain comparable, showing that the constrained sensor is not
  just deleting spatial information.

Interpretation:

- Positive pilot: worth spending compute on the full Habitat 2x2.
- Negative pilot: still informative, because the stronger claim likely requires
  the full embodied RL setting where policy pressure and GPS/action integration
  are present.
