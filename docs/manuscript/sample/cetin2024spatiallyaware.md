# Spatially-Aware Transformer for Embodied Agents

- **Authors**: Junmo Cho, et al.
- **Venue**: arXiv 2402.15160 (ICLR / NeurIPS 2024 submission line)
- **Year**: 2024
- **arXiv**: https://arxiv.org/abs/2402.15160

## Abstract (paraphrased)
Extends transformer episodic memory for embodied agents to be place-centric: rather than indexing memory only by time, the model indexes by spatial bin so retrieval is "what happened *here*" rather than "what happened *then*". Improves performance on long-horizon embodied benchmarks with explicit place-anchored read/write.

## Why it is relevant to our paper
Bucket B (encoder-memory pipelines) + Bucket D (cognitive map theory). A direct architectural test of the same hypothesis we test mechanistically: that *where the agent integrates spatial state* matters for downstream behaviour. Cite as: "modern transformer agents that explicitly bake place into memory enjoy gains analogous to those an LSTM gets when its sensor already reduces the burden on integration."

## Suggested citation key
`cho2024spatiallyaware`
