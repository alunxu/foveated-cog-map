# Mastering Memory Tasks with World Models

- **Authors**: Mohammad Reza Samsami, Artem Zholus, Janarthanan Rajendran, Sarath Chandar
- **Venue**: International Conference on Learning Representations (ICLR) 2024 (oral, top 1.2%)
- **Year**: 2024
- **arXiv**: https://arxiv.org/abs/2403.04253
- **OpenReview**: https://openreview.net/forum?id=1vDArHJ68h
- **Project page**: https://recall2imagine.github.io/

## Abstract (paraphrased)
Introduces Recall to Imagine (R2I), an extension of DreamerV3 in which the recurrent state-space model is replaced by a parallelisable Structured State-Space Model (S3M). The architecture explicitly trades off long-term memory and long-horizon credit assignment. Achieves SotA on memory-heavy benchmarks (BSuite, POPGym), superhuman on Memory Maze, comparable on Atari/DMC, and faster wall-clock than DreamerV3.

## Why it is relevant to our paper
Bucket B (encoder-memory pipelines, recent). Strong ICLR 2024 example of *deliberately* engineering memory architecture for navigation/credit-assignment tasks. Direct counterpart to our LSTM-based PointGoal: same DreamerV3 lineage, but the memory side is given more capacity. Useful for our discussion of how the encoder/memory split is currently the central architectural lever.

## Suggested citation key
`samsami2024recall2imagine`
