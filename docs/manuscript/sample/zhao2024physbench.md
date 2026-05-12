# PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding

- **Authors**: Wei Chow, Jiageng Mao, Boyi Li, Daniel Seita, Vitor Guizilini, Yue Wang, et al.
- **Venue**: International Conference on Learning Representations (ICLR) 2025
- **Year**: 2025
- **arXiv**: https://arxiv.org/abs/2501.16411
- **OpenReview**: https://openreview.net/forum?id=Q6a9W6kzv5
- **Project page**: https://physbench.github.io/

## Abstract (paraphrased)
PhysBench is a 10,002-entry interleaved video-image-text benchmark covering four domains: physical object properties, object relationships, scene understanding, and physics-based dynamics, broken into 19 subclasses and 8 capability dimensions. Across 75 representative VLMs, the authors show that frontier VLMs excel at common-sense reasoning yet fail at understanding the physical world -- attributed to the absence of physical knowledge in training data and the lack of embedded physical priors in the visual encoder.

## Why it is relevant to our paper
Bucket C (spatial reasoning failures in VLMs). Strengthens the encoder-bottleneck argument for VLMs alongside Ramakrishnan 2025 (SPACE) and Yang 2024 (VSI-Bench). PhysBench is broader (physics, not only spatial), recent (ICLR 2025), and explicitly diagnoses the visual-encoder side rather than the language side -- the same hypothesis we test mechanistically in our agent.

## Suggested citation key
`chow2025physbench`
