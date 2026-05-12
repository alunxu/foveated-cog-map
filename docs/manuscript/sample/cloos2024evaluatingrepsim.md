# Evaluating Representational Similarity Measures from the Lens of Functional Correspondence

- **Authors**: Y. Cloos, et al. (corresponding to arXiv 2411.14633)
- **Venue**: arXiv 2024 (under review for NeurIPS / TMLR)
- **Year**: 2024
- **arXiv**: https://arxiv.org/abs/2411.14633

## Abstract (paraphrased)
Empirically evaluates 8 commonly used representational similarity metrics (CKA-linear, CKA-rbf, RSA, Procrustes, soft matching distance, etc.) on the criterion of *functional correspondence*: do two networks judged similar by the metric also behave similarly on downstream tasks? Linear CKA and Procrustes distance score best at differentiating trained from untrained models and at aligning with behavioural measures.

## Why it is relevant to our paper
Bucket F (probing methodology). We use linear CKA and Procrustes; this paper is the strongest 2024 empirical justification of those choices. Useful as a defensive citation when reviewers ask "why these metrics and not RSA / soft matching?".

## Suggested citation key
`cloos2024evaluatingrepsim`
