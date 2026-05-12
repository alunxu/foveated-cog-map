# Beyond Geometry: Comparing the Temporal Structure of Computation in Neural Circuits with Dynamical Similarity Analysis

- **Authors**: Mitchell Ostrow, Adam Eisen, Leo Kozachkov, Ila Fiete
- **Venue**: Advances in Neural Information Processing Systems (NeurIPS) 2023
- **Year**: 2023
- **NeurIPS PDF**: https://proceedings.neurips.cc/paper_files/paper/2023/file/6ac807c9b296964409b277369e55621a-Paper-Conference.pdf

## Abstract (paraphrased)
Argues that geometry-only similarity measures (CKA, RSA, Procrustes) miss when two networks compute the same function differently in time. Introduces Dynamical Similarity Analysis (DSA), a modified Procrustes analysis over vector fields that accounts for how vector fields transform under orthogonal change of basis. DSA can detect equivalent dynamical computation across systems whose geometries differ.

## Why it is relevant to our paper
Bucket F (probing methodology + representational geometry, recent). We use Procrustes shape distance and CKA -- both *geometric* measures that the user-provided reviewers may flag as missing the *dynamics*. DSA is the canonical 2023 NeurIPS rebuttal: include and discuss as a methodological caveat, or run as an additional analysis. Especially relevant given our cross-condition probe transfer, which is essentially a functional-equivalence test.

## Suggested citation key
`ostrow2023dsa`
