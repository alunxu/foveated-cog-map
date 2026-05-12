# No Free Lunch from Deep Learning in Neuroscience: A Case Study through Models of the Entorhinal-Hippocampal Circuit

- **Authors**: Rylan Schaeffer, Mikail Khona, Ila Rani Fiete
- **Venue**: Advances in Neural Information Processing Systems (NeurIPS) 2022
- **Year**: 2022
- **OpenReview / proceedings**: https://proceedings.neurips.cc/paper_files/paper/2022/hash/66808849a9f5d8e2d00dbdc844de6333-Abstract-Conference.html
- **PDF**: https://proceedings.neurips.cc/paper_files/paper/2022/file/66808849a9f5d8e2d00dbdc844de6333-Paper-Conference.pdf

## Abstract (paraphrased)
The authors empirically test the claim that grid cells emerge generically from networks trained to path-integrate. Across more than 11,000 networks they find that grid-like units emerge in fewer than 10% of architectures, and only when biologically invalid supervised targets (Difference-of-Softmaxes place cells) are used. They argue that grid-cell emergence in deep path integrators is the product of post-hoc hyperparameter selection, not a robust consequence of the path-integration objective.

## Why it is relevant to our paper
Bucket A (emergent maps in deep RL navigation agents) + Bucket D (cognitive maps theory). Our paper argues that the *format* of the cognitive map depends on sensor structure. Schaeffer et al. is a direct caution against over-interpreting "grid-like" or "place-like" emergent codes as universal; supports our framing that linear-readable spatial codes can be artefacts of architecture/sensor pairing rather than necessary representations. Stronger version of the `schaeffer2023fiction` paper we already cite.

## Suggested citation key
`schaeffer2022nofreelunch`
