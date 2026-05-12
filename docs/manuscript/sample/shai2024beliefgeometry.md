# Transformers Represent Belief State Geometry in their Residual Stream

- **Authors**: Adam S. Shai, Sarah E. Marzen, Lucas Teixeira, Alexander Gietelink Oldenziel, Paul M. Riechers
- **Venue**: Advances in Neural Information Processing Systems (NeurIPS) 2024
- **Year**: 2024
- **NeurIPS PDF**: https://proceedings.neurips.cc/paper_files/paper/2024/file/8936fa1691764912d9519e1b5673ea66-Paper-Conference.pdf
- **NeurIPS poster**: https://neurips.cc/virtual/2024/poster/94708

## Abstract (paraphrased)
Shows that transformers trained on data from a hidden Markov process linearly represent the *belief-state geometry* in their residual stream -- specifically, the meta-dynamics of belief updating over hidden states of the data-generating process. Implies that transformer hidden states encode not just the current observation but the full Bayesian posterior over latent state.

## Why it is relevant to our paper
Bucket A (emergent maps in deep RL navigation agents) + Bucket F (probing methodology). Strongest 2024 NeurIPS evidence that linear probes can recover *full belief geometry* in modern architectures. Our claim that h_2 carries an integration code is a navigation-specific instance of the same phenomenon. Cite as a strong methodological precedent for linear probing of hidden-state-encoded beliefs.

## Suggested citation key
`shai2024beliefgeometry`
