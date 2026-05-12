# Navigation World Models

- **Authors**: Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, Yann LeCun
- **Venue**: Computer Vision and Pattern Recognition (CVPR) 2025 (arXiv 2024)
- **Year**: 2024 / 2025
- **arXiv**: https://arxiv.org/abs/2412.03572
- **Project page**: https://www.amirbar.net/nwm/
- **Code**: https://github.com/facebookresearch/nwm

## Abstract (paraphrased)
Introduces NWM, a 1B-parameter Conditional Diffusion Transformer trained on a large mixture of egocentric human and robot videos. NWM predicts future visual observations conditioned on past observations and navigation actions, and at inference can plan trajectories by simulating them and evaluating goal achievement, or rank candidate trajectories sampled from an external policy.

## Why it is relevant to our paper
Bucket B (encoder-memory pipelines) + Bucket A (emergent maps in deep RL navigation agents). The newest LeCun-camp navigation system that *deliberately* puts the world model in the encoder side rather than the recurrent memory. Direct contrast to Wijmans-style PointGoal LSTM agents we study: where the spatial state lives -- in the rollout-generating encoder vs. the integration RNN -- is itself the variable. Strong related work for our discussion of "encoder-vs-memory capacity allocation" in modern systems.

## Suggested citation key
`bar2024navigationworldmodels`
