# V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

- **Authors**: Mahmoud Assran, Adrien Bardes, David Fan, et al. (Meta FAIR)
- **Venue**: arXiv 2506.09985 (Meta FAIR research; sibling to V-JEPA which is already cited as `bardes2024vjepa`)
- **Year**: 2025
- **arXiv**: https://arxiv.org/abs/2506.09985
- **Meta**: https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/
- **Code**: https://github.com/facebookresearch/vjepa2

## Abstract (paraphrased)
V-JEPA 2 trains a joint-embedding predictive video architecture on >1M hours of video, then post-trains a latent action-conditioned variant (V-JEPA 2-AC) on <62 hours of unlabelled robot video to enable zero-shot planning on Franka arms. Achieves SotA on motion understanding, action anticipation, and video QA at the 8B-parameter scale.

## Why it is relevant to our paper
Bucket B (encoder-memory pipelines, recent). Strongest 2025 example of a JEPA-family world model that (i) does not reconstruct pixels, (ii) deliberately abstracts away predictable detail. The architectural choice mirrors our intuition that the encoder shapes the format of downstream memory. Useful upgrade citation alongside the existing V-JEPA 1 entry.

## Suggested citation key
`assran2025vjepa2`
