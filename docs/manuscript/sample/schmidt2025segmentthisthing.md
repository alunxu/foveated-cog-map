# Segment This Thing: Foveated Tokenization for Efficient Point-Prompted Segmentation

- **Authors**: Tanner Schmidt, et al. (FAIR)
- **Venue**: Computer Vision and Pattern Recognition (CVPR) 2025
- **Year**: 2025
- **arXiv**: https://arxiv.org/abs/2506.11131
- **CVF**: https://openaccess.thecvf.com/content/CVPR2025/papers/Schmidt_Segment_This_Thing_Foveated_Tokenization_for_Efficient_Point-Prompted_Segmentation_CVPR_2025_paper.pdf
- **Project page**: https://facebookresearch.github.io/segment_this_thing/
- **Code**: https://github.com/facebookresearch/segment_this_thing

## Abstract (paraphrased)
A point-prompted segmentation model that achieves SAM-competitive accuracy with ~24x fewer pixels by using a biologically-inspired foveated tokeniser: high-resolution patches near the prompt and progressively downsampled patches further away. The paper makes the case that foveated input is a practical, modern alternative to "shrink the model" for efficiency, and that it is interactive-frame-rate-friendly on consumer hardware.

## Why it is relevant to our paper
Bucket E (foveation / bandwidth-constrained vision in deep learning), 2025 entry. Our foveated 4x4 sensor is the same family. SegmentThisThing is the strongest recent CV publication to argue foveated tokenisation is not just biologically-plausible but practically useful -- a clean modern citation alongside Deza & Konkle 2021 and PerViT 2022.

## Suggested citation key
`schmidt2025segmentthisthing`
