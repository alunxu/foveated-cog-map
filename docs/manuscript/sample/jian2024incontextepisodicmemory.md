# Linking In-context Learning in Transformers to Human Episodic Memory

- **Authors**: Li Ji-An, Corey Y. Zhou, Marcus K. Benna, Marcelo G. Mattar
- **Venue**: Advances in Neural Information Processing Systems (NeurIPS) 2024
- **Year**: 2024
- **arXiv**: https://arxiv.org/abs/2405.14992
- **NeurIPS**: https://neurips.cc/virtual/2024/poster/96248

## Abstract (paraphrased)
This paper draws a mechanistic correspondence between the contextual-maintenance-and-retrieval (CMR) model of human episodic memory (a hippocampal model of free-recall) and induction heads in transformer LLMs. They show that CMR-like attention biases (asymmetric temporal contiguity) emerge in intermediate-to-late layers of pre-trained LLMs and that ablating these heads degrades in-context learning -- evidence of a causal contribution to ICL.

## Why it is relevant to our paper
Bucket D (cognitive maps / hippocampus theory). Bridges the recurrent-network probing tradition (which our paper extends to LSTM h_2) with the transformer-attention-as-hippocampus tradition (Whittington 2022). Useful as a citation showing that hippocampal-like memory mechanics are sensor-and-architecture dependent and emerge spontaneously, paralleling our finding that the *format* of the spatial code depends on the sensor.

## Suggested citation key
`jian2024incontextepisodicmemory`
