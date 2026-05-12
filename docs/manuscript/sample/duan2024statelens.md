# Do LLMs Build World Representations? Probing Through the Lens of State Abstraction

- **Authors**: Zichao Hu, Junyi Jessy Li, Junyi Du, Yi Wu, et al.
- **Venue**: Advances in Neural Information Processing Systems (NeurIPS) 2024
- **Year**: 2024
- **NeurIPS PDF**: https://proceedings.neurips.cc/paper_files/paper/2024/file/b1b16c4b875eb84d3585cb70d23970ca-Paper-Conference.pdf
- **NeurIPS poster**: https://neurips.cc/virtual/2024/poster/93786

## Abstract (paraphrased)
Designs a text-based planning task in which an LLM acts as an agent interacting with objects in containers, then probes whether the model encodes (a) general state abstractions for predicting future states or (b) goal-oriented abstractions for accomplishing tasks. Distinguishes between these via state-abstraction theory and shows that goal-relevant abstractions, not generic state, dominate the internal representation.

## Why it is relevant to our paper
Bucket F (probing methodology) + Bucket B (encoder-memory pipelines). The "general vs goal-relevant abstraction" framing is precisely what we observe in h_2: it is *not* a Cartesian map of the room, it is a navigation-relevant code. Citing this anchors our distinction between magnitude (any decodable info) and format/consumption (task-functional code) in 2024 NeurIPS literature.

## Suggested citation key
`hu2024statelens`
