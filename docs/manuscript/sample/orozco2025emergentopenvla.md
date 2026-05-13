# Emergent World Representations in OpenVLA

- **Authors**: Saurav Orozco, et al.
- **Venue**: arXiv 2509.24559 (workshop / preprint en route to NeurIPS 2025 / mech-interp track)
- **Year**: 2025
- **arXiv**: https://arxiv.org/abs/2509.24559
- **OpenReview**: https://openreview.net/forum?id=cydXirmduY

## Abstract (paraphrased)
Probes whether OpenVLA, a transformer-based vision-language-action model, contains an emergent internal world model. Using embedding arithmetic and linear/MLP probes on intermediate activations across layers, the authors recover statistically significant predictive ability for state transitions and frame the result via Koopman operator theory. Key findings: (i) the world model emerges in middle layers, (ii) diverse pre-training is essential, (iii) linear probes outperform MLPs (interpretability), (iv) prediction quality grows with horizon, (v) sparse-autoencoder analysis scales the probe.

## Why it is relevant to our paper
Bucket B (encoder-memory pipelines / capacity in modern embodied AI). The closest contemporary mirror of our methodology: probes a frozen VLA, finds a *linear-readable* world state, attributes it to the encoder/training mix. We argue the symmetric story for LSTM PointGoal: better visual encoders *reduce* the linear-readable code in the recurrent memory. The pair frames "capacity allocation" in the encoder-vs-memory dimension cleanly. Likely a missed citation for our related work.

## Suggested citation key
`orozco2025emergentopenvla`
