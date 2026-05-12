# Learning Successor Features the Simple Way

- **Authors**: Raymond Chua, Arna Ghosh, Christos Kaplanis, Blake A Richards, Doina Precup
- **Venue**: NeurIPS 2024 (poster)
- **Year**: 2024
- **OpenReview**: https://openreview.net/forum?id=rI7oZj1WMc
- **NeurIPS PDF**: https://proceedings.neurips.cc/paper_files/paper/2024/hash/597254dc45be8c166d3ccf0ba2d56325-Abstract-Conference.html
- **GitHub**: https://github.com/raymondchua/simple_successor_features

## Abstract (paraphrased)
Successor features (SFs), the function-approximation generalisation of Dayan's hippocampal-inspired successor representation, are a strong candidate for continual deep RL but suffer from representation collapse when trained naively. Proposes a minimal scheme — a TD loss on the value plus a reward-prediction loss enforcing linearly-readable rewards — that captures the SF mathematical definition directly, learns from pixels without pretraining, and outperforms reconstruction/orthogonality/pretraining baselines in 2D and 3D mazes.

## Structure analysis
- **§2 Related Work**: **Yes, standalone**, ~5 paragraphs structured as a typology of prior solutions (reconstruction-based, hand-crafted features, pretraining-based, orthogonality-based) plus a "most-closely related" paragraph that names Ma 2020 and lists 4 numbered differences.
- **Intro**: ~6 paragraphs, **motivation-first** (continual deep RL → SF as Dayan-inspired solution → representation-collapse problem → existing fixes have drawbacks → here is our simpler way). No explicit numbered contribution list — narrative "Here we introduce..." paragraph carries the contribution.
- **Where prior work appears**: Intro (extensive scaffolding); standalone §2 (typology); inline citations in methods when defining SR/SF mathematically (Dayan 1993, Barreto 2017); occasional in results when comparing baselines.
- **Bio-AI bridge style**: **Light** — the bio framing is concentrated in the abstract and in 2-3 inline citations to Dayan (SR origin). The paper *uses* a hippocampal-inspired construct (SR/SF) but presents results in pure deep-RL idiom; bridge implicit rather than foregrounded.
- **Section ratios**: Intro ~12%; Related Work ~6%; Preliminaries §3 ~10%; Methods (architecture/loss) §4 ~12%; Results §5-§7 ~50%; Discussion ~3%; Limitations & Broader Impact ~5%; refs.
- **Methods presentation**: **Standalone §3 Preliminaries + §4 Method**, methods front-loaded before any experiment.
- **Results vs Discussion**: **Strictly separate** — Discussion is short (~1 paragraph), followed by a labelled **§9 Limitations and Broader Impact** as its own short section. This is the modal NeurIPS structure for a method paper.
