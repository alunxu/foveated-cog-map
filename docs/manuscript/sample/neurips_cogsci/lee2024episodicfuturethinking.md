# Episodic Future Thinking Mechanism for Multi-agent Reinforcement Learning

- **Authors**: Dongsu Lee, Minhae Kwon
- **Venue**: NeurIPS 2024 (poster)
- **Year**: 2024
- **OpenReview**: https://openreview.net/forum?id=rL7OtNsD9a
- **NeurIPS PDF**: https://proceedings.neurips.cc/paper_files/paper/2024/file/1588dc2b2ef339d6e4c47d212e36f991-Paper-Conference.pdf
- **Project**: https://sites.google.com/view/eftm-neurips2024

## Abstract (paraphrased)
Introduces an episodic future thinking (EFT) mechanism for multi-agent RL inspired by human cognitive neuroscience evidence that prospective and counterfactual reasoning share neural substrates. The agent maintains a multi-character policy parameterised by reward-component weights, infers other agents' "characters" from observation-action trajectories via inverse rational control, and selects foresighted actions by mentally simulating future observations. Driving-simulator and StarCraft experiments show consistent improvements over MAPPO and counterfactual baselines.

## Structure analysis
- **§2 Related Work**: **Yes, standalone**, ~4 sub-themes (Episodic Future Thinking ~2 par with cogsci framing, Model-based RL ~2 par, Machine Theory of Mind ~2 par, False Consensus Effect ~1 par). Total ~7 paragraphs.
- **Intro**: ~6 paragraphs, **motivation-first** with explicit cogsci framing ("Understanding human decision-making in multi-agent interactions is a significant focus in cognitive science"). Ends with **3 explicit bulleted contributions** under heading "Summary of contributions:".
- **Where prior work appears**: Standalone §2 (concentrated), occasional inline citations in methods (defining IRC, MA-POMDP). Limited in results.
- **Bio-AI bridge style**: **Dedicated subsections inside §2 Related Work** ("Episodic Future Thinking" subsection literally opens "Cognitive neuroscience aims to understand how humans use memory in decision-making"). Brain/cognitive-science *grounding* lives in §2 and intro; methods are pure ML; **Discussion** does not return to the bio framing — it is a 1-paragraph Conclusion + 1 Broader-Impact paragraph + 1 Limitations paragraph.
- **Section ratios**: Intro ~10%; Related Works ~12%; Methods (§3 Character Inference + §4 Foresight Action Selection) ~30%; Experiments (§5) ~35%; Discussion (§6 Conclusion+Broader Impact+Limitations) ~5%; refs/appendix.
- **Methods presentation**: **Standalone §3 + §4** with explicit subsections (Problem Formulation, Multi-character Policy, Character Inference, Foresight Action Selection).
- **Results vs Discussion**: **Strictly separate.** §5 Experiments is empirical only; §6 Discussion is short and contains Conclusion, Broader Impacts, Limitations as 3 labelled paragraphs.
