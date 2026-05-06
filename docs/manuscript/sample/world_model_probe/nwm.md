# Navigation World Models (Bar 2024)

* Paper: <https://arxiv.org/abs/2412.03572> (CVPR 2025)
* Repo: <https://github.com/facebookresearch/nwm>
* HF: <https://huggingface.co/facebook/nwm>
* Project: <https://www.amirbar.net/nwm/>

## Public checkpoints — verified (HF)

5 CDiT (Conditional Diffusion Transformer) checkpoints, all under CC-BY-NC-4.0:
* CDiT/S (50M, 100k steps, RECON+SCAND+TartanDrive+HuRoN)
* CDiT/B (200M, same data)
* CDiT/L (700M, same data)
* CDiT/XL (1B, same data)
* CDiT/XL (1B, 200k steps, +Ego4D)

`.pth.tar` format. Inference script: `isolated_nwm_infer.py`.

## Architecture pieces

* Encoder: ViT-style; encodes context observations into tokens.
* Conditional Diffusion Transformer: predicts future visual frames given past frames
  + action sequence.
* Output: pixels (diffusion).

## Position labels

The training datasets (RECON, SCAND, TartanDrive, HuRoN) include odometry / position;
the training pipeline uses these for action conditioning. So they should be available
alongside trajectories — but the *NWM model itself* receives action xy not absolute
xy as input, so probing the model's hidden state for absolute xy makes most sense
only after a long roll-in.

## Probe-readiness

* `isolated_nwm_infer.py` exposes the inference loop but not internal features per the
  README; would require monkey-patching diffusion-transformer block outputs.
* CDiT internals are 24+ transformer layers; probes have to target a chosen layer.
* Diffusion sampling has multiple denoising steps — what timestep do you probe? Adds
  a hyperparameter axis.

## Estimated wall-clock

* HF download for CDiT/S (50M): 5 min, ~200MB.
* CDiT/L (700M): 30+ min, 3GB.
* Diffusion inference per trajectory is *slow* (~5-30s/sample with 50 denoising steps),
  so 200 trajectories × 32 frames = ~5h just for inference on CDiT/L.
* Setup + custom hooking: 2-3h.
* **Total: 7-9h on CDiT/S, infeasible on CDiT/L within 8h.**

## Killer caveats

1. Diffusion-based, so the "hidden state" notion is fuzzier — at which denoising step
   do you probe? The original Orozco-style residual-stream probe is harder to motivate.
2. License is CC-BY-NC-4.0; OK for research / paper, but not for any code release that
   redistributes the weights.
3. Per-frame inference is slow; probe across 5 conditions × hundreds of trajectories
   blows the 8h budget.
4. Encoder is fixed (ViT-S/B/L/XL across NWM sizes — but you have to also retrain to
   change it, since the diffusion predictor is co-trained).

## Verdict

Cite, do not run within 8h. Excellent reference for the "world-model in encoder
rather than recurrence" framing in our discussion (and is already cited in our
related-work via `bar2024navigationworldmodels.md`). NWM as an experimental probe
target is too heavy for the budget; the inference cost alone disqualifies it.
