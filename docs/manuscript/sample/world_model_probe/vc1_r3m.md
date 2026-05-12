# VC-1 (Majumdar 2023) and R3M (Nair 2022)

Frozen pretrained visual encoders for embodied AI. Considered together because they
fill the same role in our plan: swap-in encoders of varying bandwidth.

## VC-1 — verified

* Paper: <https://arxiv.org/abs/2303.18240>
* Repo: <https://github.com/facebookresearch/eai-vc>
* HF (large): <https://huggingface.co/facebook/vc1-large>
* HF (base): <https://huggingface.co/facebook/vc1-base> (referenced from MODEL_CARD)
* Sizes: ViT-B (~86M) and ViT-L (~307M); 224×224 input; 1024-d output for ViT-L
* Pretraining: 4000h egocentric video + ImageNet, MAE objective
* Loading: `from vc_models.models.vit import model_utils; model, _, _, _ =
  model_utils.load_model("vc1_vitl")`

## R3M — verified

* Paper: <https://arxiv.org/abs/2203.12601>
* Repo: <https://github.com/facebookresearch/r3m>
* Sizes: ResNet-18, ResNet-34, ResNet-50 (~12M / 22M / 25M params)
* Pretraining: Ego4D (3670h), TCN + L1 + language alignment losses
* Loading: `from r3m import load_r3m; m = load_r3m("resnet50")` — auto-downloads via
  torch.hub.

## Position labels available?

Neither encoder has built-in position labels — they're *encoders*, not full agents.
Used in our plan to embed Memory Maze frames; ground-truth `agent_pos` comes from the
Memory Maze NPZ files.

## Probe-readiness

* VC-1: returns `(B, 1024)` CLS-pooled embedding by default. Can hook earlier
  transformer blocks if we want layer-wise probes.
* R3M: returns `(B, 2048)` ResNet-50 final-layer pooled embedding. Hook intermediate
  ResNet stages for layer-wise probes.

Both: *trivially* probe-ready. No internal recurrence, so probing a single embedding
per frame is the natural unit.

## Estimated wall-clock

* Loading: ~30s.
* Embedding 500 trajectories × 1001 frames at 224×224 with VC-1-Base on a single A100:
  ~25-30 min.
* R3M r18 same task: ~10-15 min (smaller model + ResNet is fast).
* Probe training (linear + MLP) on cached features: <5 min each.

## Killer caveats

1. Memory Maze frames are **64×64**; upsampling to 224×224 adds little visual signal
   but the encoders expect that resolution. Both VC-1 and R3M have been used this way
   in CortexBench / D4RL benchmarks, so it's standard practice but is a known limitation.
2. R3M's pretraining objective includes language alignment — its features are biased
   toward semantic / object-centric content, so spatial xy decoding may be weak even
   for the highest-bandwidth variant. Could be a feature (interesting null) or a bug.
3. VC-1 does *not* expose intermediate layer outputs through the standard loader — need
   to hook manually.

## Verdict

**Use both, plus DINOv2 (separate file).** The 3-encoder × 3-size grid (VC-1-{B,L},
R3M-{r18,r34,r50}, DINOv2-{S,B,L,G}) gives a clean across-family bandwidth axis if we
stick to the static-encoder probe. For the LSTM-on-top variant, pick one encoder
family and vary size.
