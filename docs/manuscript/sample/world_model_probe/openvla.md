# OpenVLA + Orozco 2025 probe

* OpenVLA paper: <https://arxiv.org/abs/2406.09246>
* OpenVLA HF: <https://huggingface.co/openvla/openvla-7b>
* Orozco 2025 (probe): <https://arxiv.org/abs/2509.24559>
* Replication repo: <https://github.com/FlexCode29/reproducibility-emergent-world-model-openvla>

## Public checkpoints — verified

* `openvla/openvla-7b` (~7B param, fused DinoV2 + SigLIP encoder + Llama-2-7B). Public.
* `moojink/openvla-7b-oft-finetuned-libero-{spatial,object,goal,10}` — task-finetuned.
* Replication repo is public; uses HF transformers `AutoModelForVision2Seq`.

## Architecture pieces

* Visual encoder: DinoV2 + SigLIP (fused), 224×224 input, ~256 visual tokens per image.
* Projector: MLP from visual to LM-token space.
* LM: Llama-2-7B; 32 transformer layers, 4096 d_model.
* No explicit recurrent memory — context-window-based memory only.
* Probe target in Orozco: state-transition vectors in *embedding arithmetic* sense
  (k(s_t+1) − k(s_t) ≈ residual-stream delta). They use linear + MLP probes on
  per-layer activations.

## Position labels available?

* **LIBERO datasets** (manipulation) have end-effector and object xy/z available, not
  agent-position-in-environment. These do not match our paper's "navigation GPS"
  framing cleanly.
* **OpenVLA on navigation tasks:** there is no released finetune for indoor PointNav /
  Habitat. Would have to construct one ourselves — out of scope.

## Probe-readiness

* FlexCode29 repo includes `cache_embeddings.py --model_path openvla/openvla-7b
  --layers 0,30` — verified. Extracts residual stream activations.
* Pipeline: cache embeddings → train linear/MLP probes → analyze.
* Multi-GPU (uses GPUs 0-2). 7B model on single A100 (40GB) is tight; A100-80 OK.

## Estimated wall-clock

* Setup OpenVLA + downloading 7B weights: 1-2h depending on bandwidth.
* Extracting per-layer activations on a LIBERO subset (say 200 trajectories ×
  ~50 frames): ~1-2h on a single A100.
* Probe training: 30 min for the layer sweep.
* Total: **5-7h**, tight but feasible.

## Killer caveats

1. **Domain mismatch.** LIBERO is manipulation; arm xy ≠ navigation xy. Our paper's
   prediction is about *navigation* GPS. Replicating Orozco's setup gives a probe-
   exists / probe-doesn't result, not a falsifier of the encoder-bandwidth claim.
2. **Encoder bandwidth axis is hard to vary.** OpenVLA's encoder is fixed (DinoV2 +
   SigLIP fused). To create our 5-condition design we'd have to *finetune* the
   projector + early LM layers under different visual transforms — that's a multi-
   day fine-tuning, not 8h.
3. The Orozco result already exists; replicating it is not a new contribution. Adding
   the bandwidth axis would be — but see (2).
4. **Memory horizon.** OpenVLA has no recurrence; "integration over time" is bounded
   by the context window. The claim "linear-readable code in memory erodes with
   encoder bandwidth" needs *memory*, which OpenVLA proper doesn't have. Use the OFT
   variant or one of the ACT-style finetunes if we go this route.

## Verdict

**Cite, do not run.** Orozco 2025 is the closest existing precedent and is already
in our related work. Within 8h, attempting to extend it to a navigation domain or to
vary encoder bandwidth introduces too many moving parts. Save the contribution for a
follow-up.
