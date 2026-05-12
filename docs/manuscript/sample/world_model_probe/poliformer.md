# PoliFormer (Zeng 2024)

* Paper: <https://arxiv.org/abs/2406.20083> (CoRL 2024 Outstanding Paper Award)
* Repo: <https://github.com/allenai/PoliFormer>

## Public checkpoints — verified

Repo provides a download script for "trained model checkpoints" (text-nav, pure
box-nav, text+box-nav variants). README confirms: *"For evaluation only, a trained
model checkpoint... Download pretrained ckpt"*.

## Architecture pieces

* Visual encoder: vision transformer (paper says "foundational ViT", DinoV2 backbone).
* Sequence model: causal transformer decoder for long-context memory + reasoning.
* Action head: discrete action distribution.

Paper claims SOTA on ObjectNav (CHORES challenge) and PointNav with very long
context windows.

## Position labels

* AI2-THOR / ProcTHOR-Objaverse simulator exposes ground-truth agent xy via
  `controller.last_event.metadata['agent']['position']`.
* Training data HDF5 (`hdf5_sensors.hdf5`) contains "house id, starting pose, and
  target object type/id" per README — starting pose only, not per-step xy. We'd
  need to re-run rollouts in AI2-THOR to get per-step xy.

## Probe-readiness

* Inference script is documented; hidden-state extraction is not. ~1-2h of code-
  reading to expose transformer block outputs.
* Strong dependency: AI2-THOR install (custom PyPI index, X-server, GL/OSMesa
  rendering — historically an install-pain point on cluster machines).

## Estimated wall-clock

* AI2-THOR install + ckpt download + ProcTHOR-Objaverse asset download: ~2-3h on a
  fresh machine, sometimes much more if Vulkan / GPU rendering misbehaves on RCP.
* One eval episode: ~1 min.
* 200 episodes × 5 sensor variants × hidden-state caching: ~3-4h.
* **Total: 6-9h, mostly setup-dominated.**

## Killer caveats

1. **Setup risk is the killer.** AI2-THOR has historically been hostile to headless
   cluster environments. Multiple RCP-style failure modes documented in the
   PoliFormer issues tracker.
2. Closest match in *spirit* (transformer-based PointNav navigator), but the
   environment is not Habitat / Gibson — it's ProcTHOR-Objaverse. Different scenes,
   different action space, different sensor model.
3. The 5-condition encoder-bandwidth design requires re-finetuning the visual encoder
   (or pre-blurring the input pixels), which is *not* a probe — it's a re-train.
4. License: AI2 model licenses for academic use, generally permissive but check.

## Verdict

Cite as the closest spiritual match for "what a transformer PointNav navigator looks
like in 2024". Do not run in 8h — setup dominates the budget. If we get extra time
post-deadline, this is the highest-value follow-up because the architecture is closest
to our claim.
