# ViNT (Shah 2023) — Top-2 backup

* Paper: <https://arxiv.org/abs/2306.14846> (CoRL 2023)
* Repo: <https://github.com/robodhruv/visualnav-transformer>
* Project: <https://general-navigation-models.github.io/vint/>

## Public checkpoints — verified

Repo provides pretrained `*.pth` files via Google Drive (link in README, downloaded
into `vint_release/deployment/model_weights/`). Three released models:
* GNM (the smaller predecessor)
* ViNT (foundation navigation transformer)
* NoMaD (diffusion-policy variant)

## Architecture pieces

* Visual encoder: **EfficientNet-B0** (~5M params). Per-frame embedding ~512-d.
* Goal encoder: same EfficientNet backbone over the goal image.
* Sequence model: **4-layer Transformer** over 5 past observation tokens + 1 goal
  token; multi-head attention.
* Action head: predicts temporal distance + waypoint sequence (normalised xy).

## Position labels available?

**Yes, in the training data.** Per the data-loader convention referenced in the repo
(see `vint_train/data/data_split.py` and `dataset.py`), each trajectory pickle
contains:
* `position`: (T, 2) xy in meters (robot frame, integrated odometry)
* `yaw`: (T,) heading
* RGB frames at 96×96 (canonical resolution)

Datasets:
* **RECON** (Berkeley off-road): public release ~10GB. Has GPS-ish odometry.
* **GO Stanford 2** (indoor): public, smaller.
* **TartanDrive**, **SCAND** (sidewalks): public.
* **HuRoN/SACSon**: partially private.

## Probe-readiness

* Inference script `deployment/src/explore.py` runs the model in real-time. Hidden-
  state extraction not directly exposed but the model is small enough that we can
  monkey-patch hooks on the transformer block outputs in <1h of code.
* Probe target: predict `position[t]` from the post-attention token corresponding to
  the current observation (last in context window).

## Estimated wall-clock

* Clone + ckpt download + env: 30 min.
* Hook transformer hidden states + run inference on 200 RECON trajectories (~100k
  frames): ~1h on A100.
* Probe training (linear + MLP): 10 min.
* **Total: ~2.5h** end-to-end, fastest of the three "real navigation" candidates.

## Bandwidth axis

* Built-in: vary the input image resolution by downsampling+resizing 96→48→24→12.
* Vary context length: 1, 4, 8 past frames. Tests the "memory horizon" axis.
* Vary EfficientNet feature stage (early vs late): tests the encoder-depth axis.

## Killer caveats

1. ViNT's xy is **robot-frame relative, not allocentric**. Probing absolute position
   is not directly meaningful; we'd probe *short-horizon translation* instead. This
   is a weaker analog of our paper's "absolute GPS" probe but still tests the
   linear-readable-position-code claim.
2. ViNT was trained as a *goal-conditioned* navigator; the probe target depends on
   how trajectories are split. Use a held-out dataset split.
3. EfficientNet is a CNN, not a transformer, and the *visual* bandwidth varies with
   input resolution rather than encoder family. Less rich than the DINOv2/VC-1/R3M
   sweep. But this is also the closest match to our Habitat agent's CNN encoder.
4. RECON / GO Stanford trajectories have action commands in continuous units, not
   discrete actions like Habitat — fine for probes, awkward for transplant
   experiments.

## Verdict

Solid backup. Has *real* navigation data with position labels, public checkpoint, and
clean probe access. If Memory-Maze data download fails or the LSTM training in
Top-1 step 2 underfits in 1.5h, switch here. Recommended even as a *parallel* run if
we have GPU budget for both.
