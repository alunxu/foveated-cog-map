# DINO-WM (Zhou 2024 / ICML 2025)

* arXiv: <https://arxiv.org/abs/2411.04983>
* Repo: <https://github.com/gaoyuezhou/dino_wm>
* Project: <https://dino-wm.github.io/>

## Public checkpoints — verified

* Repo states: "We have uploaded our trained world model checkpoints for **PointMaze,
  PushT, and Wall** here under `checkpoints/`." Verified via repo README. No
  checkpoint released for Rope, Granular, Deformable.

## Architecture pieces

* Encoder: **frozen DINOv2-Base** (ViT-B/14), patch features (Z_t in R^{N_patch × d}).
* Predictor: ViT-style transformer that predicts patch features at t+1 conditioned on
  past patches and action.
* No reconstruction loss; pure feature-space prediction.
* No recurrence — purely a transformer over a context window of past frames.

## Position labels available?

* **PointMaze**: explicit (x,y) state — yes, accessible via the underlying gym env.
  Image is the rendered top-down or first-person view (paper renders the maze).
* **PushT**: object pose, not navigation xy.
* **Wall**: 2D point + barrier — has xy.

The PointMaze + Wall combination would let us probe the predictor's mid-layer hidden
states for xy of the agent, given encoder features at varying input resolutions.

## Probe-readiness

* Hooking forward activations on a HF / PyTorch ViT is straightforward (register hooks
  on attention block outputs).
* But: codebase is wrapped in Hydra configs; getting clean hidden-state extraction in
  isolation requires ~1-2h of code-spelunking on top of `git clone`.
* No published probing API in the repo.

## Estimated wall-clock

* Clone + env setup (Mujoco 210 dependency!): 1.5h. Mujoco install is notoriously
  finicky; Mac-only setup may break. RCP linux: probably OK.
* Inference over 1000 trajectories of PointMaze: ~30 min on A100.
* Probe training: 10 min.
* Total realistic: **3-4h on RCP**, longer if mujoco install fights us.

## Killer caveats

1. PointMaze is a *2D top-down* environment, and DINO-WM is shown to "near-perfectly"
   solve the task, suggesting the encoder already saturates the position information.
   Less interesting variation than Memory Maze.
2. Encoder is *fixed* (DINOv2-B). To vary "encoder bandwidth" we'd swap to DINOv2-S /
   DINOv2-L, which means retraining the predictor — too expensive in 8h.
3. Action conditioning is via FiLM-style modulation; the integrative role of "memory"
   here is short (a few past frames in context, not a full trajectory).

## Verdict

Worse fit than Memory Maze. Use only if Memory Maze data download fails. The frozen-
DINOv2 encoder is a useful pretrained component, but DINO-WM the published checkpoint
adds the *predictor* on top, and varying encoder bandwidth requires retraining that
predictor. Replicate the encoder-side claim using bare DINOv2 instead.
