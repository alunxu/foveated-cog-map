# 8-Hour Pretrained-Model Probe Plan

Goal: empirically test the architecture-agnostic prediction (encoder-bandwidth → linear position-code in memory)
on 1-2 pretrained models in 8 hours, with verified checkpoints and a deterministic probing
protocol that already exists in the literature.

## TL;DR

* **Top-1 (recommended): Memory-Maze frozen-encoder probe.** Probe DINOv2 / VC-1 / R3M
  on the Memory Maze offline dataset at varying input resolutions, with `agent_pos` and
  `targets_vec` as ground-truth probe targets. Adds a small trainable LSTM on top to
  recreate the encoder→memory split. Reuses the published Pasukonis-Lillicrap-Hafner 2023
  ICLR probing protocol (4-layer MLP, 1024 units, predict object positions from hidden
  states). All pieces verified public: encoders on HF, dataset on Drive, probe code in
  the memory-maze repo.
* **Top-2 (backup): ViNT activation probe.** Frozen ViNT (foundation navigation
  transformer, EfficientNet + 4-layer Transformer, public ckpt on Drive) processes RECON
  trajectories (which include `position` xy + `yaw` per frame); we probe the transformer
  context tokens for xy. No training required. Switch here if Memory-Maze data download
  fails or is too slow.
* **Honest negative:** if neither works in 8h we report "tried, here is the breakdown"
  in §5.4 and fall back to the qualitative architecture-agnostic claim that already
  cites Orozco 2025 (OpenVLA) and Bar 2024 (NWM).

## Why probing pretrained encoders is the right scoped experiment

Our paper's prediction is about *encoder bandwidth*, not specifically RSSM vs LSTM. A
clean test is: take ONE recurrent integrator (a small LSTM we train ourselves), feed it
features from encoders of *known* spatial bandwidth (DINOv2 small / base / large /
giant; VC-1 base/large; R3M r18/r34/r50; or via input-resolution downsampling on a
single backbone), and probe the LSTM hidden state for `agent_pos`. If the same
monotone falloff appears across pretrained encoders, our claim that "the regularity is
in capacity allocation, not in the specific encoder family" lands as a transformer-era
falsifier.

Training a Dreamer/world-model from scratch in 8h is infeasible (Pasukonis et al. 2023
report 14 GPU-days for one Dreamer run on Memory Maze 9×9). NM512/dreamerv3-torch ships
**no pretrained checkpoints**. R2I/S5WM ships **no checkpoints**. So the cleanest path
that respects the 8h budget is the encoder-side probe.

## Top-1 plan: Memory-Maze frozen-encoder probe (~7h)

### Step 0 — Sanity prep (0:00 → 0:30)

* `pip install memory-maze` (verified on PyPI, 1.0.2)
* Download a 1-trajectory subset to confirm NPZ schema:
  `image (1001,64,64,3) uint8`, `agent_pos (1001,2) float`, `agent_dir (1001,2)`,
  `maze_layout (9,9) binary`, `target_pos (1001,2)`, `targets_vec (1001,3,2)`,
  `action (1001,6)` — confirmed from the published Table 2 of Pasukonis 2023.
* Download URL: <https://drive.google.com/drive/folders/1RcnkTZVwEHnAQeEuw7X8Y1RPSmrFLDFB>
  (folder `memory-maze-9x9`). Full dataset is ~100GB; fetch ~200 trajectories
  (~600MB) for prototyping, then ~2k for the run.

### Step 1 — Encoder zoo + resolution sweep (0:30 → 2:30)

Choose **one** of (a) or (b):

* **(a) Vary capacity within a family.** Run all of {VC-1 base, VC-1 large, R3M r18,
  R3M r34, R3M r50, DINOv2-S, DINOv2-B, DINOv2-L} on the 64×64 Memory-Maze frames
  (upsampled to 224 via bilinear; this is what published embodied-AI work does,
  acknowledged limitation).
* **(b) Vary resolution within one encoder.** Take DINOv2-Base, feed images at native
  64, 32, 16 by downsampling+upsampling, plus a foveated/log-polar variant matching
  our paper's transforms (we already have the code in `src/.../sensors/`). This is
  the *cleanest analogue* of our 5-condition design.

Recommend (b) — closer 1:1 mapping to our paper's contrast.

Cache features to disk: for each trajectory, output a `(T, D)` tensor of CLS-pooled
features. With DINOv2-B (D=768), 1001 frames × 2000 trajectories × 768 × 4 bytes ≈ 6 GB.

Wall-clock: DINOv2-B on a single A100 ≈ 200 fps for 224×224, so 2000×1001 frames ≈ 2.8h.
**Budget cut:** use 500 trajectories instead of 2000, gives ~42 min per condition × 5
conditions = 3.5h max. Run in parallel across GPUs if available.

### Step 2 — Train small LSTM integrator on top of frozen features (2:30 → 4:30)

Architecture (mirrors our paper's recurrent core):
* LSTM(input=D_enc, hidden=512, layers=2)
* Input: encoder features concatenated with one-hot last action
* Train objective: predict next-frame encoder feature (representation prediction loss),
  with auxiliary action-classification head trained on the offline action targets

Train for 30k steps on the cached features. With sequence length 100, batch 16, this
fits in memory and trains in ~1.5h on a single GPU. (No environment, no rollouts,
purely offline supervised.)

### Step 3 — Probe LSTM hidden state for agent_pos (4:30 → 6:00)

Following Pasukonis 2023 Appendix D — they motivate **4-layer MLP, 1024 units** but
report linear-probe baselines too. We need both:

* **Linear probe** (R² on `agent_pos`, second half of trajectory only — burn-in
  removed): the headline number.
* **MLP probe** (R² on `agent_pos`): the format-shift baseline.

Concat `(h_t, c_t, last_action)` as probe input. Train probe for 1M *frames* (= ~3k
trajectories × 333 frames each) on train split, eval on held-out 1k trajectories.
Probe training runs in <10 min (small MLP, GPU optional).

**Key plot:** linear probe R² vs encoder spatial bandwidth, plus the same for MLP
probe. Predicted shape: linear R² falls monotonically with bandwidth; MLP R² is
much flatter (recovery from format shift, mirroring our paper's fov-LP/uniform
result).

### Step 4 — Memory transplant analogue (6:00 → 7:00)

Take the LSTM trained on bottleneck features (smallest resolution / lowest
bandwidth) and feed it features from the rich encoder, and vice versa. Measure
representation-prediction loss. Predicts asymmetry: rich→bottleneck breaks more than
bottleneck→rich. This recreates our paper's transplant experiment in the pretrained-
encoder regime.

### Step 5 — Plotting + writeup (7:00 → 8:00)

One figure (encoder bandwidth × probe R² × probe class), one table (transplant deltas).
Drop into §5.4 of the manuscript.

## What counts as success

* **Strong (publishable extension):** monotone fall in linear R² across ≥3 encoder
  bandwidths, MLP recovers ≥2× the linear gap on the lowest-bandwidth condition,
  asymmetric transplant. We add a paragraph + figure in §5.4 framing it as
  "the prediction reproduces in modern frozen encoders + LSTM, on a different
  environment (Memory Maze) and with a published probing protocol".
* **Moderate:** monotone trend on linear, but no MLP recovery. We claim "the
  encoder-bandwidth → linear-readable position trend reproduces; format-shift
  recovery is a Habitat-specific signature." Discuss honestly.
* **Negative (all R² ≈ 0 or non-monotone):** report; argue the prediction is
  Habitat-/Wijmans-specific and may not transfer to the offline-replay /
  scripted-policy regime of Memory Maze. Still informative for the discussion
  section.

## Top-2 backup: ViNT activation probe (~5h if Top-1 stalls in step 0–1)

* Repo: <https://github.com/robodhruv/visualnav-transformer>; checkpoints on Google Drive
  (linked in repo README). 4-layer Transformer over EfficientNet-encoded images, ~30M
  params.
* Data: RECON public release from Berkeley (~10GB), each trajectory has `position`
  (xy in meters) + `yaw` per frame — verified from ViNT data-loader code references.
* Probe: extract the post-attention token embedding for the *current* observation token
  (last in the context window) at each frame; linear/MLP regress to xy. Vary context
  length (1, 4, 8 frames) and image resolution (downsample input from 96×96 to 32×32
  to 16×16) to recreate the bandwidth axis.
* No training required — purely inference + probe. ~3-4h end-to-end.
* Caveat: ViNT's xy is *robot-frame relative*, not allocentric, so this tests
  short-horizon integration only.

## What we are NOT doing in 8h, and why

* **Training Dreamer/RSSM/IRIS/R2I from scratch** — 14+ GPU-days per condition.
* **NWM (Bar 2024)** — 1B-param diffusion transformer; inference is heavy and the
  HF model weights don't expose hidden states via the standard inference script;
  refactoring would eat the budget.
* **OpenVLA replication (Orozco 2025 / FlexCode29)** — runnable, but LIBERO is
  manipulation, not navigation; the "position-code in memory" probe target is
  awkward (arm xy ≠ navigation xy). Use only if both Top-1 and Top-2 fail and we
  pivot to "VLA emergent state" framing.
* **DINO-WM checkpoint (Zhou 2024)** — checkpoint exists but only for PointMaze
  / PushT / Wall, not a 3D maze with position labels in the published probing
  format.
* **PoliFormer** — checkpoint exists but for ProcTHOR-Objaverse text-nav; running
  one rollout requires AI2-THOR install (~1-2h) and the position labels are not
  obviously exposed in the eval pipeline. Possible but high setup risk.

## Honest probability assessment

| Plan | P(8h success) | P(strong / publishable) |
|------|---------------|-------------------------|
| Top-1 (Memory-Maze frozen-encoder + LSTM) | ~70% | ~40% |
| Top-2 (ViNT + RECON) | ~80% | ~25% |
| Both | ~90% finishes something | ~50% paper-worthy |

Rationale: the encoder-bandwidth signal exists in our Habitat work and is mechanistically
plausible. Two main failure modes are (a) the offline-replay / scripted-policy regime
of Memory-Maze doesn't elicit the same encoder-vs-integration tradeoff (the LSTM may
collapse to ignore vision or to ignore action), and (b) the LSTM training in step 2
under-fits in 1.5h, making the probe results noisy. We mitigate (a) by also reporting
the static-encoder probe (no LSTM) as a control: if encoder-only linear R² already
varies with bandwidth, the bandwidth claim is supported even without the integration
story.

## Source repos / URLs to bookmark

* Memory Maze: <https://github.com/jurgisp/memory-maze>
* Memory Maze offline data: <https://drive.google.com/drive/folders/1RcnkTZVwEHnAQeEuw7X8Y1RPSmrFLDFB>
* Pasukonis 2023 paper (probing protocol): <https://openreview.net/pdf?id=yHLvIlE9RGN>
* DINOv2: <https://github.com/facebookresearch/dinov2> (HF: facebook/dinov2-{small,base,large,giant})
* VC-1: <https://huggingface.co/facebook/vc1-large>
* R3M: <https://github.com/facebookresearch/r3m>
* ViNT: <https://github.com/robodhruv/visualnav-transformer>
* Orozco probe replication: <https://github.com/FlexCode29/reproducibility-emergent-world-model-openvla>

## Concrete commands to start with tomorrow

```bash
pip install memory-maze
python -c "import memory_maze, gym; e=gym.make('memory_maze:MemoryMaze-9x9-ExtraObs-v0'); o=e.reset(); print(list(o.keys()))"

# Pull DINOv2 from HF
python -c "import torch; m=torch.hub.load('facebookresearch/dinov2','dinov2_vitb14'); print(sum(p.numel() for p in m.parameters()))"

# Pull memory-maze data subset (first 200 NPZ files of 9x9 train split)
gdown --folder https://drive.google.com/drive/folders/1RcnkTZVwEHnAQeEuw7X8Y1RPSmrFLDFB
# (or fetch a single file via the web UI to confirm schema before bulk download)
```
