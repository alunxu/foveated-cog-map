# DreamerV3 (Hafner 2023)

* arXiv: <https://arxiv.org/abs/2301.04104>
* Official JAX: <https://github.com/danijar/dreamerv3>
* PyTorch port: <https://github.com/NM512/dreamerv3-torch>

## Public checkpoints — verified

* **Official danijar/dreamerv3:** *no released pretrained checkpoints*. Hafner stated
  publicly (TalkRL podcast on Dreamer 4) that getting the paper out under current
  release climate was already exceptional; checkpoint release was deprioritised.
* **NM512/dreamerv3-torch:** *no released checkpoints*. README's note: "This codebase
  was implemented prior to major updates to DreamerV3 and does not reflect those
  changes." Author points to newer `r2dreamer` repo which is also code-only.
* **sheeprl (Eclectic-Sheep):** PyTorch DreamerV3 implementation, framework supports
  saving and reloading agents, but no public pretrained models.
* **Dreamer 4 / IamCreateAI/Dreamerv4-MC:** community Minecraft port with HF-hosted
  checkpoint, but Minecraft does not expose 2D agent xy in a probing-friendly way.

## Architecture pieces (relevant for probing)

* CNN encoder (~3M params, 64×64 → embedding dim 1024)
* RSSM = GRU sequence model + categorical stochastic state
  * `h_t` (deterministic recurrent): typical 1024 dim for Memory Maze configs
  * `z_t` (stochastic, 32 categories × 32 classes): 1024-dim flattened
  * `latent = concat(h_t, z_t)` is the standard probe input (2048 dim)
* Decoder + reward + continue heads (probe these or not, your choice)
* Inference API in NM512 port:
  `latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])`
  exposes both `latent["stoch"]` and `latent["deter"]` — straightforward extraction.

## Position labels available?

* **Memory Maze (best fit).** Built-in `agent_pos` (T,2) + `maze_layout` + `targets_pos`
  in the offline NPZ files. Pasukonis 2023 ICLR already publishes the canonical probing
  protocol with target = predicted wall layout (BCE) and predicted object positions
  (MSE). agent_pos is given as *additional probe input*, not a target, in their setup
  — to make the wall-layout probe non-trivial. We can repurpose it as a *target* for
  our position-code probe.
* **Crafter.** No 2D xy label exposed in the standard observation; would need
  modification.
* **DMC / Atari.** No spatial position semantics that match our paper's GPS claim.

## Probe-readiness

* Extracting `latent["deter"]` and `latent["stoch"]` is one line of code in the
  PyTorch port.
* Hardest part: training the world model itself (1M gradient steps, 14 GPU-days for
  Memory Maze 9×9 per the original paper).

## Estimated wall-clock

* Loading + offline probe of an *existing* checkpoint: 1-2h.
* Training Dreamer from scratch on 5 sensor conditions: ~3 weeks per condition.
* **Within 8h budget: out of scope** unless someone releases checkpoints.

## Killer caveats

1. **No public checkpoints.** Hard blocker for the full Dreamer-on-Memory-Maze
   replication of our paper's prediction.
2. Even if we trained, varying "encoder bandwidth" inside DreamerV3 means modifying
   the CNN — not the same as our 5-condition Habitat sensor design (foveation /
   log-polar / uniform). Less direct mapping.
3. Categorical stochastic state z_t is non-Gaussian; linear-probe interpretation
   needs care vs Habitat continuous sensors.

## Verdict

Use only as ablation if pretrained checkpoints surface. Within 8h, replace with
**frozen-encoder + small-LSTM** probe on Memory Maze offline data (see EXPERIMENT_PLAN
Top-1). Memory-Maze the *environment* is still the right testbed; what's infeasible
is training Dreamer's world model in the time budget.
