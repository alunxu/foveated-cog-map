"""Upload the canonical post-retrain checkpoints for all 5 visual
conditions to alunxu/spatial-memory-checkpoints, replacing any stale
files. Designed to run on RCP where /scratch holds the checkpoints.

Uploads the FULL training trajectory (all per-update checkpoints) for
every condition so teammates can do training-trajectory memory
analysis at any granularity.

Mapping (canonical post-retrain, hp-consistent):
  blind             -> blind_izar          (ckpts 0..34, 35 files)
  coarse            -> dh-probe-1          (ckpts 0..49, 50 files)
  foveated          -> dh-probe-2          (ckpts 0..49, 50 files)
  uniform           -> dh-probe-3          (ckpts 0..49, 50 files)
  foveated_logpolar -> dh-probe-4          (ckpts 0..49, 50 files)

Total: 235 files, ~10.5 GB.
"""
import os, sys, json
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete

REPO = "alunxu/spatial-memory-checkpoints"
CKPT_ROOT = Path("/scratch/wxu/habitat_checkpoints_rcp")

# (target folder on HF, RCP source dir, list of ckpt indices)
PLAN = [
    ("blind",             "blind_izar",  list(range(0, 35))),    # 0..34
    ("coarse",            "dh-probe-1",  list(range(0, 50))),    # 0..49
    ("foveated",          "dh-probe-2",  list(range(0, 50))),
    ("uniform",           "dh-probe-3",  list(range(0, 50))),
    ("foveated_logpolar", "dh-probe-4",  list(range(0, 50))),
]


def main():
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)

    print(f"Inspecting repo {REPO} ...")
    info = api.repo_info(REPO, repo_type="model")
    existing = sorted(s.rfilename for s in info.siblings)
    keep = {"README.md", ".gitattributes"}
    to_delete = sorted(f for f in existing if f not in keep)
    print(f"  existing files: {len(existing)}; "
          f"will delete {len(to_delete)} stale entries")

    # Verify all sources first.
    plan_files = []
    total_bytes = 0
    for hf_dir, src_dir, ckpts in PLAN:
        for k in ckpts:
            src = CKPT_ROOT / src_dir / f"ckpt.{k}.pth"
            if not src.exists():
                print(f"  MISSING source: {src}", file=sys.stderr)
                sys.exit(1)
            sz = src.stat().st_size
            total_bytes += sz
            target = f"{hf_dir}/ckpt.{k}.pth"
            plan_files.append((hf_dir, target, src, sz))
    print(f"Plan: {len(plan_files)} files, {total_bytes/1e9:.2f} GB total")

    # README — minimal load-only content.
    readme_path = Path("/tmp/spatial_memory_README.md")
    readme_path.write_text(README_TMPL)

    # Step 1: single commit that deletes stale and uploads README.
    delete_ops = [CommitOperationDelete(path_in_repo=f) for f in to_delete]
    if delete_ops:
        print(f"\n[1/{len(PLAN)+1}] Deleting {len(delete_ops)} stale files + "
              f"replacing README ...")
        api.create_commit(
            repo_id=REPO, repo_type="model",
            operations=delete_ops + [
                CommitOperationAdd(path_in_repo="README.md",
                                   path_or_fileobj=str(readme_path))
            ],
            commit_message="Clean stale ckpts + minimal README "
                           "(post-retrain release in flight)",
        )

    # Step 2..N+1: one commit per condition (chunked; 35-50 files / ~2 GB each).
    for i, (hf_dir, src_dir, ckpts) in enumerate(PLAN, start=2):
        ops = []
        sub = 0
        for k in ckpts:
            src = CKPT_ROOT / src_dir / f"ckpt.{k}.pth"
            sz = src.stat().st_size
            sub += sz
            ops.append(CommitOperationAdd(
                path_in_repo=f"{hf_dir}/ckpt.{k}.pth",
                path_or_fileobj=str(src)))
        print(f"\n[{i}/{len(PLAN)+1}] Uploading {hf_dir}: "
              f"{len(ops)} files, {sub/1e9:.2f} GB ...")
        api.create_commit(
            repo_id=REPO, repo_type="model",
            operations=ops,
            commit_message=f"Add full training trajectory for {hf_dir} "
                           f"({len(ops)} ckpts, post-retrain)",
        )

    print("\nDONE — final HF state:")
    info = api.repo_info(REPO, repo_type="model")
    n = len(info.siblings)
    print(f"  {n} files in repo")


README_TMPL = """\
# Spatial-memory checkpoints (5 visual conditions)

Frozen post-training DD-PPO PointNav agents on Habitat for five visual sensor
conditions on a shared ResNet-18 + 3-layer LSTM (512-d) backbone. Hidden
state `h_2` (top LSTM layer) is the canonical 512-d cognitive-map readout.

| folder              | encoder                                                | final ckpt |
| ------------------- | ------------------------------------------------------ | ---------- |
| `blind/`            | no visual encoder                                      | `ckpt.34.pth` |
| `coarse/`           | 48 x 48 RGB, 1 x 1 encoder feature map                 | `ckpt.49.pth` |
| `foveated/`         | 256 x 256 RGB, eccentricity Gaussian blur, 4 x 4 map   | `ckpt.49.pth` |
| `foveated_logpolar/`| 64 x 64 log-polar resampled, ~2 x 2 map                | `ckpt.49.pth` |
| `uniform/`          | 256 x 256 RGB, no blur, 4 x 4 map                      | `ckpt.49.pth` |

The **full training trajectory** is included for every condition (one
checkpoint per DD-PPO save event): sighted ckpts `0..49` (50 each),
blind ckpts `0..34` (35). This supports per-checkpoint training-time
analyses (subspace evolution, probe-R^2 trajectories, eigenspectrum
emergence, etc.) at the finest available granularity.

## Load a checkpoint

```python
import torch
from huggingface_hub import hf_hub_download

cond = "foveated"          # or: blind | coarse | uniform | foveated_logpolar
ckpt_path = hf_hub_download(
    repo_id="alunxu/spatial-memory-checkpoints",
    filename=f"{cond}/ckpt.49.pth",
)

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state_dict = ckpt["state_dict"]              # actor-critic policy weights
config     = ckpt["config"]                  # habitat-baselines OmegaConf
```

Each `.pth` is a habitat-baselines checkpoint with keys `state_dict`,
`config`, and `extra_state` (training step counters).

## Rebuild the policy and read `h_2`

```python
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.common import get_action_space_info
from habitat_baselines.config.default import get_config
from habitat.config.default_structured_configs import HabitatConfigPlugin

# 1. Build the same env the policy was trained on (for obs/action spaces).
env_config = config.habitat                  # already inside ckpt
# ... construct the eval env from env_config (see code repo) ...

# 2. Instantiate the policy class registered for this config and load weights.
policy_cls = baseline_registry.get_policy(
    config.habitat_baselines.rl.policy.name)
policy = policy_cls.from_config(
    config=config,
    observation_space=env.observation_space,
    action_space=env.action_space,
)
policy.load_state_dict(state_dict)
policy.eval()

# 3. Run a rollout. The recurrent hidden state has shape
#    (num_envs, num_layers=3, hidden=512). h_2 is the top layer:
#       h_2 = recurrent_hidden_states[:, 2, :]
#    Pass `recurrent_hidden_states` back into `policy.act(...)` each step.
```

Code, configs, and the deterministic-rollout probing pipeline that produced
this release: <https://github.com/alunxu/foveated-cog-map>.
"""


if __name__ == "__main__":
    main()
