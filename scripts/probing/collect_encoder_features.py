"""
Collect ResNet encoder output features (post-encoder, pre-LSTM) per
condition, alongside ground-truth GPS / compass / position.

Question: paper §3.2 attributes matched-compute's bottleneck to its
1×1 feature map (no spatial structure preserved).  Test directly:
probe the encoder's 512-d channel vector for GPS.  If GPS is decodable
from the encoder output alone, the encoder feature map carries
position even after spatial collapse — and the bottleneck claim
needs re-framing.

Method:
  1. Forward-hook the agent's ResNetEncoder.compression layer (the
     final 1×1 conv that produces the output the LSTM reads).
  2. Run deterministic-rollout inference for N episodes.
  3. Save (B*T, encoder_dim, h, w) features paired with GPS / compass.

Usage:
    python scripts/probing/collect_encoder_features.py \\
        --config-name=pointnav/ddppo_pointnav_matched_gibson \\
        --ckpt=/scratch/izar/wxu/habitat_checkpoints/matched_gibson/ckpt.49.pth \\
        --episodes=300 \\
        --out=/scratch/izar/wxu/probing_data/matched_gibson_encfeat_det.npz

Then probe GPS / compass from the (flattened) encoder feature vector
using analyze_encoder_features.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Make src/ importable.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-steps-per-ep", type=int, default=2000)
    args = ap.parse_args()

    import src.habitat  # noqa: F401  (registers policies)
    from src.utils.habitat_env import (
        load_habitat_config,
        load_policy,
        compute_spl,
        heading_from_quaternion,
    )

    cfg = load_habitat_config(args.config_name, split="val")
    policy, env = load_policy(cfg, args.ckpt)
    policy.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    # Hook the encoder's compression layer (the final 1x1 conv that
    # produces the output flattened into the LSTM input vector).
    enc = policy.net.visual_encoder
    compression = enc.compression  # nn.Sequential ending in conv 1x1
    captured: list[torch.Tensor] = []

    def _hook(module, _inputs, output):
        captured.append(output.detach().cpu())

    handle = compression.register_forward_hook(_hook)

    all_feats: list[np.ndarray] = []
    all_gps: list[np.ndarray] = []
    all_compass: list[np.ndarray] = []
    all_positions: list[np.ndarray] = []
    all_episode_ids: list[int] = []
    all_scene_ids: list[str] = []
    all_step_in_episode: list[int] = []

    print(f"Collecting encoder features from {args.ckpt} ({args.episodes} eps)")
    obs = env.reset()
    n_eps_done = 0
    rnn_hidden = torch.zeros(
        1, policy.net.num_recurrent_layers, policy.net.hidden_size,
        device=device,
    )
    prev_action = torch.zeros(1, 1, dtype=torch.long, device=device)
    masks = torch.zeros(1, 1, dtype=torch.bool, device=device)

    while n_eps_done < args.episodes:
        ep = env.current_episode
        scene_id = getattr(ep, "scene_id", "unknown")
        ep_id = int(getattr(ep, "episode_id", n_eps_done))

        for step in range(args.max_steps_per_ep):
            if env.episode_over:
                break
            captured.clear()
            obs_batch = {
                k: torch.as_tensor(v, device=device).unsqueeze(0)
                for k, v in obs.items()
                if not isinstance(v, dict)
            }

            with torch.no_grad():
                action_data = policy.act(
                    obs_batch, rnn_hidden, prev_action, masks,
                    deterministic=True,
                )
            action = action_data.actions
            rnn_hidden = action_data.rnn_hidden_states
            prev_action = action

            # Feature captured by hook
            if captured:
                feat = captured[0].squeeze(0).numpy()  # (C, h, w)
                all_feats.append(feat.reshape(-1))     # flatten

            # Ground-truth labels
            agent_state = env.sim.get_agent_state()
            pos = agent_state.position[[0, 2]]  # (x, z) in world frame
            heading = heading_from_quaternion(agent_state.rotation)

            all_positions.append(pos.copy())
            all_gps.append(obs["gps"].copy() if "gps" in obs else pos.copy())
            all_compass.append(np.array([float(heading)]))
            all_episode_ids.append(ep_id)
            all_scene_ids.append(str(scene_id))
            all_step_in_episode.append(step)

            obs = env.step({"action": int(action.item())})
            masks = torch.ones(1, 1, dtype=torch.bool, device=device)

        n_eps_done += 1
        if n_eps_done % 25 == 0:
            print(f"  Episode {n_eps_done}/{args.episodes}", flush=True)
        if n_eps_done < args.episodes:
            obs = env.reset()
            rnn_hidden = torch.zeros_like(rnn_hidden)
            prev_action = torch.zeros_like(prev_action)
            masks = torch.zeros_like(masks)

    handle.remove()

    feats_arr = np.stack(all_feats, axis=0).astype(np.float32)
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        encoder_features=feats_arr,
        positions=np.stack(all_positions, axis=0).astype(np.float32),
        gps=np.stack(all_gps, axis=0).astype(np.float32),
        compass=np.array(all_compass, dtype=np.float32).reshape(-1),
        episode_ids=np.array(all_episode_ids, dtype=np.int32),
        scene_ids=np.array(all_scene_ids),
        step_in_episode=np.array(all_step_in_episode, dtype=np.int32),
    )
    print(f"\nSaved {feats_arr.shape} encoder features to {out_path}")


if __name__ == "__main__":
    main()
