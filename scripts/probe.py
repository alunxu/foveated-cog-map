"""
CS503 Project — Probing Analysis

Trains linear probes on frozen GRU hidden states to test for
emergent spatial representations (maps) in the agent's memory.

Three probes:
  1. Occupancy grid decoder — does memory encode the map layout?
  2. Agent position decoder — does memory encode where the agent is?
  3. Collision detector — are there collision-detection neurons?

Usage:
    python scripts/probe.py --checkpoint outputs/blind_agent/checkpoint_final.pt \
                            --config cfgs/blind.yaml \
                            --n_episodes 200 \
                            --output probing_results/blind/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.nav_env import NavigationEnv
from src.envs.wrappers import (
    PointGoalWrapper, BlindWrapper, FoveatedWrapper,
    UniformWrapper, MatchedComputeWrapper, N_ACTIONS,
)
from src.models import NavigationAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Probing analysis on agent memory")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--output", type=str, default="probing_results/")
    return parser.parse_args()


def load_config(config_path, overrides=None):
    cfg = OmegaConf.load(config_path)
    if "defaults" in cfg:
        base_path = Path(config_path).parent
        for default in cfg.defaults:
            base_cfg = OmegaConf.load(base_path / f"{default}.yaml")
            cfg = OmegaConf.merge(base_cfg, cfg)
        del cfg["defaults"]
    return cfg


def make_env(cfg, seed_offset=0):
    env = NavigationEnv(
        env_id=cfg.env.id,
        image_size=cfg.env.image_size,
        max_steps=cfg.env.max_steps,
        seed=cfg.get("seed", 42) + seed_offset,
    )
    env = PointGoalWrapper(
        env,
        success_reward=cfg.env.success_reward,
        time_penalty=cfg.env.time_penalty,
    )
    encoder_type = cfg.model.encoder.get("type", "cnn")
    if encoder_type == "vector":
        env = BlindWrapper(env, pointgoal_dim=cfg.env.pointgoal_dim)
    elif cfg.foveation.enabled:
        env = FoveatedWrapper(
            env,
            fovea_radius=cfg.foveation.fovea_radius,
            blur_sigma_max=cfg.foveation.blur_sigma_max,
            falloff=cfg.foveation.falloff,
        )
    elif cfg.env.get("image_size", 64) < 64:
        env = MatchedComputeWrapper(env, target_size=cfg.env.image_size)
    else:
        env = UniformWrapper(env)
    return env


def preprocess_obs(obs, encoder_type, device):
    if encoder_type == "vector":
        return torch.from_numpy(obs[np.newaxis]).float().to(device)
    else:
        return torch.from_numpy(obs[np.newaxis]).float().permute(0, 3, 1, 2).to(device) / 255.0


def collect_probing_data(agent, cfg, device, n_episodes):
    """Run agent and collect (hidden_state, ground_truth) pairs."""
    encoder_type = cfg.model.encoder.get("type", "cnn")
    is_foveated = cfg.foveation.get("gaze_action", False)

    all_hidden = []
    all_positions = []
    all_directions = []
    all_collisions = []
    all_occupancy = []

    for ep in range(n_episodes):
        env = make_env(cfg, seed_offset=ep + 5000)
        obs, info = env.reset()
        hidden = None
        prev_action = torch.zeros(1, dtype=torch.long, device=device)
        done = False

        while not done:
            obs_t = preprocess_obs(obs, encoder_type, device)
            pg = torch.from_numpy(info["pointgoal"][np.newaxis]).float().to(device)

            with torch.no_grad():
                action, gaze, _, _, new_hidden = agent.act(
                    obs_t, hidden, pointgoal=pg, prev_action=prev_action,
                )

            # Store hidden state (the GRU output, not hidden -- we want the memory content)
            # new_hidden is (num_layers, 1, hidden_size), take the last layer
            h_np = new_hidden[-1, 0].cpu().numpy()
            all_hidden.append(h_np)
            all_positions.append(list(info["agent_pos"]))
            all_directions.append(info["agent_dir"])

            act_int = action.item()
            if is_foveated and gaze is not None:
                gaze_np = gaze.cpu().numpy().squeeze()
                env_action = np.array([float(act_int), gaze_np[0], gaze_np[1]], dtype=np.float32)
            else:
                env_action = act_int

            obs, reward, terminated, truncated, info = env.step(env_action)
            all_collisions.append(float(info["collision"]))

            # Store occupancy grid (same for entire episode, but we store per-step for alignment)
            gt = info["ground_truth"]
            all_occupancy.append(gt["occupancy_grid"].flatten())

            hidden = new_hidden
            prev_action = action.unsqueeze(0) if action.dim() == 0 else action
            done = terminated or truncated

        env.close()

        if (ep + 1) % 50 == 0:
            logger.info(f"  Collected {ep + 1}/{n_episodes} episodes ({len(all_hidden)} steps)")

    return {
        "hidden_states": np.array(all_hidden),
        "positions": np.array(all_positions, dtype=np.float32),
        "directions": np.array(all_directions, dtype=np.int64),
        "collisions": np.array(all_collisions, dtype=np.float32),
        "occupancy": np.array(all_occupancy, dtype=np.float32),
    }


def probe_occupancy(data, max_samples=50000):
    """Train a linear probe to decode occupancy grid from hidden states."""
    logger.info("  Probe 1: Occupancy Grid Decoding")

    X = data["hidden_states"]
    Y = data["occupancy"]

    # Subsample if too large
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X, Y = X[idx], Y[idx]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_cells = Y.shape[1]
    accuracies = []
    f1s = []

    for cell_idx in range(n_cells):
        y_tr = Y_train[:, cell_idx]
        y_te = Y_test[:, cell_idx]

        # Skip cells that are all-same (no variance)
        if len(np.unique(y_tr)) < 2:
            continue

        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        clf.fit(X_train, y_tr)

        pred = clf.predict(X_test)
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, zero_division=0)
        accuracies.append(acc)
        f1s.append(f1)

    results = {
        "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "n_cells_probed": len(accuracies),
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test),
    }
    logger.info(f"    Accuracy: {results['mean_accuracy']:.4f} | F1: {results['mean_f1']:.4f} | Cells: {results['n_cells_probed']}")
    return results


def probe_position(data, max_samples=50000):
    """Train a linear probe to decode agent position from hidden states."""
    logger.info("  Probe 2: Agent Position Decoding")

    X = data["hidden_states"]
    Y = data["positions"]

    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X, Y = X[idx], Y[idx]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    reg = Ridge(alpha=1.0)
    reg.fit(X_train, Y_train)

    pred = reg.predict(X_test)
    mae = np.mean(np.abs(pred - Y_test))
    r2 = r2_score(Y_test, pred)

    results = {
        "mae_grid_cells": float(mae),
        "r2": float(r2),
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test),
    }
    logger.info(f"    MAE: {results['mae_grid_cells']:.3f} grid cells | R2: {results['r2']:.4f}")
    return results


def probe_collision(data, max_samples=50000):
    """Train a linear probe to detect collisions from hidden states."""
    logger.info("  Probe 3: Collision Detection")

    X = data["hidden_states"]
    Y = data["collisions"]

    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X, Y = X[idx], Y[idx]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Full probe
    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
    clf.fit(X_train_s, Y_train)
    pred = clf.predict(X_test_s)
    pred_prob = clf.predict_proba(X_test_s)

    acc = accuracy_score(Y_test, pred)
    collision_rate = float(np.mean(Y))
    chance = max(collision_rate, 1 - collision_rate)

    try:
        auc = roc_auc_score(Y_test, pred_prob[:, 1])
    except (ValueError, IndexError):
        auc = 0.5

    # Per-neuron probe: find most predictive individual neurons
    n_neurons = X.shape[1]
    neuron_accs = []
    for neuron_idx in range(n_neurons):
        x_tr = X_train_s[:, neuron_idx:neuron_idx+1]
        x_te = X_test_s[:, neuron_idx:neuron_idx+1]
        clf_n = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs")
        try:
            clf_n.fit(x_tr, Y_train)
            neuron_accs.append(accuracy_score(Y_test, clf_n.predict(x_te)))
        except Exception:
            neuron_accs.append(chance)

    neuron_accs = np.array(neuron_accs)
    top_10_idx = np.argsort(neuron_accs)[-10:][::-1]
    top_10_accs = neuron_accs[top_10_idx]

    # Probe with only top-10 neurons
    X_train_top = X_train_s[:, top_10_idx]
    X_test_top = X_test_s[:, top_10_idx]
    clf_top = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
    clf_top.fit(X_train_top, Y_train)
    top10_acc = accuracy_score(Y_test, clf_top.predict(X_test_top))

    results = {
        "full_accuracy": float(acc),
        "full_auroc": float(auc),
        "chance_accuracy": float(chance),
        "collision_rate": float(collision_rate),
        "top10_neurons": top_10_idx.tolist(),
        "top10_neuron_accs": top_10_accs.tolist(),
        "top10_combined_accuracy": float(top10_acc),
        "n_neurons": n_neurons,
    }
    logger.info(f"    Full: {acc:.4f} | AUROC: {auc:.4f} | Top-10: {top10_acc:.4f} | Chance: {chance:.4f}")
    return results


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_type = cfg.model.encoder.get("type", "cnn")
    is_foveated = cfg.foveation.get("gaze_action", False)
    n_actions = cfg.env.get("n_actions", N_ACTIONS)

    # Load agent
    agent = NavigationAgent(
        encoder_type=encoder_type,
        image_size=cfg.env.image_size,
        encoder_channels=list(cfg.model.encoder.get("channels", [16, 32, 64])),
        hidden_size=cfg.model.memory.hidden_size,
        num_memory_layers=cfg.model.memory.get("num_layers", 1),
        n_actions=n_actions,
        gaze_enabled=is_foveated,
        pointgoal_dim=cfg.env.get("pointgoal_dim", 4),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()

    logger.info(f"Loaded checkpoint: {args.checkpoint} ({ckpt['timesteps']:,} steps)")

    # Collect probing data
    logger.info(f"Collecting probing data ({args.n_episodes} episodes)...")
    data = collect_probing_data(agent, cfg, device, args.n_episodes)
    logger.info(f"  Total samples: {len(data['hidden_states'])}")
    logger.info(f"  Hidden dim: {data['hidden_states'].shape[1]}")
    logger.info(f"  Occupancy grid dim: {data['occupancy'].shape[1]}")
    logger.info(f"  Collision rate: {data['collisions'].mean():.3f}")

    # Run probes
    logger.info("\nRunning probes...")
    results = {}
    results["occupancy"] = probe_occupancy(data)
    results["position"] = probe_position(data)
    results["collision"] = probe_collision(data)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "probe_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Save raw data for further analysis
    np.savez_compressed(
        output_dir / "probe_data.npz",
        hidden_states=data["hidden_states"],
        positions=data["positions"],
        directions=data["directions"],
        collisions=data["collisions"],
        occupancy=data["occupancy"],
    )
    logger.info(f"Raw data saved to {output_dir / 'probe_data.npz'}")


if __name__ == "__main__":
    main()
