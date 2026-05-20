"""
collect_from_episodes.py
========================
Collects LSTM hidden states from real Gibson PointNav episodes.
reads episode geometry directly from the JSON episode files and feeds real GPS/compass
observations through the reconstructed LSTM.

Usage
-----
  python collect_from_episodes.py --mode late              # converged ckpts only
  python collect_from_episodes.py --mode all               # full training sweep
  python collect_from_episodes.py --folder foveated --ckpt 49 --episodes 5

Requirements
------------
  pip install torch huggingface_hub numpy scipy
  src/ folder from foveated-cog-map repo on PYTHONPATH

Data
----
  Gibson PointNav v1 episodes: ~/data/datasets/pointnav/gibson/v1/val/val.json.gz
"""

import sys
import math
import gzip
import json
import pickle
import zipfile
import types
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path.home()))

import omegaconf
import omegaconf.dictconfig
import habitat         
import habitat_baselines  

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ID      = "alunxu/spatial-memory-checkpoints"
EPISODE_FILE = Path.home() / "data/datasets/pointnav/gibson/v1/val/val.json.gz"
OUTPUT_DIR   = Path("activations_real")
OUTPUT_DIR.mkdir(exist_ok=True)

HIDDEN_SIZE     = 512
NUM_LSTM_LAYERS = 3
EMBED_DIM       = 32
NUM_ACTIONS     = 4
MAX_STEPS       = 200
N_DIST_BINS     = 6
N_ANGLE_BINS    = 8

LATE_CKPT = {
    "blind": 34, "coarse": 49,
    "foveated": 49, "foveated_logpolar": 49, "uniform": 49,
}

ALL_CKPTS = {
    "blind":             [10, 20, 30, 34],
    "coarse":            [20, 30, 40, 49],
    "foveated":          [20, 30, 40, 49],
    "foveated_logpolar": [20, 30, 40, 49],
    "uniform":           [20, 30, 40, 49],
}

# ---------------------------------------------------------------------------
# Checkpoint loader (stubs broken habitat imports during unpickling)
# ---------------------------------------------------------------------------

def _safe_load(path: str) -> dict:
    """Load a .pth checkpoint with full tensor reconstruction.

    Attempts plain torch.load first; falls back to a custom unpickler
    that stubs unresolvable classes (e.g. old habitat config classes)
    while correctly reconstructing all tensor storage blobs.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        pass

    _DTYPE_MAP = {
        "FloatStorage":    (torch.float32, 4),
        "DoubleStorage":   (torch.float64, 8),
        "HalfStorage":     (torch.float16, 2),
        "BFloat16Storage": (torch.bfloat16, 2),
        "LongStorage":     (torch.int64,   8),
        "IntStorage":      (torch.int32,   4),
        "ShortStorage":    (torch.int16,   2),
        "ByteStorage":     (torch.uint8,   1),
        "BoolStorage":     (torch.bool,    1),
    }

    with zipfile.ZipFile(path) as zf:
        names    = zf.namelist()
        pkl_name = next(n for n in names if n.endswith("/data.pkl"))
        prefix   = pkl_name[: pkl_name.rfind("/data.pkl") + 1]
        blobs    = {n.split("/")[-1]: zf.read(n)
                    for n in names if n.startswith(prefix + "data/")}
        cache    = {}

        class _Unpickler(pickle.Unpickler):
            class _Stub:
                def __init__(self, *a, **kw): pass
                def __setstate__(self, s): pass
                def __getattr__(self, n): return self.__class__()
                def __iter__(self): return iter([])

            def find_class(self, module, name):
                try:
                    return super().find_class(module, name)
                except Exception:
                    return self._Stub

            def persistent_load(self, pid):
                if not isinstance(pid, tuple) or pid[0] != "storage":
                    return pid
                _, storage_type, idx, _, numel = pid
                if idx in cache:
                    return cache[idx]
                type_name = getattr(storage_type, "__name__", "FloatStorage")
                dtype, _  = _DTYPE_MAP.get(type_name, (torch.float32, 4))
                blob      = blobs.get(str(idx), b"")
                arr = (torch.frombuffer(bytearray(blob), dtype=dtype)
                       if blob else torch.zeros(numel, dtype=dtype))
                cache[idx] = arr.storage()
                return cache[idx]

        with zf.open(pkl_name) as f:
            return _Unpickler(f).load()


# ---------------------------------------------------------------------------
# Minimal LSTM policy (reconstructed from state_dict weights)
# ---------------------------------------------------------------------------

class MinimalPolicy(nn.Module):
    """Reconstructs the sensor-embedding + LSTM from a state_dict.

    Visual features are zeroed out (sighted agents) so that hidden states
    reflect what the LSTM encodes from GPS/compass/goal alone.
    This isolates memory content from visual perception for RSA.
    """

    def __init__(self, state_dict: dict, is_blind: bool):
        super().__init__()
        self.is_blind = is_blind
        self.g_emb       = nn.Linear(2, EMBED_DIM)
        self.gps_emb     = nn.Linear(2, EMBED_DIM)
        self.compass_emb = nn.Linear(2, EMBED_DIM)
        self.close_emb   = nn.Linear(1, EMBED_DIM)
        self.action_emb  = nn.Embedding(NUM_ACTIONS + 1, EMBED_DIM)
        lstm_input = 5 * EMBED_DIM if is_blind else HIDDEN_SIZE + 5 * EMBED_DIM
        self.lstm = nn.LSTM(lstm_input, HIDDEN_SIZE, NUM_LSTM_LAYERS,
                            batch_first=False)
        self._load_weights(state_dict)
        self.eval()

    def _load_weights(self, sd: dict):
        def _load(module, prefix):
            sub = {k[len(prefix):]: v for k, v in sd.items()
                   if k.startswith(prefix)}
            if sub:
                module.load_state_dict(sub, strict=False)
        _load(self.g_emb,       "net.g_embedding.")
        _load(self.gps_emb,     "net.gps_embedding.")
        _load(self.compass_emb, "net.compass_embedding.")
        _load(self.close_emb,   "net.close_embedding.")
        _load(self.action_emb,  "net.prev_action_embedding.")
        lstm_sd = {k[len("net.state_encoder.rnn."):]: v
                   for k, v in sd.items()
                   if k.startswith("net.state_encoder.rnn.")}
        if lstm_sd:
            self.lstm.load_state_dict(lstm_sd, strict=False)

    def step(self, g, gps, compass, close, prev_action, hx, cx):
        """Single LSTM step. Returns (output, hx, cx)."""
        compass_cs = torch.cat([torch.sin(compass), torch.cos(compass)], -1)
        parts = [
            self.g_emb(g),
            self.gps_emb(gps),
            self.compass_emb(compass_cs),
            self.close_emb(close),
            self.action_emb(prev_action),
        ]
        if not self.is_blind:
            parts.insert(0, torch.zeros(g.shape[0], HIDDEN_SIZE))
        x = torch.cat(parts, -1).unsqueeze(0)   # (1, B, input_size)
        out, (hx, cx) = self.lstm(x, (hx, cx))
        return out.squeeze(0), hx, cx           # (B, hidden_size)


# ---------------------------------------------------------------------------
# Episode loading and geometry
# ---------------------------------------------------------------------------

def load_episodes(n: int = 200) -> list:
    """Load the first n episodes from the Gibson val set."""
    with gzip.open(EPISODE_FILE, "rt") as f:
        data = json.load(f)
    return data["episodes"][:n]


def episode_to_trajectory(ep: dict, rng: np.random.RandomState) -> list:
    """Convert a Gibson episode dict into a list of observation dicts.

    Simulates greedy navigation toward the goal, computing GPS and compass
    at each step from the episode geometry (no renderer needed).

    Returns a list of dicts with keys:
        g        : (2,) goal in start frame (fixed throughout episode)
        gps      : (2,) current goal vector in agent frame
        compass  : (1,) current heading (radians)
        close    : float, min(dist_to_goal, 0.5)
        dist     : float, distance to goal
        ecc      : float, |angle to goal| in agent frame (eccentricity proxy)
    """
    start   = ep["start_position"]           # [x, y, z]
    goal    = ep["goals"][0]["position"]      # [x, y, z]
    start_xz = np.array([start[0], start[2]])
    goal_xz  = np.array([goal[0],  goal[2]])

    # Extract yaw from start quaternion [x, y, z, w]
    q = ep.get("start_rotation", [0, 0, 0, 1])
    heading = 2.0 * math.atan2(q[1], q[3])

    g_start = goal_xz - start_xz   # goal in start frame (fixed)
    pos     = start_xz.copy()
    steps   = []

    for _ in range(MAX_STEPS):
        rel = goal_xz - pos
        dist = float(np.linalg.norm(rel))
        if dist < 0.2:
            break

        # Rotate goal vector into agent frame
        c, s  = math.cos(-heading), math.sin(-heading)
        gps_x =  c * rel[0] + s * rel[1]
        gps_z = -s * rel[0] + c * rel[1]
        ecc   = abs(math.atan2(gps_z, gps_x))

        steps.append({
            "g":       g_start,
            "gps":     np.array([gps_x, gps_z]),
            "compass": np.array([heading]),
            "close":   min(dist, 0.5),
            "dist":    dist,
            "ecc":     ecc,
        })

        # Greedy step: turn toward goal then move forward
        direction = rel / (dist + 1e-8)
        turn = math.atan2(direction[1], direction[0]) - heading
        turn = (turn + math.pi) % (2 * math.pi) - math.pi
        heading += float(np.clip(turn, -0.2618, 0.2618))
        pos += 0.25 * np.array([math.cos(heading), math.sin(heading)])

    return steps


def bin_condition(dist: float, ecc: float) -> int:
    """Map (distance, eccentricity) to a condition index (0..47)."""
    d = min(int(dist / 2.0), N_DIST_BINS - 1)
    a = min(int(ecc / (math.pi / N_ANGLE_BINS)), N_ANGLE_BINS - 1)
    return d * N_ANGLE_BINS + a


# ---------------------------------------------------------------------------
# Main collection routine
# ---------------------------------------------------------------------------

def collect(folder: str, ckpt_idx: int, n_episodes: int = 200) -> dict:
    """Collect hidden states for one checkpoint.

    Downloads the checkpoint, reconstructs the LSTM, runs n_episodes
    real Gibson episodes, and returns arrays of hidden states + metadata.
    """
    print(f"\n{'='*60}")
    print(f"  Collecting: {folder}/ckpt.{ckpt_idx}.pth  ({n_episodes} episodes)")
    print(f"{'='*60}")

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"{folder}/ckpt.{ckpt_idx}.pth",
    )
    ckpt = _safe_load(path)
    sd   = ckpt["state_dict"]

    is_blind = not any("visual_encoder" in k for k in sd)
    print(f"  Architecture: {'blind' if is_blind else 'sighted'}")

    model    = MinimalPolicy(sd, is_blind)
    episodes = load_episodes(n_episodes)
    rng      = np.random.RandomState(42)

    all_hidden = []
    all_cond   = []
    all_ecc    = []
    all_dist   = []

    for ep_idx, ep in enumerate(episodes):
        traj = episode_to_trajectory(ep, rng)
        if not traj:
            continue

        hx          = torch.zeros(NUM_LSTM_LAYERS, 1, HIDDEN_SIZE)
        cx          = torch.zeros(NUM_LSTM_LAYERS, 1, HIDDEN_SIZE)
        prev_action = torch.tensor([0])

        with torch.no_grad():
            for step in traj:
                g       = torch.tensor(step["g"][None],       dtype=torch.float32)
                gps     = torch.tensor(step["gps"][None],     dtype=torch.float32)
                compass = torch.tensor(step["compass"][None], dtype=torch.float32)
                close   = torch.tensor([[step["close"]]],     dtype=torch.float32)

                out, hx, cx = model.step(g, gps, compass, close, prev_action,
                                         hx, cx)
                prev_action = torch.tensor([rng.randint(1, NUM_ACTIONS + 1)])

                all_hidden.append(out.squeeze().numpy())
                all_cond.append(bin_condition(step["dist"], step["ecc"]))
                all_ecc.append(step["ecc"])
                all_dist.append(step["dist"])

        if (ep_idx + 1) % 50 == 0:
            print(f"  Episode {ep_idx + 1}/{n_episodes} done")

    print(f"  Collected {len(all_hidden)} timesteps")

    return {
        "hidden":       np.array(all_hidden, dtype=np.float32),
        "conditions":   np.array(all_cond,   dtype=np.int32),
        "eccentricity": np.array(all_ecc,    dtype=np.float32),
        "dist_to_goal": np.array(all_dist,   dtype=np.float32),
        "folder":       np.array([folder]),
        "ckpt_idx":     np.array([ckpt_idx]),
    }


def collect_all(checkpoints: dict, n_episodes: int):
    for folder, indices in checkpoints.items():
        for idx in indices:
            out = OUTPUT_DIR / f"{folder}_ckpt{idx:02d}.npz"
            if out.exists():
                print(f"  Skipping {out} (already exists)")
                continue
            data = collect(folder, idx, n_episodes)
            np.savez(out, **data)
            print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect LSTM hidden states from Gibson PointNav episodes."
    )
    parser.add_argument("--mode",     default="late", choices=["late", "all"],
                        help="'late' = converged checkpoints only; 'all' = full sweep")
    parser.add_argument("--folder",   default=None,
                        help="Single folder to collect (e.g. foveated)")
    parser.add_argument("--ckpt",     default=None, type=int,
                        help="Single checkpoint index (used with --folder)")
    parser.add_argument("--episodes", default=200, type=int,
                        help="Number of episodes per checkpoint")
    args = parser.parse_args()

    if args.folder and args.ckpt is not None:
        data = collect(args.folder, args.ckpt, args.episodes)
        out  = OUTPUT_DIR / f"{args.folder}_ckpt{args.ckpt:02d}.npz"
        np.savez(out, **data)
        print(f"Saved → {out}")
    elif args.mode == "late":
        print("Collecting late (converged) checkpoints only.")
        collect_all({f: [i] for f, i in LATE_CKPT.items()}, args.episodes)
    else:
        print("Collecting all checkpoints (full training dynamics sweep).")
        collect_all(ALL_CKPTS, args.episodes)