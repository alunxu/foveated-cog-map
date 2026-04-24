"""
Diagnostic: why does probe data for fov-fix/uniform show 0% success rate
while fov-learned shows 96.6%?

Hypothesis: collect.py uses `deterministic=False` (line 225). Fov-fix/uniform
policies may have higher policy entropy, so stochastic sampling picks STOP
action early. Fov-learned may have low entropy (hence ≈deterministic), so
it actually navigates.

What this script checks:
  1. Episode length distribution (histogram)
  2. Per-step action inferred from position deltas (forward ~0.25m vs turn ~0m)
  3. Distribution of final-step reasons (terminated via STOP vs max_steps vs goal)
  4. For comparison: same stats limited to first-N steps
"""
import numpy as np
from collections import Counter

# Habitat PointNav action IDs (assuming standard)
#   0 = STOP, 1 = MOVE_FORWARD, 2 = TURN_LEFT, 3 = TURN_RIGHT
# We can't read actions directly, but we can infer from position & heading deltas.

FWD_STEP = 0.25  # meters per forward action in Habitat PointNav
TURN_RAD = np.deg2rad(30)  # radians per turn action


def infer_action(dpos_xz, dheading):
    """Infer action from (Δx, Δz, Δheading). Heuristic."""
    dpos_mag = np.sqrt(dpos_xz[0] ** 2 + dpos_xz[1] ** 2)
    if dpos_mag > 0.1:  # moved
        return "FWD"
    if abs(dheading) > 0.1:  # turned
        return "TURN_L" if dheading > 0 else "TURN_R"
    return "STILL"  # stopped or blocked


def analyze(npz_path, name):
    d = np.load(npz_path, allow_pickle=True)
    pos = d["positions"]
    ep = d["episode_ids"]
    h = d["headings"]
    steps = d["step_in_episode"]
    d2g = d["distance_to_goal"]
    goal_pos = d["goal_positions"]

    unique_eps = np.unique(ep)
    ep_lens = []
    final_dists = []
    first_step_d2g = []
    last_step_d2g = []
    action_counts = Counter()
    episode_stopped_at = Counter()  # maps length-in-steps -> count
    moved_episodes = 0  # episodes that had any forward step
    sp1_episodes = 0   # 1-step episodes
    sp_between_1_and_5 = 0

    for e in unique_eps:
        idx = np.where(ep == e)[0]
        ep_len = len(idx)
        ep_lens.append(ep_len)
        episode_stopped_at[ep_len] += 1
        if ep_len == 1:
            sp1_episodes += 1
        if 1 < ep_len <= 5:
            sp_between_1_and_5 += 1
        p = pos[idx]
        hd = h[idx]
        g = goal_pos[idx[0]]
        first_step_d2g.append(d2g[idx[0]])
        last_step_d2g.append(d2g[idx[-1]])
        final_dists.append(np.linalg.norm(p[-1] - g))
        any_fwd = False
        for t in range(1, ep_len):
            dpos = p[t] - p[t - 1]
            # Habitat world: agent moves in xz plane; y is height
            dpos_xz = np.array([dpos[0], dpos[2]])
            dheading = hd[t] - hd[t - 1]
            dheading = np.arctan2(np.sin(dheading), np.cos(dheading))  # wrap
            act = infer_action(dpos_xz, dheading)
            action_counts[act] += 1
            if act == "FWD":
                any_fwd = True
        if any_fwd:
            moved_episodes += 1

    ep_lens = np.array(ep_lens)
    final_dists = np.array(final_dists)
    first_step_d2g = np.array(first_step_d2g)

    print(f"\n=== {name} (n_ep={len(unique_eps)}, n_steps={len(pos)}) ===")
    print(f"  Ep len:  mean {ep_lens.mean():.1f}, median {np.median(ep_lens):.0f}, "
          f"min {ep_lens.min()}, max {ep_lens.max()}, std {ep_lens.std():.1f}")

    # Histogram of episode lengths
    print(f"  Ep len distribution:")
    bins = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 50), (51, 200), (201, 500), (501, 9999)]
    for lo, hi in bins:
        c = ((ep_lens >= lo) & (ep_lens <= hi)).sum()
        frac = c / len(ep_lens)
        if c > 0:
            bar = "#" * int(40 * frac)
            print(f"    [{lo:3d}-{hi:4d}]  {c:4d} ({100*frac:5.1f}%) {bar}")

    print(f"  Start d2g: mean {first_step_d2g.mean():.2f} m (initial distance to goal)")
    print(f"  Final d2g: mean {final_dists.mean():.2f} m")
    print(f"  Success (<0.2m): {100*(final_dists<0.2).mean():.1f}%")
    print(f"  Ep with any FWD action: {100*moved_episodes/len(unique_eps):.1f}%")
    print(f"  1-step episodes (instant STOP): {sp1_episodes} ({100*sp1_episodes/len(unique_eps):.1f}%)")
    print(f"  2-5 step episodes: {sp_between_1_and_5} ({100*sp_between_1_and_5/len(unique_eps):.1f}%)")

    # Action inference summary
    total_infer = sum(action_counts.values())
    print(f"  Inferred actions (from step deltas, N={total_infer}):")
    for act in ["FWD", "TURN_L", "TURN_R", "STILL"]:
        c = action_counts[act]
        frac = c / total_infer if total_infer else 0
        print(f"    {act:8s}: {c:6d} ({100*frac:5.1f}%)")


for name, path in [
    ("foveated-fix", "/scratch/izar/wxu/probing_data/foveated_gibson.npz"),
    ("foveated-learned", "/scratch/izar/wxu/probing_data/foveated_learned_gibson.npz"),
    ("uniform", "/scratch/izar/wxu/probing_data/uniform_gibson.npz"),
    ("blind", "/scratch/izar/wxu/probing_data/blind_gibson.npz"),
    ("matched", "/scratch/izar/wxu/probing_data/matched_gibson.npz"),
]:
    try:
        analyze(path, name)
    except FileNotFoundError:
        print(f"SKIP {name}: file not found")
    except Exception as e:
        print(f"ERROR {name}: {type(e).__name__}: {e}")

print("\n" + "=" * 70)
print("INTERPRETATION:")
print("=" * 70)
print("""
If fov-fix/uniform show:
  - mostly 1-5 step episodes (>80%)
  - <20% of episodes have any FWD action
  - high frac of STILL actions
Then hypothesis confirmed: deterministic=False causes early STOP sampling
for these policies. Fix: re-collect with deterministic=True.

If fov-fix/uniform show:
  - broad length distribution, many 50+ step episodes
  - normal FWD fraction (~60%)
  - but still fail to reach goal
Then the issue is different (e.g., policy does navigate but wanders, or
config has wrong max_steps, or wrong success radius).
""")
