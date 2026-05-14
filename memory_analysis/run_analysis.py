"""
run_analysis.py
===============
Orchestrates the full RSA pipeline:

  Step 1. Load pre-collected activations from activations/
  Step 2. Build WUC-corrected RDMs for each agent
  Step 3. H1 — eccentricity test: compare low-ecc vs high-ecc RDMs
           (does the foveated agent represent peripheral positions
            more distinctly in memory than the uniform agent?)
  Step 4. H2 — cross-agent comparison: do different agents encode
           the same spatial information in structurally different ways?
  Step 5. Training dynamics: how do RDMs evolve from init → convergence?
  Step 6. Save results + figures to results/

Usage
-----
  # First collect activations (run once):
  python collect_activations.py late

  # Then run the analysis:
  python run_analysis.py
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works without a display)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

from rsa import (
    build_rdm,
    eccentricity_rdms,
    cross_agent_similarity,
    kendall_tau_a_fast,
    permutation_test_rdm_similarity,
    rdm_stats,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ACT_DIR     = Path("activations")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Agents present in the late-checkpoint activations
AGENTS = ["blind", "coarse", "foveated", "foveated_logpolar", "uniform"]

# Display names for plots
DISPLAY_NAMES = {
    "blind":            "Blind",
    "coarse":           "Coarse",
    "foveated":         "Foveated",
    "foveated_logpolar":"Fov-LogPolar",
    "uniform":          "Uniform",
}

# Agent colours (consistent across all plots)
COLOURS = {
    "blind":            "#555555",
    "coarse":           "#4477AA",
    "foveated":         "#EE6677",
    "foveated_logpolar":"#AA3377",
    "uniform":          "#228833",
}

# Late checkpoint index per folder (must match collect_activations.py)
LATE_CKPT = {
    "blind": 34, "coarse": 49,
    "foveated": 49, "foveated_logpolar": 49, "uniform": 49,
}

# Frames per checkpoint index per folder
FRAMES_PER_CKPT = {
    "blind": 10.06e6, "coarse": 5.0e6,
    "foveated": 5.0e6, "foveated_logpolar": 5.0e6, "uniform": 5.0e6,
}

N_PERMS = 500   # permutation test iterations (increase to 2000 for final paper)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_activations(agent: str, ckpt_idx: int) -> Optional[dict]:
    """Load a saved .npz activation file. Returns None if not found."""
    path = ACT_DIR / f"{agent}_ckpt{ckpt_idx:02d}.npz"
    if not path.exists():
        print(f"  [WARN] {path} not found — skipping.")
        return None
    d = dict(np.load(path, allow_pickle=True))
    # Scalar arrays stored as 0-d — extract value
    for key in ("is_blind", "folder", "ckpt_idx"):
        if key in d and d[key].ndim == 1:
            d[key] = d[key][0]
    return d


def load_all_late() -> Dict[str, dict]:
    """Load late-checkpoint activations for all agents."""
    data = {}
    for agent in AGENTS:
        idx = LATE_CKPT[agent]
        d = load_activations(agent, idx)
        if d is not None:
            data[agent] = d
    if not data:
        raise FileNotFoundError(
            "No activation files found in activations/.\n"
            "Run:  python collect_activations.py late"
        )
    return data


# ---------------------------------------------------------------------------
# Step 2: Build RDMs for all agents
# ---------------------------------------------------------------------------

def build_all_rdms(data: Dict[str, dict], correct_noise: bool = True) -> Dict[str, np.ndarray]:
    """Build WUC-corrected RDMs for all agents."""
    rdms = {}
    for agent, d in data.items():
        print(f"  Building RDM for {agent} ...")
        rdm, cond_ids = build_rdm(
            d["hidden"], d["conditions"], correct_noise=correct_noise
        )
        rdms[agent] = rdm
        stats = rdm_stats(rdm)
        print(f"    {agent}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
              f"n_pairs={stats['n_pairs']}")
    return rdms


# ---------------------------------------------------------------------------
# Step 3: H1 — eccentricity test
# ---------------------------------------------------------------------------

def h1_eccentricity_test(data: Dict[str, dict]) -> dict:
    """Test H1: foveated agent should show larger low-ecc vs high-ecc RDM shift.

    For each agent, computes:
      - RDM for low-eccentricity timesteps
      - RDM for high-eccentricity timesteps
      - τ_a similarity between the two RDMs
      - Difference in mean RDM distance (low vs high)

    A high τ_a (similar structure across eccentricity bins) suggests the
    agent encodes space the same way regardless of where the goal is.
    A low τ_a (different structure) suggests eccentricity modulates memory.

    Under H1, the foveated agent should show a LOWER τ_a similarity
    (and/or larger mean distance in the high-ecc condition) compared
    to the uniform agent.
    """
    results = {}

    for agent, d in data.items():
        print(f"  H1 test for {agent} ...")
        threshold = float(np.median(d["eccentricity"]))

        rdm_low, rdm_high = eccentricity_rdms(
            d["hidden"], d["conditions"], d["eccentricity"],
            threshold=threshold, correct_noise=True,
        )

        # Similarity between low- and high-eccentricity RDMs
        tau, p_val = permutation_test_rdm_similarity(
            rdm_low, rdm_high, n_perms=N_PERMS
        )

        stats_low  = rdm_stats(rdm_low)
        stats_high = rdm_stats(rdm_high)

        results[agent] = {
            "rdm_low":  rdm_low,
            "rdm_high": rdm_high,
            "tau_low_vs_high": tau,
            "p_low_vs_high":   p_val,
            "mean_low":  stats_low["mean"],
            "mean_high": stats_high["mean"],
            "ecc_threshold": threshold,
        }
        print(f"    τ_a(low,high) = {tau:.3f}, p = {p_val:.3f}  "
              f"[mean_low={stats_low['mean']:.4f}, mean_high={stats_high['mean']:.4f}]")

    return results


# ---------------------------------------------------------------------------
# Step 4: H2 — cross-agent RDM similarity
# ---------------------------------------------------------------------------

def h2_cross_agent(rdms: Dict[str, np.ndarray]) -> dict:
    """Test H2: do foveated and uniform agents have different memory geometry?

    Computes the full pairwise τ_a similarity matrix and runs permutation
    tests on the key comparison: foveated vs. uniform.
    """
    sim_matrix, agent_names = cross_agent_similarity(rdms)

    print("\n  Cross-agent RDM similarity (Kendall τ_a):")
    header = "           " + "  ".join(f"{DISPLAY_NAMES.get(a, a):>12s}" for a in agent_names)
    print(header)
    for i, ni in enumerate(agent_names):
        row = f"  {DISPLAY_NAMES.get(ni, ni):>10s}"
        for j in range(len(agent_names)):
            val = sim_matrix[i, j]
            if np.isnan(val):
                row += "          nan"
            else:
                row += f"  {val:>12.3f}"
        print(row)

    # Key comparison: foveated vs uniform
    key_pairs = [
        ("foveated", "uniform"),
        ("foveated", "blind"),
        ("uniform",  "blind"),
        ("foveated", "foveated_logpolar"),
        ("coarse",   "uniform"),
    ]
    pairwise_tests = {}
    for a1, a2 in key_pairs:
        if a1 in rdms and a2 in rdms:
            tau, p = permutation_test_rdm_similarity(rdms[a1], rdms[a2], n_perms=N_PERMS)
            pairwise_tests[f"{a1}_vs_{a2}"] = {"tau": tau, "p": p}
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            print(f"  {a1} vs {a2}: τ_a={tau:.3f}, p={p:.3f} {sig}")

    return {
        "sim_matrix":    sim_matrix,
        "agent_names":   agent_names,
        "pairwise_tests": pairwise_tests,
    }


# ---------------------------------------------------------------------------
# Step 5: Training dynamics
# ---------------------------------------------------------------------------

def training_dynamics(agents: List[str] = None) -> dict:
    """Track how RDM structure evolves from init to convergence.

    For each agent, loads all available checkpoints and computes:
      - The τ_a similarity of each RDM to the final (late) RDM
      - The mean RDM distance as a function of frames
    """
    if agents is None:
        agents = ["foveated", "uniform", "blind"]

    from collect_activations import CHECKPOINTS

    dynamics = {}
    for agent in agents:
        if agent not in CHECKPOINTS:
            continue
        agent_data = []
        for idx in sorted(CHECKPOINTS[agent]):
            d = load_activations(agent, idx)
            if d is None:
                continue
            rdm, _ = build_rdm(d["hidden"], d["conditions"], correct_noise=True)
            frames = FRAMES_PER_CKPT[agent] * idx
            stats = rdm_stats(rdm)
            agent_data.append({
                "ckpt_idx": idx,
                "frames": frames,
                "rdm": rdm,
                "mean_dist": stats["mean"],
            })
        if agent_data:
            # Similarity of each RDM to the final RDM
            final_rdm = agent_data[-1]["rdm"]
            for entry in agent_data:
                tau = kendall_tau_a_fast(entry["rdm"], final_rdm)
                entry["tau_to_final"] = tau
            dynamics[agent] = agent_data

    return dynamics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rdm_grid(rdms: Dict[str, np.ndarray], title: str, fname: str):
    """Plot all agent RDMs in a grid."""
    n = len(rdms)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    # Shared colour scale
    all_vals = np.concatenate([
        rdm[np.triu_indices(rdm.shape[0], k=1)] for rdm in rdms.values()
    ])
    all_vals = all_vals[np.isfinite(all_vals)]
    vmin, vmax = np.percentile(all_vals, [2, 98])

    for ax, (agent, rdm) in zip(axes, rdms.items()):
        im = ax.imshow(rdm, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(DISPLAY_NAMES.get(agent, agent), fontsize=11, fontweight="bold",
                     color=COLOURS.get(agent, "black"))
        ax.set_xlabel("Condition index")
        ax.set_ylabel("Condition index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="WUC distance")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    path = RESULTS_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_h1_eccentricity(h1_results: dict, fname: str = "h1_eccentricity.png"):
    """Bar chart: mean RDM distance for low vs high eccentricity per agent."""
    agents   = [a for a in AGENTS if a in h1_results]
    n        = len(agents)
    x        = np.arange(n)
    width    = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: mean distances
    ax = axes[0]
    lows  = [h1_results[a]["mean_low"]  for a in agents]
    highs = [h1_results[a]["mean_high"] for a in agents]
    colours = [COLOURS.get(a, "grey") for a in agents]

    bars_low  = ax.bar(x - width/2, lows,  width, label="Low ecc",
                       color=colours, alpha=0.6, edgecolor="k", linewidth=0.7)
    bars_high = ax.bar(x + width/2, highs, width, label="High ecc",
                       color=colours, alpha=1.0, edgecolor="k", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES.get(a, a) for a in agents], rotation=20, ha="right")
    ax.set_ylabel("Mean WUC distance")
    ax.set_title("H1: Mean RDM distance by eccentricity bin")
    ax.legend()

    # Right: τ_a similarity (low vs high RDM) — lower = more eccentricity effect
    ax = axes[1]
    taus = [h1_results[a]["tau_low_vs_high"] for a in agents]
    bars = ax.bar(x, taus, color=colours, edgecolor="k", linewidth=0.7)

    # Significance asterisks
    for i, agent in enumerate(agents):
        p = h1_results[agent]["p_low_vs_high"]
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        if sig:
            ax.text(i, taus[i] + 0.01, sig, ha="center", va="bottom", fontsize=10)

    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES.get(a, a) for a in agents], rotation=20, ha="right")
    ax.set_ylabel("Kendall τ_a  (low vs high ecc RDMs)")
    ax.set_title("H1: Low–high eccentricity RDM similarity\n(lower = stronger eccentricity effect)")

    fig.suptitle("H1: Eccentricity effect on memory representations", fontsize=13)
    fig.tight_layout()
    path = RESULTS_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_h2_similarity_matrix(h2_results: dict, fname: str = "h2_cross_agent.png"):
    """Heatmap of cross-agent RDM similarity matrix."""
    sim   = h2_results["sim_matrix"]
    names = h2_results["agent_names"]
    labels = [DISPLAY_NAMES.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(6, 5))
    norm = TwoSlopeNorm(vmin=-0.1, vcenter=0.5, vmax=1.0)
    im = ax.imshow(sim, cmap="RdYlGn", norm=norm, aspect="auto")

    for i in range(len(names)):
        for j in range(len(names)):
            val = sim[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if 0.3 < val < 0.8 else "white")

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax, label="Kendall τ_a")
    ax.set_title("H2: Cross-agent RDM similarity\n(high = similar memory geometry)")
    fig.tight_layout()
    path = RESULTS_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_training_dynamics(dynamics: dict, fname: str = "training_dynamics.png"):
    """Line plot of τ_a-to-final-RDM vs training frames per agent."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: τ_a to final RDM
    ax = axes[0]
    for agent, entries in dynamics.items():
        frames = [e["frames"] / 1e6 for e in entries]
        taus   = [e["tau_to_final"] for e in entries]
        ax.plot(frames, taus, marker="o", color=COLOURS.get(agent, "grey"),
                label=DISPLAY_NAMES.get(agent, agent))
    ax.set_xlabel("Training frames (M)")
    ax.set_ylabel("τ_a similarity to final RDM")
    ax.set_title("Memory geometry convergence during training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: mean RDM distance as a function of frames
    ax = axes[1]
    for agent, entries in dynamics.items():
        frames = [e["frames"] / 1e6 for e in entries]
        means  = [e["mean_dist"] for e in entries]
        ax.plot(frames, means, marker="o", color=COLOURS.get(agent, "grey"),
                label=DISPLAY_NAMES.get(agent, agent))
    ax.set_xlabel("Training frames (M)")
    ax.set_ylabel("Mean WUC RDM distance")
    ax.set_title("Mean representational distance during training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = RESULTS_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_eccentricity_rdm_pair(h1_results: dict, agent: str = "foveated",
                                fname: str = "ecc_rdm_pair.png"):
    """Side-by-side RDMs for low vs high eccentricity for a single agent."""
    if agent not in h1_results:
        return
    r = h1_results[agent]
    rdm_low  = r["rdm_low"]
    rdm_high = r["rdm_high"]

    all_vals = np.concatenate([
        rdm_low[np.triu_indices(rdm_low.shape[0], k=1)],
        rdm_high[np.triu_indices(rdm_high.shape[0], k=1)],
    ])
    all_vals = all_vals[np.isfinite(all_vals)]
    vmin, vmax = np.percentile(all_vals, [2, 98])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, rdm, label in zip(axes, [rdm_low, rdm_high], ["Low eccentricity", "High eccentricity"]):
        im = ax.imshow(rdm, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Condition index")
        ax.set_ylabel("Condition index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="WUC distance")

    tau = r["tau_low_vs_high"]
    p   = r["p_low_vs_high"]
    fig.suptitle(
        f"{DISPLAY_NAMES.get(agent, agent)}: eccentricity-split RDMs  "
        f"(τ_a = {tau:.3f}, p = {p:.3f})",
        fontsize=12
    )
    fig.tight_layout()
    path = RESULTS_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Save numerical results
# ---------------------------------------------------------------------------

def save_results_txt(h1: dict, h2: dict):
    """Write a human-readable summary to results/summary.txt."""
    path = RESULTS_DIR / "summary.txt"
    lines = ["=" * 60, "RSA RESULTS SUMMARY", "=" * 60, ""]

    lines += ["H1: Eccentricity effect on memory", "-" * 40]
    for agent in AGENTS:
        if agent not in h1:
            continue
        r = h1[agent]
        lines.append(
            f"  {DISPLAY_NAMES.get(agent, agent):14s}  "
            f"τ_a(low,high) = {r['tau_low_vs_high']:+.3f}  "
            f"p = {r['p_low_vs_high']:.3f}  "
            f"Δmean = {r['mean_high'] - r['mean_low']:+.4f}"
        )

    lines += ["", "H2: Cross-agent RDM similarity (Kendall τ_a)", "-" * 40]
    for key, vals in h2["pairwise_tests"].items():
        lines.append(f"  {key:30s}  τ_a = {vals['tau']:+.3f}  p = {vals['p']:.3f}")

    path.write_text("\n".join(lines) + "\n")
    print(f"  Saved → {path}")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  SPATIAL MEMORY RSA ANALYSIS")
    print("=" * 60)

    # ---- Step 1: Load activations ----
    print("\n[1] Loading activations ...")
    data = load_all_late()
    print(f"  Loaded activations for: {list(data.keys())}")

    # ---- Step 2: Build RDMs ----
    print("\n[2] Building WUC-corrected RDMs ...")
    rdms = build_all_rdms(data)
    plot_rdm_grid(rdms, "WUC-corrected RDMs (converged agents)", "rdm_grid.png")

    # ---- Step 3: H1 — eccentricity ----
    print("\n[3] H1 — Eccentricity test ...")
    h1_results = h1_eccentricity_test(data)
    plot_h1_eccentricity(h1_results)
    plot_eccentricity_rdm_pair(h1_results, agent="foveated")
    plot_eccentricity_rdm_pair(h1_results, agent="uniform",
                                fname="ecc_rdm_pair_uniform.png")

    # ---- Step 4: H2 — cross-agent ----
    print("\n[4] H2 — Cross-agent similarity ...")
    h2_results = h2_cross_agent(rdms)
    plot_h2_similarity_matrix(h2_results)

    # ---- Step 5: Training dynamics (if available) ----
    dyn_agents = [a for a in ["foveated", "uniform", "blind"]
                  if any((ACT_DIR / f"{a}_ckpt{i:02d}.npz").exists()
                          for i in range(50))]
    if len(dyn_agents) >= 2:
        print("\n[5] Training dynamics ...")
        dynamics = training_dynamics(dyn_agents)
        if dynamics:
            plot_training_dynamics(dynamics)
    else:
        print("\n[5] Skipping training dynamics "
              "(run 'python collect_activations.py all' first)")

    # ---- Step 6: Save summary ----
    print("\n[6] Saving summary ...")
    save_results_txt(h1_results, h2_results)

    print(f"\n  All results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()