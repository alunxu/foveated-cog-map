# CS503 Visual Intelligence — Course Project

## Project Structure

```
Project/
├── README.md                    # This file
├── pyproject.toml               # Python package config (editable install)
├── requirements.txt             # Pinned dependencies
├── setup_env.sh                 # One-time cluster environment setup
├── submit_job.sh                # Single-node SLURM batch submission
├── submit_job_multi_node.sh     # Multi-node SLURM batch submission
├── sync_to_cluster.sh           # rsync project to SCITAS
├── sync_from_cluster.sh         # rsync results back from SCITAS
│
├── cfgs/                        # Experiment configs (YAML)
│   └── example.yaml             # Example config template
│
├── src/                         # Main Python package
│   ├── __init__.py
│   ├── data/                    # Dataloaders and data utilities
│   │   └── __init__.py
│   ├── models/                  # Model implementations
│   │   └── __init__.py
│   ├── losses/                  # Loss functions
│   │   └── __init__.py
│   └── utils/                   # Training, evaluation, checkpointing utilities
│       └── __init__.py
│
├── scripts/                     # Standalone scripts  
│   ├── train.py                 # Main training entry point
│   ├── evaluate.py              # Evaluation / inference script
│   └── visualize.py             # Visualization and qualitative analysis
│
├── notebooks/                   # Jupyter notebooks for analysis
│   └── .gitkeep
│
├── tests/                       # Unit tests
│   └── __init__.py
│
├── outputs/                     # Training outputs (gitignored)
│   └── .gitkeep
│
└── docs/                        # Project webpage (GitHub Pages)
    └── index.html               # Final report webpage
```

## Storage Strategy

### Where things live on SCITAS (Izar)

| What | Where | Why |
|------|-------|-----|
| **Code** | `/home/<username>/CS503_Project/` | Backed up nightly, 100 GB quota, permanent |
| **Datasets** | `/scratch/<username>/CS503_Project/data/` | Large, ephemeral (>2 week auto-delete) |
| **Checkpoints** | `/scratch/<username>/CS503_Project/checkpoints/` | Large, ephemeral — download important ones |
| **Logs / W&B** | `/home/<username>/CS503_Project/outputs/` | Small, keep with code |
| **Best model** | `/home/<username>/CS503_Project/outputs/best/` | Small enough for Home, or download locally |

### Local machine

| What | Where |
|------|-------|
| **Code** | This directory (git-tracked) |
| **Results / Figures** | `outputs/` (gitignored, rsync from cluster) |
| **Report assets** | `docs/` (git-tracked for GitHub Pages) |

## Quick Start

### 1. First-time setup on SCITAS

```bash
# SSH into the cluster
ssh -X wxu@izar.epfl.ch

# Upload project
bash sync_to_cluster.sh

# On the cluster, set up the environment
cd /home/wxu/CS503_Project
bash setup_env.sh
```

### 2. Running experiments

```bash
# Interactive (debugging, short runs)
srun -t 120 -A cs-503 --qos=cs-503 --gres=gpu:1 --mem=16G --pty bash
conda activate cs503_project
python scripts/train.py --config cfgs/example.yaml

# Batch (longer training)
sbatch submit_job.sh cfgs/example.yaml <WANDB_API_KEY> 1
```

### 3. Syncing results back

```bash
# From local machine
bash sync_from_cluster.sh
```

## Environment

The project uses a dedicated conda environment (`cs503_project`) separate from the NanoFM homework environment to avoid dependency conflicts.

## W&B Integration

All experiments log to Weights & Biases:
- **Entity**: `alun-xu-epfl`
- **Project**: `CS503_Project`

## Team

| Name | SCIPER | Role |
|------|--------|------|
| Weilun Xu | — | — |
| (add teammates) | — | — |

## Deadlines

| Milestone | Date | Deliverable |
|-----------|------|-------------|
| Proposal | TBD | 1–2 page PDF on Moodle |
| Progress Report | TBD | ≤2 page PDF on Moodle |
| Midterm Presentation | TBD | 3 min slides |
| Final Presentation | TBD | 5 min slides |
| Final Webpage + Code | TBD | GitHub Pages + ZIP on Moodle |
