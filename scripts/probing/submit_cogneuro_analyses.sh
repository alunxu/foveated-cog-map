#!/bin/bash
# submit_cogneuro_analyses.sh
#
# SLURM submission script for grid-cell signature + time-cell analyses.
#
# These are CPU-only analysis jobs that read pre-collected NPZ files
# (from scripts/probing/collect.py) and write JSON + NPZ results.
# No GPU required; adjust --mem and --cpus-per-task if needed.
#
# Usage (from repo root on Izar):
#   sbatch scripts/probing/submit_cogneuro_analyses.sh
#
# Prerequisites:
#   - conda activate habitat   (or habitat-analysis for lightweight env)
#   - NPZ files already collected under $PROBING_DIR/<cond>.npz
#
# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=cogneuro_analyses
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --output=slurm_logs/%j_cogneuro.out
#SBATCH --error=slurm_logs/%j_cogneuro.err
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── paths — edit these for your Izar layout ───────────────────────────────────
REPO_DIR="/scratch/izar/${USER}/foveated-cog-map"
PROBING_DIR="/scratch/izar/${USER}/probing_data"
RESULTS_DIR="/scratch/izar/${USER}/probing_results"
FIGURES_DIR="${REPO_DIR}/docs/manuscript/fig"

# Conda env name (use 'habitat' if full env already exists on Izar,
# or 'habitat-analysis' for the lightweight local env).
CONDA_ENV="${CONDA_ENV:-habitat}"

# Conditions to analyse (must match <cond>.npz filenames in $PROBING_DIR).
CONDS=(
    "blind_gibson"
    "uniform_gibson"
    "foveated_gibson"
    "foveated_logpolar_gibson"
)

# ── environment setup ─────────────────────────────────────────────────────────
module load gcc/12.2.0 2>/dev/null || true
module load cuda/12.1.1 2>/dev/null || true

# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${REPO_DIR}"
pip install -e . --quiet

mkdir -p "${RESULTS_DIR}" "${FIGURES_DIR}" slurm_logs

echo "=========================================="
echo " Job: cogneuro analyses"
echo " Date: $(date)"
echo " Node: $(hostname)"
echo " Conda env: ${CONDA_ENV}"
echo " Probing dir: ${PROBING_DIR}"
echo " Conditions: ${CONDS[*]}"
echo "=========================================="

# ── 1. Grid-cell signature analysis ──────────────────────────────────────────
echo ""
echo "── Grid-cell signature ──────────────────────────────────────────────────"
python scripts/probing/grid_cell_signature.py \
    --in-dir    "${PROBING_DIR}" \
    --conds     "${CONDS[@]}" \
    --out       "${RESULTS_DIR}/grid_cell_signature.json" \
    --n-bins    24 \
    --n-shuffles 200 \
    --n-top-scenes 10 \
    --min-scene-steps 500 \
    --layer     -1 \
    --save-maps

echo "Grid-cell signature done."

# ── 2. Time-cell analysis ─────────────────────────────────────────────────────
echo ""
echo "── Time cells ───────────────────────────────────────────────────────────"
python scripts/probing/time_cells.py \
    --in-dir    "${PROBING_DIR}" \
    --conds     "${CONDS[@]}" \
    --out       "${RESULTS_DIR}/time_cells.json" \
    --out-npz   "${RESULTS_DIR}/time_cells_curves.npz" \
    --n-bins    20 \
    --n-shuffles 500 \
    --min-ep-steps 20 \
    --layer     -1

echo "Time-cell analysis done."

# ── 3. Figures ────────────────────────────────────────────────────────────────
echo ""
echo "── Generating figures ───────────────────────────────────────────────────"

python scripts/paper_figures/make_grid_cell_figure.py \
    --data "${RESULTS_DIR}/grid_cell_signature.json" \
    --out  "${FIGURES_DIR}/grid_cell_signature.pdf"

python scripts/paper_figures/make_time_cells_figure.py \
    --data   "${RESULTS_DIR}/time_cells.json" \
    --curves "${RESULTS_DIR}/time_cells_curves.npz" \
    --out    "${FIGURES_DIR}/time_cells.pdf"

echo ""
echo "=========================================="
echo " All done.  Results in ${RESULTS_DIR}"
echo " Figures in  ${FIGURES_DIR}"
echo " Elapsed: ${SECONDS}s"
echo "=========================================="
