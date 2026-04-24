#!/bin/bash
#
# Regenerate all paper figures from deterministic-rollout probe data.
#
# Context: `collect.py` had a sampling bug (deterministic=False) that
# produced 4-step quasi-static trajectories for high-entropy policies,
# inflating probe R² values across conditions. The fix landed in
# commit c81352e; after that we re-collected per-condition probes as
# `<cond>_gibson_det.npz`. Several figures in the paper (h2 CKA heatmap,
# h3_content panel a, place_cells, multilayer_heatmap, h2 t-SNE, MP3D
# generalization) pre-date that re-collection and were built from the
# old stochastic data.
#
# This script:
#   1. Ensures every det analysis JSON exists (re-runs analyze.py,
#      analyze_cross.py, unaligned_cka.py, goal_vector_probe.py on
#      the det NPZs if needed).
#   2. Re-runs every paper-figure script pointing at det inputs and
#      writes outputs directly to docs/NeurIPS_2026/fig/.
#
# Run this from the repo root after det NPZs are in place:
#   sbatch scripts/cluster/regenerate_det_figures.sh
#
#SBATCH --job-name=regen_det_figs
#SBATCH --time=01:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

set -e
source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

FIG_OUT="${PROJECT_DIR}/docs/NeurIPS_2026/fig"
PY="python -u"

# ---- Step 0: ensure det analysis JSONs exist ----
need_rerun=0
for cond in blind uniform foveated foveated_learned matched; do
    if [ ! -f "${RESULTS_DIR}/${cond}_gibson_det_analysis.json" ]; then
        need_rerun=1
        echo "[missing] ${cond}_gibson_det_analysis.json — will run analyze.py"
    fi
done
for req in cross_transfer_det.json goal_vector_det.json cka_det.json; do
    if [ ! -f "${RESULTS_DIR}/${req}" ]; then
        need_rerun=1
        echo "[missing] ${req} — will run downstream analyses"
    fi
done

if [ ${need_rerun} -eq 1 ]; then
    echo ""
    echo "=== Running rerun_analyses_on_det.sh to produce missing JSONs ==="
    bash "${PROJECT_DIR}/scripts/cluster/rerun_analyses_on_det.sh"
fi

echo ""
echo "=== Regenerating paper figures ==="

# ---- Fig 2: H1 bottleneck hero ----
# Numbers are hardcoded in the script (match Table 1); no data path needed.
${PY} "${PROJECT_DIR}/scripts/paper_figures/make_bottleneck_figure.py" \
    --out "${FIG_OUT}/h1_bottleneck.pdf"

# ---- Fig 3: H2 CKA heatmap + Table 2: probe transfer ----
# The H1/H2 figure script auto-prefers _det_analysis.json now.
${PY} "${PROJECT_DIR}/scripts/paper_figures/make_h1h2_figures.py" \
    --in-dir "${RESULTS_DIR}" \
    --out-dir "${FIG_OUT}"

# ---- Fig 4: Transplant sweep (not affected by collect.py bug, but regenerate for consistency) ----
${PY} "${PROJECT_DIR}/scripts/paper_figures/make_transplant_sweep_figure.py" \
    --in-dir "${PROJECT_DIR}/data/transplant" \
    --out "${FIG_OUT}/transplant_sweep.pdf" || echo "[warn] transplant_sweep skipped"

# ---- Fig 5: H3 content (a) goal-vector from det + (b) shortcut (deterministic already) ----
${PY} "${PROJECT_DIR}/scripts/paper_figures/make_h3_content_figure.py" \
    --goal-vector "${RESULTS_DIR}/goal_vector_det.json" \
    --shortcut-dir "${PROJECT_DIR}/data/shortcut" \
    --out "${FIG_OUT}/h3_content.pdf"

# ---- Fig 6: place cells + multilayer ----
${PY} "${PROJECT_DIR}/scripts/paper_figures/make_additional_figures.py" \
    --in-dir "${RESULTS_DIR}" \
    --out-dir "${FIG_OUT}"

# ---- Fig 7: MP3D generalisation (requires MP3D det probes) ----
if ls "${RESULTS_DIR}"/*_mp3d_det_analysis.json >/dev/null 2>&1; then
    ${PY} "${PROJECT_DIR}/scripts/paper_figures/make_mp3d_generalization_figure.py" \
        --results-dir "${RESULTS_DIR}" \
        --out-dir "${FIG_OUT}" \
        --suffix "_det"
else
    echo "[skip] Fig 7: no MP3D det analyses yet (${RESULTS_DIR}/*_mp3d_det_analysis.json missing). The paper caption flags this."
fi

# ---- App-B: t-SNE of pooled hidden states ----
${PY} "${PROJECT_DIR}/scripts/paper_figures/make_embedding_figures.py" \
    --in-dir "${PROBE_DIR}" \
    --out-dir "${FIG_OUT}"

# ---- App-A: training curves (TB scalars, not probe-data; regenerate for freshness) ----
${PY} "${PROJECT_DIR}/scripts/paper_figures/make_training_curves.py" \
    --out "${FIG_OUT}/training_curves.pdf" || echo "[warn] training_curves skipped"

echo ""
echo "=== All figures regenerated into ${FIG_OUT} at $(date) ==="
echo "Re-compile the paper: cd docs/NeurIPS_2026 && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex"
