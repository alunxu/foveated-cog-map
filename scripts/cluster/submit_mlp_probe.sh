#!/bin/bash
#SBATCH --job-name=cs503_mlp
#SBATCH --time=01:00:00
#SBATCH --account=cs-503
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Run mlp_probe_sanity.py on the deterministic-rollout NPZs.
# B3 sanity: confirm the H1 ordering is not a linear-probe artefact.

source "${SLURM_SUBMIT_DIR}/scripts/cluster/common.sh"

python -u "${PROJECT_DIR}/scripts/probing/mlp_probe_sanity.py" \
    --in-dir "${PROBE_DIR}" \
    --suffix _det \
    --out "${RESULTS_DIR}/mlp_sanity_det.json"
