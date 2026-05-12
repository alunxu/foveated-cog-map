#!/bin/bash
# Paper-level clean SPL/Success eval, 5 conditions × 500 deterministic episodes.
#
# Strategy: bypass habitat-baselines' run.main() (whose dataset config keeps
# resolving to default `habitat-test-scenes` no matter how many overrides we
# pass).  Use the proven probe_agent.py-style pattern:
#   load_habitat_config() + habitat.Env() + load_policy()
# via scripts/eval/eval_paper_5cond.py.
#
# Submits 5 parallel runai-rcp jobs (one per condition).
# ETA per cond: ~25-40 min single-env @ 500 eps; total wall-clock ~40 min.
set -e

IMAGE="registry.rcp.epfl.ch/dhlab-wxu/habitat:v2"
OUT_DIR="/scratch/wxu/habitat_checkpoints_rcp/eval_5cond"
N_EPS=500
SPLIT=train  # our merged Gibson+MP3D pool only has train; rng-sampled (start,goal) pairs are held-out

declare -a CONDS=(
    "blind|pointnav/ddppo_pointnav_blind_gibson|/scratch/wxu/habitat_checkpoints_rcp/dh-blind/ckpt.49.pth"
    "coarse|pointnav/ddppo_pointnav_coarse_gibson|/scratch/wxu/habitat_checkpoints_rcp/dh-probe-1/ckpt.49.pth"
    "foveated|pointnav/ddppo_pointnav_foveated_gibson|/scratch/wxu/habitat_checkpoints_rcp/dh-probe-2/ckpt.49.pth"
    "uniform|pointnav/ddppo_pointnav_uniform_gibson|/scratch/wxu/habitat_checkpoints_rcp/dh-probe-3/ckpt.49.pth"
    "foveated_logpolar|pointnav/ddppo_pointnav_foveated_logpolar_gibson|/scratch/wxu/habitat_checkpoints_rcp/dh-probe-4/ckpt.49.pth"
)

for entry in "${CONDS[@]}"; do
    IFS='|' read -r COND CFG CKPT <<< "$entry"
    JOB_NAME="eval5-${COND//_/-}"
    OUT_JSON="${OUT_DIR}/${COND}.json"
    LOG="${OUT_DIR}/${COND}.log"

    # Use PVC-resident inner script to avoid runai-cli inline-quoting bug
    # (2026-05-12: same bug bit dh-blind-resume; inline INNER_CMD with
    # backslash continuations + here-doc-style strings gets mangled).
    INNER_CMD="bash /scratch/wxu/dh-spatial/scripts/cluster/_eval5_inner.sh ${COND} ${CFG} ${CKPT} ${N_EPS} ${SPLIT}"

    echo "Submitting ${JOB_NAME} ..."
    RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod -p dhlab-wxu submit "$JOB_NAME" \
        --image="$IMAGE" \
        --gpu=1 --cpu=4 --memory=24G \
        --pvc=dhlab-scratch:/scratch \
        --pvc=home:/home/wxu \
        --large-shm \
        --command -- bash -c "$INNER_CMD"
done

echo
echo "All 5 conditions submitted.  Outputs will land in ${OUT_DIR}/"
echo "  - blind.json"
echo "  - coarse.json"
echo "  - foveated.json"
echo "  - uniform.json"
echo "  - foveated_logpolar.json"
echo "Combine with: python scripts/cluster/combine_5cond_eval.py"
