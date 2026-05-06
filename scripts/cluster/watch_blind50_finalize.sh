#!/bin/bash
# Watcher for probe-5-c5 (blind_izar ckpt5 = 50M-equivalent probe-collect job).
# Polls every 60s until the job succeeds, then:
#   1. submits a follow-up analyze runai job
#   2. polls until analyze succeeds
#   3. rsyncs the resulting blind_izar_det_ckpt5_analysis.json to /tmp/rcp_analysis/
#   4. regenerates docs/manuscript/fig/fig_magnitude.pdf
#   5. auto-commits the figure update
#
# Run in background:
#   nohup bash scripts/cluster/watch_blind50_finalize.sh > /tmp/blind50_watcher.log 2>&1 &
set -e

REPO=/Users/alunx/Desktop/Aluniverse/Courses/2026-Spring-CS503-Visual-Intelligence-Homework/Project
WORKTREE="$(pwd)"
JOB="probe-5-c5"
ANALYZE_JOB="probe-5-c5-analyze"
NPZ_PATH="/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp/blind_izar_det_ckpt5.npz"
JSON_PATH="/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp/blind_izar_det_ckpt5_analysis.json"
LOCAL_JSON="/tmp/rcp_analysis/blind_izar_det_ckpt5_analysis.json"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ── Step 1: wait for probe-5-c5 to succeed ────────────────────────────
log "Watching $JOB status..."
while true; do
    STATUS=$(RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod describe job $JOB --project dhlab-wxu 2>/dev/null \
             | grep -E "^Status:" | awk '{print $2}')
    log "$JOB status: ${STATUS:-unknown}"
    if [ "$STATUS" = "Succeeded" ]; then
        log "$JOB succeeded — proceeding to analyze."
        break
    fi
    if [ "$STATUS" = "Failed" ]; then
        log "$JOB failed — aborting watcher."
        exit 1
    fi
    sleep 60
done

# ── Step 2: submit analyze job ────────────────────────────────────────
log "Submitting $ANALYZE_JOB (analyze.py on $NPZ_PATH)..."

INNER="set -e; source /opt/miniconda3/etc/profile.d/conda.sh; conda activate habitat; cd /scratch/wxu/dh-spatial; export PYTHONPATH=/scratch/wxu/dh-spatial:\$PYTHONPATH; python -u /scratch/wxu/dh-spatial/scripts/probing/analyze.py --data=${NPZ_PATH} --out=${JSON_PATH} --pca-dim=0 --min-steps-scene=15; echo ANALYZE_DONE; ls -la ${JSON_PATH}"

RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod submit "$ANALYZE_JOB" \
    --project dhlab-wxu \
    --image="registry.rcp.epfl.ch/dhlab-wxu/habitat:v2" \
    --cpu=4 \
    --memory=16G \
    --pvc=dhlab-scratch:/scratch \
    --command -- bash -c "$INNER" || {
        log "submit failed (job may already exist; will keep polling)"
    }

# ── Step 3: wait for analyze to succeed ───────────────────────────────
log "Watching $ANALYZE_JOB status..."
while true; do
    STATUS=$(RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod describe job $ANALYZE_JOB --project dhlab-wxu 2>/dev/null \
             | grep -E "^Status:" | awk '{print $2}')
    log "$ANALYZE_JOB status: ${STATUS:-unknown}"
    if [ "$STATUS" = "Succeeded" ]; then
        log "$ANALYZE_JOB succeeded — fetching JSON."
        break
    fi
    if [ "$STATUS" = "Failed" ]; then
        log "$ANALYZE_JOB failed — aborting watcher."
        exit 1
    fi
    sleep 30
done

# ── Step 4: rsync JSON from RCP to local /tmp/rcp_analysis/ ───────────
# Spawn a one-shot pod to read from PVC and tar to stdout — same trick
# as sync_from_cluster.sh but only for one tiny JSON file.
log "Pulling $JSON_PATH from RCP scratch to $LOCAL_JSON..."
mkdir -p "$(dirname $LOCAL_JSON)"

POD_NAME="blind50-fetch-$RANDOM"
RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod submit "$POD_NAME" \
    --project dhlab-wxu \
    --image=alpine \
    --cpu=1 --memory=1G \
    --pvc=dhlab-scratch:/scratch \
    --command -- sh -c "cat $JSON_PATH; sleep 5" 2>&1 | head -3 || true

sleep 15
# kubectl logs gives us the JSON contents on stdout
kubectl logs -n runai-dhlab-wxu \
    "$(kubectl get pods -n runai-dhlab-wxu -l release=$POD_NAME -o name | head -1 | cut -d/ -f2)" \
    > "$LOCAL_JSON" 2>/dev/null || {
        log "kubectl-logs fetch failed; try manual rsync"
    }

# Verify JSON parses
python3 -c "import json; d = json.load(open('$LOCAL_JSON')); print('blind_izar ckpt5 cv_r2:', d['1b_global_gps_compass']['gps_cv_r2_mean'])" || {
    log "JSON parse failed — fetched content may be wrong"
    exit 1
}

# Cleanup pod
RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod delete job "$POD_NAME" --project dhlab-wxu 2>/dev/null || true

# ── Step 5: regenerate figure ──────────────────────────────────────────
log "Regenerating fig_magnitude.pdf..."
cd "$REPO"
python3 scripts/paper_figures/make_magnitude_3panel.py

# ── Step 6: compile + commit ──────────────────────────────────────────
log "Compiling main.tex..."
cd "$REPO/docs/manuscript"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/p1.log 2>&1
bibtex main > /tmp/bib.log 2>&1 || true
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/p2.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/p3.log 2>&1
rm -f main.aux main.bbl main.blg main.log main.out

cd "$REPO"
git add docs/manuscript/main.pdf docs/manuscript/fig/fig_magnitude.pdf docs/manuscript/fig/fig_magnitude.png
git commit -m "fig_magnitude Panel B: add blind 50M point (probe-5-c5 landed)

probe-5-c5 cluster job (blind_izar ckpt.5 = 50M-equivalent probe-collect)
finished. Ran analyze.py on the resulting npz, synced
blind_izar_det_ckpt5_analysis.json to /tmp/rcp_analysis/, regenerated
fig_magnitude.pdf. Blind now has 4 points within the [50M, 250M] window
(50M, 100M, 200M, 252M) — symmetric with the 5 sighted points." || log "no changes to commit"

log "Done. Figure updated."
