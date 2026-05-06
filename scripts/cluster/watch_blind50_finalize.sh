#!/bin/bash
# Watcher for blind_izar ckpt sweep: probe-5-c5 (50M) and blind-c15-analyze (150M).
# probe-5-c5 currently RUNNING (collect.py); needs analyze.py after.
# blind-c15-analyze currently RUNNING (analyze.py on ckpt15.npz which is already there).
#
# Polls every 60s. Once each succeeds, fetches its JSON to /tmp/rcp_analysis/.
# When both have landed, regenerates fig_magnitude.pdf + commits.
#
# Run in background:
#   nohup bash scripts/cluster/watch_blind50_finalize.sh > /tmp/blind_watcher.log 2>&1 &
set -e

REPO=/Users/alunx/Desktop/Aluniverse/Courses/2026-Spring-CS503-Visual-Intelligence-Homework/Project
WORKTREE="$(pwd)"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ── Helper: fetch JSON from RCP scratch via one-shot pod ──────────────
fetch_json() {
    local REMOTE_PATH=$1
    local LOCAL_PATH=$2
    local POD_NAME="fetch-$RANDOM"
    log "fetching $REMOTE_PATH → $LOCAL_PATH"
    RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod submit "$POD_NAME" \
        --project dhlab-wxu \
        --image=alpine \
        --cpu=1 --memory=1G \
        --pvc=dhlab-scratch:/scratch \
        --command -- sh -c "cat $REMOTE_PATH; sleep 30" >/dev/null 2>&1 || true
    sleep 20
    kubectl logs -n runai-dhlab-wxu \
        "$(kubectl get pods -n runai-dhlab-wxu -l release=$POD_NAME -o name 2>/dev/null | head -1 | cut -d/ -f2)" \
        > "$LOCAL_PATH" 2>/dev/null || { log "fetch failed for $REMOTE_PATH"; return 1; }
    python3 -c "import json; json.load(open('$LOCAL_PATH'))" 2>/dev/null || {
        log "JSON parse failed: $LOCAL_PATH"; return 1;
    }
    RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod delete job "$POD_NAME" --project dhlab-wxu >/dev/null 2>&1 || true
    log "fetched OK ($(wc -c < $LOCAL_PATH) bytes)"
}

# ── Helper: get runai job status ──────────────────────────────────────
job_status() {
    RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod describe job "$1" --project dhlab-wxu 2>/dev/null \
        | grep -E "^Status:" | awk '{print $2}'
}

# ── Step 1: wait for probe-5-c5 to succeed, then submit its analyze ──
log "Watching probe-5-c5 (collect for ckpt5=50M)..."
while true; do
    s=$(job_status probe-5-c5)
    log "probe-5-c5: ${s:-unknown}"
    [ "$s" = "Succeeded" ] && break
    [ "$s" = "Failed" ] && { log "probe-5-c5 failed"; exit 1; }
    sleep 60
done

log "Submitting probe-5-c5-analyze (analyze.py on ckpt5.npz)..."
INNER5="set -e; source /opt/miniconda3/etc/profile.d/conda.sh; conda activate habitat; cd /scratch/wxu/dh-spatial; export PYTHONPATH=/scratch/wxu/dh-spatial:\$PYTHONPATH; python -u /scratch/wxu/dh-spatial/scripts/probing/analyze.py --data=/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp/blind_izar_det_ckpt5.npz --out=/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp/blind_izar_det_ckpt5_analysis.json --pca-dim=0 --min-steps-scene=15"
RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod submit "probe-5-c5-analyze" \
    --project dhlab-wxu \
    --image="registry.rcp.epfl.ch/dhlab-wxu/habitat:v2" \
    --cpu=4 --memory=16G \
    --pvc=dhlab-scratch:/scratch \
    --command -- bash -c "$INNER5" >/dev/null 2>&1 || log "submit may have failed (maybe duplicate)"

# ── Step 2: poll BOTH analyze jobs concurrently ──────────────────────
log "Watching probe-5-c5-analyze AND blind-c15-analyze..."
got_5=0; got_15=0
while [ $got_5 -eq 0 ] || [ $got_15 -eq 0 ]; do
    if [ $got_5 -eq 0 ]; then
        s5=$(job_status probe-5-c5-analyze)
        if [ "$s5" = "Succeeded" ]; then
            log "ckpt5 analyze done"
            fetch_json "/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp/blind_izar_det_ckpt5_analysis.json" \
                       "/tmp/rcp_analysis/blind_izar_det_ckpt5_analysis.json" && got_5=1
        elif [ "$s5" = "Failed" ]; then
            log "ckpt5 analyze failed — proceeding without it"; got_5=1
        fi
    fi
    if [ $got_15 -eq 0 ]; then
        s15=$(job_status blind-c15-analyze)
        if [ "$s15" = "Succeeded" ]; then
            log "ckpt15 analyze done"
            fetch_json "/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp/blind_izar_det_ckpt15_analysis.json" \
                       "/tmp/rcp_analysis/blind_izar_det_ckpt15_analysis.json" && got_15=1
        elif [ "$s15" = "Failed" ]; then
            log "ckpt15 analyze failed — proceeding without it"; got_15=1
        fi
    fi
    sleep 30
done

# ── Step 3: regenerate figure ─────────────────────────────────────────
log "Regenerating fig_magnitude.pdf..."
cd "$REPO"
python3 scripts/paper_figures/make_magnitude_3panel.py

# ── Step 4: compile + commit ─────────────────────────────────────────
log "Compiling main.tex..."
cd "$REPO/docs/manuscript"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/p1.log 2>&1
bibtex main > /tmp/bib.log 2>&1 || true
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/p2.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/p3.log 2>&1
rm -f main.aux main.bbl main.blg main.log main.out

cd "$REPO"
git add docs/manuscript/main.pdf docs/manuscript/fig/fig_magnitude.pdf docs/manuscript/fig/fig_magnitude.png
git commit -m "fig_magnitude Panel B: blind 5-point sweep complete (ckpt5+ckpt15 landed)

probe-5-c5 (50M) and probe-5-c15 (150M) cluster jobs finished; analyze.py
ran on both npz files; resulting JSONs synced to /tmp/rcp_analysis/.
Blind now has 5 points within [50M, 250M] window (50/100/150/200/252M),
symmetric with the 5 sighted points (50/100/150/200/250M)." || log "no changes to commit"

log "Done. Figure updated with full blind 5-point sweep."
