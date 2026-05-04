#!/bin/bash
# Recompute Skaggs spatial info with paper-faithful rectified h2 (max(h2, 0))
# rather than analyze.py's raw h2 (which causes 100x inflation when global_mean
# is near 0). Standalone NumPy job, NO GPU needed.

set -e

JOB_NAME="skaggs-recompute"
IMAGE="registry.rcp.epfl.ch/dhlab-wxu/habitat:v2"
RESULTS_DIR="/scratch/wxu/habitat_checkpoints_rcp/analysis_results"
NPZ_DIR="/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp"

INNER_CMD="set -e; source /opt/miniconda3/etc/profile.d/conda.sh; conda activate habitat; cd /scratch/wxu/dh-spatial; export PATH=/home/wxu/.local/bin:\$PATH; mkdir -p ${RESULTS_DIR}; python -u <<'PY' 2>&1 | tee ${RESULTS_DIR}/skaggs_rectified.log
import numpy as np, json, os
def skaggs_rectified(H, positions, scene_ids, n_bins=20, min_steps=20):
    H_rect = np.maximum(H, 0)
    unique_scenes = np.unique(scene_ids)
    all_si = []
    for sid in unique_scenes:
        mask = scene_ids == sid
        if mask.sum() < min_steps: continue
        h_s = H_rect[mask]
        xz = positions[mask][:, [0, 2]]
        x_edges = np.linspace(xz[:,0].min()-1e-6, xz[:,0].max()+1e-6, n_bins+1)
        z_edges = np.linspace(xz[:,1].min()-1e-6, xz[:,1].max()+1e-6, n_bins+1)
        x_bin = np.clip(np.digitize(xz[:,0], x_edges)-1, 0, n_bins-1)
        z_bin = np.clip(np.digitize(xz[:,1], z_edges)-1, 0, n_bins-1)
        bin_idx = x_bin * n_bins + z_bin
        occupancy = np.bincount(bin_idx, minlength=n_bins*n_bins).astype(float)
        sum_act = np.zeros((n_bins*n_bins, h_s.shape[1]))
        np.add.at(sum_act, bin_idx, h_s)
        occ_mask = occupancy > 0
        if occ_mask.sum() < 4: continue
        p_occ = occupancy[occ_mask] / occupancy[occ_mask].sum()
        mean_act = sum_act[occ_mask] / occupancy[occ_mask, None]
        global_mean = h_s.mean(axis=0)
        si = np.zeros(h_s.shape[1])
        for j in range(h_s.shape[1]):
            lam = global_mean[j]
            if lam < 1e-8: continue
            ratio = mean_act[:, j] / lam
            ratio = np.clip(ratio, 1e-8, None)
            si[j] = np.sum(p_occ * ratio * np.log2(ratio))
        all_si.append(si)
    return np.array(all_si) if all_si else None

NPZ_DIR = '/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp'
RESULTS = '/scratch/wxu/habitat_checkpoints_rcp/analysis_results'
out = {}
print(f'{\"cond\":<22} {\"mean\":>10} {\"std\":>10} {\"max\":>10} {\"n_scenes\":>10} {\"place_units(>1bit)\":>20} {\"place_units(>0.5bit)\":>22}')
for c in ['coarse', 'foveated', 'uniform', 'foveated_logpolar']:
    p = f'{NPZ_DIR}/{c}_det.npz'
    if not os.path.exists(p):
        continue
    d = np.load(p, allow_pickle=True)
    si_mat = skaggs_rectified(d['hidden_states'], d['positions'], d['scene_ids'])
    if si_mat is None:
        continue
    mean_si = si_mat.mean(axis=0)
    res = {
        'mean_per_unit_per_scene': float(mean_si.mean()),
        'std_per_unit_per_scene': float(mean_si.std()),
        'max_per_unit_per_scene': float(mean_si.max()),
        'n_scenes_used': int(len(si_mat)),
        'n_place_units_1bit': int((mean_si > 1.0).sum()),
        'n_place_units_05bit': int((mean_si > 0.5).sum()),
    }
    out[c] = res
    print(f'{c:<22} {res[\"mean_per_unit_per_scene\"]:>10.4f} {res[\"std_per_unit_per_scene\"]:>10.4f} {res[\"max_per_unit_per_scene\"]:>10.4f} {res[\"n_scenes_used\"]:>10} {res[\"n_place_units_1bit\"]:>20} {res[\"n_place_units_05bit\"]:>22}')
with open(f'{RESULTS}/skaggs_rectified.json', 'w') as f:
    json.dump(out, f, indent=2)
print()
print('Paper L277: 1.25 (blind), 1.32 (coarse), 1.18 (fov), 1.16 (uni); place-units 174/116/237/116')
PY
echo SKAGGS_DONE"

RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod submit "$JOB_NAME" \
    --project dhlab-wxu --image="$IMAGE" --cpu=4 --memory=16G \
    --pvc=dhlab-scratch:/scratch --pvc=home:/home/wxu \
    --command -- bash -c "$INNER_CMD"
echo "Submitted $JOB_NAME (CPU-only). Track via: kubectl logs -n runai-dhlab-wxu \$(kubectl get pods -n runai-dhlab-wxu -l release=$JOB_NAME -o name | head -1 | cut -d/ -f2) -f"
