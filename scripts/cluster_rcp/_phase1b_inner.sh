#!/bin/bash
# Phase 1b: rerun LOSO + subspace with new blind (dh-blind) + F2 (fnorm) replacing legacy blind_izar + original foveated.
set -e
cd /scratch/wxu/dh-spatial
source /scratch/wxu/miniconda3/etc/profile.d/conda.sh
conda activate habitat
pip install scikit-learn 2>&1 | tail -2

# Patch CONDS in both scripts to use the new NPZs (blind_det = dh-blind retrain; fnorm_det_ckpt49 = F2 final).
# Log-polar/coarse/uniform stay as originals (post-revert decision).
sed -i 's|blind_izar_det\.npz|blind_det.npz|g' scripts/probing/loop_5cond_loso.py scripts/probing/loop_5cond_subspace_v2.py
sed -i 's|foveated_det\.npz|fnorm_det_ckpt49.npz|g' scripts/probing/loop_5cond_loso.py scripts/probing/loop_5cond_subspace_v2.py

echo "=== loop_5cond_loso.py CONDS ==="
grep -E "blind_det|fnorm_det|foveated_logpolar_det|coarse_det|uniform_det" scripts/probing/loop_5cond_loso.py | head -7

echo ""
echo "=== running LOSO ==="
python scripts/probing/loop_5cond_loso.py

echo ""
echo "=== running subspace_v2 ==="
python scripts/probing/loop_5cond_subspace_v2.py

# Revert CONDS so the script on PVC is back to original
sed -i 's|blind_det\.npz|blind_izar_det.npz|g' scripts/probing/loop_5cond_loso.py scripts/probing/loop_5cond_subspace_v2.py
sed -i 's|fnorm_det_ckpt49\.npz|foveated_det.npz|g' scripts/probing/loop_5cond_loso.py scripts/probing/loop_5cond_subspace_v2.py

echo ""
echo "DONE."
ls -la /scratch/wxu/habitat_checkpoints_rcp/analysis_results/loso_5cond.json /scratch/wxu/habitat_checkpoints_rcp/analysis_results/subspace_divergence_5cond.json
