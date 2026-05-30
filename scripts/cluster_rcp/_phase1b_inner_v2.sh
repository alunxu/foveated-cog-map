#!/bin/bash
# Phase 1b v2: rerun LOSO + subspace WITHOUT pip install (numpy upgrade breaks matplotlib).
# Patch script to save JSON BEFORE figure to be safe.
set -e
cd /scratch/wxu/dh-spatial
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate habitat

# Patch CONDS in both scripts: new blind + F2
sed -i 's|blind_izar_det\.npz|blind_det.npz|g' scripts/probing/loop_5cond_loso.py scripts/probing/loop_5cond_subspace_v2.py
sed -i 's|foveated_det\.npz|fnorm_det_ckpt49.npz|g' scripts/probing/loop_5cond_loso.py scripts/probing/loop_5cond_subspace_v2.py

# Reorder save calls so JSON is written BEFORE savefig (matplotlib PDF font sometimes crashes; we need JSON regardless)
python3 << 'PYEOF'
from pathlib import Path
for fp in ["/scratch/wxu/dh-spatial/scripts/probing/loop_5cond_loso.py",
           "/scratch/wxu/dh-spatial/scripts/probing/loop_5cond_subspace_v2.py"]:
    s = Path(fp).read_text()
    # Replace savefig...json sequence with json...savefig
    # LOSO: lines 108-112; subspace: lines 115-127
    if "loop_5cond_loso.py" in fp:
        s = s.replace(
            '''    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    print(f"Figure saved to {OUT_FIG}")''',
            '''    Path(OUT_JSON).write_text(json.dumps(results, indent=2))
    print(f"JSON saved to {OUT_JSON}")
    try:
        fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
        print(f"Figure saved to {OUT_FIG}")
    except Exception as e:
        print(f"savefig failed (non-fatal): {e}")''',
            1
        )
        s = s.replace(
            '''    Path(OUT_JSON).write_text(json.dumps(results, indent=2))
    print(f"JSON saved to {OUT_JSON}")''',
            "", 1  # remove the late JSON save
        )
    else:
        s = s.replace(
            '    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")',
            '''    Path(OUT_JSON).write_text(json.dumps(out, indent=2))
    print(f"JSON saved to {OUT_JSON}")
    try:
        fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    except Exception as e:
        print(f"savefig failed (non-fatal): {e}")''',
            1
        )
        # Remove the late JSON save
        s = s.replace(
            '    Path(OUT_JSON).write_text(json.dumps(out, indent=2))',
            "    # (already saved above)", 1
        )
    Path(fp).write_text(s)
print("Patched.")
PYEOF

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
echo "DONE. Outputs:"
ls -la /scratch/wxu/habitat_checkpoints_rcp/analysis_results/loso_5cond.json /scratch/wxu/habitat_checkpoints_rcp/analysis_results/subspace_divergence_5cond.json
