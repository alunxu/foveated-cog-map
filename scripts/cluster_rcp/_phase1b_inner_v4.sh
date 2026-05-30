#!/bin/bash
# Phase 1b v4: install sklearn with numpy<2 pin to avoid matplotlib font bug.
set -e
cd /scratch/wxu/dh-spatial
source /scratch/wxu/miniconda3/etc/profile.d/conda.sh
conda activate habitat

# Install sklearn but PIN numpy<2 (numpy 2.x breaks matplotlib PDF font embed)
pip install scikit-learn 'numpy<2' 2>&1 | tail -3
echo "numpy: $(python -c 'import numpy; print(numpy.__version__)')"
echo "sklearn: $(python -c 'import sklearn; print(sklearn.__version__)')"

# Patch CONDS: new blind + F2
sed -i 's|blind_izar_det\.npz|blind_det.npz|g' scripts/probing/loop_5cond_loso.py scripts/probing/loop_5cond_subspace_v2.py
sed -i 's|foveated_det\.npz|fnorm_det_ckpt49.npz|g' scripts/probing/loop_5cond_loso.py scripts/probing/loop_5cond_subspace_v2.py

# Reorder save calls so JSON is written BEFORE savefig
python3 << 'PYEOF'
from pathlib import Path
for fp in ["/scratch/wxu/dh-spatial/scripts/probing/loop_5cond_loso.py",
           "/scratch/wxu/dh-spatial/scripts/probing/loop_5cond_subspace_v2.py"]:
    s = Path(fp).read_text()
    if "_loso.py" in fp:
        old = '''    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    print(f"Figure saved to {OUT_FIG}")'''
        new = '''    Path(OUT_JSON).write_text(json.dumps(results, indent=2))
    print(f"JSON saved to {OUT_JSON}")
    try:
        fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
        print(f"Figure saved to {OUT_FIG}")
    except Exception as e:
        print(f"savefig failed (non-fatal): {e}")'''
        s = s.replace(old, new, 1)
        s = s.replace('    Path(OUT_JSON).write_text(json.dumps(results, indent=2))\n    print(f"JSON saved to {OUT_JSON}")', '    # (saved above)', 1)
    else:
        old = '    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")'
        new = '''    Path(OUT_JSON).write_text(json.dumps(out, indent=2))
    print(f"JSON saved to {OUT_JSON}")
    try:
        fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    except Exception as e:
        print(f"savefig failed (non-fatal): {e}")'''
        s = s.replace(old, new, 1)
        s = s.replace('    Path(OUT_JSON).write_text(json.dumps(out, indent=2))', '    # (saved above)', 1)
    Path(fp).write_text(s)
print("Patched.")
PYEOF

echo "=== running LOSO ==="
python scripts/probing/loop_5cond_loso.py

echo ""
echo "=== running subspace_v2 ==="
python scripts/probing/loop_5cond_subspace_v2.py

# Revert CONDS
sed -i 's|blind_det\.npz|blind_izar_det.npz|g' scripts/probing/loop_5cond_loso.py scripts/probing/loop_5cond_subspace_v2.py
sed -i 's|fnorm_det_ckpt49\.npz|foveated_det.npz|g' scripts/probing/loop_5cond_loso.py scripts/probing/loop_5cond_subspace_v2.py

echo ""
echo "DONE. Outputs:"
ls -la /scratch/wxu/habitat_checkpoints_rcp/analysis_results/loso_5cond.json /scratch/wxu/habitat_checkpoints_rcp/analysis_results/subspace_divergence_5cond.json
