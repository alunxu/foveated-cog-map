#!/bin/bash
# Phase 1a v2: re-run MINE with the CANONICAL mine_multiseed.py (fix my buggy bias-correction)
# + analyze.py on blind_det_ckpt10.npz (which had sklearn-missing on collect-time analyze).
set -e
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate habitat
pip install scikit-learn 2>&1 | tail -2

cd /scratch/wxu/dh-spatial
export PYTHONPATH=/scratch/wxu/dh-spatial:$PYTHONPATH

# ---- (a) MINE: patch canonical script to add fnorm/flp2 to its CONDS ----
ORIG=/scratch/wxu/dh-spatial/scripts/probing/mine_multiseed.py
TMP=/tmp/mine_multiseed_norm.py
cp "$ORIG" "$TMP"
python3 - <<'PYPATCH'
import re
with open("/tmp/mine_multiseed_norm.py") as f:
    code = f.read()
old = """CONDS = [
    ("blind",             f"{NPZ_DIR}/blind_izar_det.npz",         "Blind"),
    ("coarse",            f"{NPZ_DIR}/coarse_det.npz",             "Coarse"),
    ("foveated",          f"{NPZ_DIR}/foveated_det.npz",           "Foveated"),
    ("foveated_logpolar", f"{NPZ_DIR}/foveated_logpolar_det.npz",  "Fov-LP"),
    ("uniform",           f"{NPZ_DIR}/uniform_det.npz",            "Uniform"),
]"""
new = """CONDS = [
    ("fnorm", f"{NPZ_DIR}/fnorm_det_ckpt49.npz",  "Foveated+Norm"),
    ("flp2",  f"{NPZ_DIR}/flp2_det_ckpt49.npz",   "Fov-LP+Norm"),
]"""
code = code.replace(old, new)
# Also change output path so we don't overwrite blind/coarse/etc. existing
code = code.replace(
    'OUT_JSON = "/scratch/wxu/habitat_checkpoints_rcp/analysis_results/mine_multiseed_5cond.json"',
    'OUT_JSON = "/scratch/wxu/habitat_checkpoints_rcp/analysis_results/mine_norm_pair.json"',
)
with open("/tmp/mine_multiseed_norm.py", "w") as f:
    f.write(code)
PYPATCH

python -u /tmp/mine_multiseed_norm.py 2>&1 | tail -40

# Now merge mine_norm_pair.json INTO mine_multiseed_5cond.json (replace foveated/logpolar entries)
python3 << 'PYMERGE'
import json
EXIST = "/scratch/wxu/habitat_checkpoints_rcp/analysis_results/mine_multiseed_5cond.json"
NEW   = "/scratch/wxu/habitat_checkpoints_rcp/analysis_results/mine_norm_pair.json"
d = json.load(open(EXIST))
new = json.load(open(NEW))
# Add fnorm + flp2 (don't overwrite foveated / log-polar yet — keep both for now)
for k, v in new.items():
    d[k] = v
with open(EXIST, "w") as f:
    json.dump(d, f, indent=2)
print(f"PATCHED {EXIST} with keys: {list(new.keys())}")
PYMERGE

# ---- (b) analyze.py on blind_det_ckpt10.npz (sklearn was missing earlier) ----
NPZ=/scratch/wxu/habitat_checkpoints_rcp/probing_data_rcp/blind_det_ckpt10.npz
OUT=/scratch/wxu/habitat_checkpoints_rcp/analysis_results/blind_det_ckpt10_analysis.json
if [ -f "$NPZ" ] && [ ! -f "$OUT" ]; then
    echo "== analyzing $NPZ"
    python -u /scratch/wxu/dh-spatial/scripts/probing/analyze.py --data "$NPZ" --out "$OUT" 2>&1 | tail -10
fi

echo "PHASE 1a v2 DONE"
