#!/bin/bash
set -e
# Inner script invoked from the LOSO analysis pod.
# Runs the python script in-place and writes results to PVC.

cd /scratch/wxu/dh-spatial
pip install --quiet 'numpy<2' scikit-learn || true
python3 /scratch/wxu/dh-spatial/scripts/cluster/_loso_5cond_inner.py
