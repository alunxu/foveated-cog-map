#!/usr/bin/env bash
# CS503 Project — One-time environment setup on SCITAS
# Usage: bash setup_env.sh
set -euo pipefail

eval "$(conda shell.bash hook)"

ENV_NAME="cs503_project"

# Create conda environment (separate from NanoFM homework env)
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists. Activating..."
else
    echo "Creating conda environment '${ENV_NAME}'..."
    conda create -n "${ENV_NAME}" python=3.10 -y
fi

conda activate "${ENV_NAME}"

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install project as editable package
pip install -e .

# Register Jupyter kernel
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "CS503 Project (${ENV_NAME})"

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "  Activate with: conda activate ${ENV_NAME}"
echo "  torchrun at:   $(which torchrun)"
echo "=========================================="
