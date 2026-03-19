#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="seman"
PYTHON_VERSION="3.10"
CUDA_TAG="cu124"
TORCH_VERSION="2.4.0"
TORCHVISION_VERSION="0.19.0"
TORCHAUDIO_VERSION="2.4.0"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
SCATTER_INDEX="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

echo "============================================"
echo "  Creating conda env: ${ENV_NAME}"
echo "  Python ${PYTHON_VERSION} | PyTorch ${TORCH_VERSION} | CUDA 12.4"
echo "============================================"

# ---------- 1. Create conda environment ----------
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# ---------- 2. Install CUDA toolkit via conda ----------
conda install nvidia/label/cuda-12.4.0::cuda -y

# ---------- 3. Install PyTorch ----------
pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
    --index-url "${TORCH_INDEX}"

# ---------- 4. Install torch-scatter (needs special index) ----------
pip install torch-scatter -f "${SCATTER_INDEX}"

# ---------- 5. Install remaining dependencies ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "${SCRIPT_DIR}/requirements.txt"

# ---------- 6. System libraries needed by open3d / vtk ----------
if command -v apt &>/dev/null; then
    apt install -y libx11-6 libgl1 libxrender1 2>/dev/null || \
        echo "[WARN] Could not install system libs. Run with sudo if needed."
fi

echo ""
echo "============================================"
echo "  Environment '${ENV_NAME}' is ready!"
echo "  Activate with:  conda activate ${ENV_NAME}"
echo "============================================"
