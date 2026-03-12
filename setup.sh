#!/bin/bash
# Setup script for Pixel Art Generator on remote GPU instances (e.g., Lightning AI, RunPod)

echo "🔄 Starting environment setup..."

# 1. Install project and its dependencies (including huggingface-cli)
echo "📦 Installing Python dependencies..."
pip install -e .

# 2. Unzip dataset if needed
echo "📂 Checking for dataset..."
if [ -f "dataset32.zip" ]; then
    echo "📦 Extracting dataset32.zip..."
    mkdir -p dataset
    unzip -qo dataset32.zip -d .
    echo "✅ Dataset extracted."
else
    echo "⚠️ dataset32.zip not found. Please upload it if you plan to run prepare_dataset.py"
fi

# 3. Download the FLUX GGUF Model using Hugging Face CLI
echo "🧠 Downloading FLUX model from Hugging Face..."
MODEL_DIR="models"
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DIR/flux1-schnell-Q4_K_S.gguf" ]; then
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='city96/FLUX.1-schnell-gguf',
    filename='flux1-schnell-Q4_K_S.gguf',
    local_dir='$MODEL_DIR',
    local_dir_use_symlinks=False
)
"
    echo "✅ Model downloaded successfully."
else
    echo "✅ Model already exists, skipping download."
fi

echo "🚀 Setup complete! You can now run:"
echo "1. python src/prepare_dataset.py"
echo "2. python src/train.py"
