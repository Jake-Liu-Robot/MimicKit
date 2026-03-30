#!/bin/bash
# Server setup script for RunPod / Lambda / Vast.ai with 4x RTX 4090
# Run once after SSH into server: bash scripts/setup_server.sh

set -e

echo "=== MimicKit Server Setup ==="
echo "Start time: $(date)"

# 1. System check
echo ""
echo "--- GPU Check ---"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPUs detected: $GPU_COUNT"
if [ "$GPU_COUNT" -lt 4 ]; then
    echo "WARNING: Expected 4 GPUs, found $GPU_COUNT. Adjust batch scripts accordingly."
fi

# 2. Install Conda if not present
if ! command -v conda &> /dev/null; then
    echo ""
    echo "--- Installing Miniconda ---"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    rm /tmp/miniconda.sh
else
    eval "$(conda shell.bash hook)"
fi

# 3. Create conda environment
echo ""
echo "--- Creating conda environment: mimickit ---"
conda create -n mimickit python=3.8 -y
conda activate mimickit

# 4. Install Isaac Gym
echo ""
echo "--- Installing Isaac Gym ---"
cd IsaacGym_Preview_4_Package/isaacgym/python
pip install -e .
cd ../../..

# 5. Install MimicKit requirements
echo ""
echo "--- Installing MimicKit requirements ---"
pip install -r requirements.txt

# 6. Install TensorBoard for logging
pip install tensorboard

# 7. Create output directory
mkdir -p output

# 8. Verify installation
echo ""
echo "--- Verifying installation ---"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -c "import isaacgym; print('Isaac Gym: OK')"

# 9. Verify motion data exists
echo ""
echo "--- Checking motion data ---"
REQUIRED_FILES=(
    "data/motions/humanoid/humanoid_walk.pkl"
    "data/motions/humanoid/humanoid_spinkick.pkl"
    "data/motions/humanoid/humanoid_run.pkl"
    "data/motions/humanoid/humanoid_jog.pkl"
    "data/motions/humanoid/humanoid_getup_facedown.pkl"
    "data/motions/humanoid/humanoid_getup_faceup.pkl"
    "data/assets/humanoid/humanoid.xml"
    "data/datasets/dataset_humanoid_locomotion.yaml"
    "data/datasets/exp7_loco_getup.yaml"
    "data/datasets/exp5_diverse_motions.yaml"
)
ALL_OK=true
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "WARNING: Some files are missing. Download assets from:"
    echo "  https://1sfu-my.sharepoint.com/:u:/g/personal/xbpeng_sfu_ca/EclKq9pwdOBAl-17SogfMW0Bved4sodZBQ_5eZCiz9O--w?e=bqXBaa"
    echo "Extract into data/ directory."
fi

# 10. Make scripts executable
chmod +x scripts/run_batch1.sh scripts/run_batch2.sh scripts/run_tests.sh

echo ""
echo "=== Setup complete: $(date) ==="
echo ""
echo "Next steps:"
echo "  1. If motion data is missing, download and extract assets"
echo "  2. Run Batch 1:  conda activate mimickit && bash scripts/run_batch1.sh"
echo "  3. Run Batch 2:  bash scripts/run_batch2.sh"
echo "  4. Run Tests:    bash scripts/run_tests.sh"
echo "  5. TensorBoard:  tensorboard --logdir=output --port=6006 --bind_all"
