#!/bin/bash
#SBATCH --job-name=pg-train-all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=backend/logs/train_all_%j.out
#SBATCH --error=backend/logs/train_all_%j.err
#SBATCH --mail-type=END,FAIL

# ============================================================
# PilotGuard — Full Training Pipeline (H200 GPU)
#
# Runs all stages:
#   1. Pre-compute DINOv2 features for all datasets
#   2. Train XGBoost geometric classifier (CPU)
#   3. Train DINOv2 classification heads (GPU)
#
# Usage: sbatch scripts/slurm_train_all.sh
# ============================================================

set -euo pipefail

# Load HPC modules
module load python/3.13.5 cuda/12.8.0

# Navigate to backend directory
cd "$SLURM_SUBMIT_DIR/backend" || cd "$HOME/pilotguard/backend"

# Activate virtual environment
source venv/bin/activate

# Create log directory
mkdir -p logs

# Print environment info
echo "============================================================"
echo "PilotGuard Training Pipeline"
echo "============================================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "Partition:  $SLURM_JOB_PARTITION"
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "CUDA:       $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not available")')"
echo "PyTorch:    $(python -c 'import torch; print(torch.__version__)')"
echo "bf16:       $(python -c 'import torch; print(torch.cuda.is_bf16_supported())' 2>/dev/null || echo 'N/A')"
echo "Start:      $(date)"
echo "============================================================"

# Run full pipeline
python -m scripts.train_hpc --profile all --stage all

echo "============================================================"
echo "Completed: $(date)"
echo "============================================================"
