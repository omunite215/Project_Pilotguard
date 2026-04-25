#!/bin/bash
#SBATCH --job-name=pilotguard-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# ===== PilotGuard SLURM Training Template =====
# Usage: sbatch scripts/slurm_template.sh <training_script> [args]
# Example: sbatch scripts/slurm_template.sh src.ml.train_fusion --config configs/training_config.yaml

set -euo pipefail

# Load modules
module load python/3.13.5 cuda/12.8.0

# Activate environment
source venv/bin/activate

# Create logs directory if needed
mkdir -p logs

# Print job info
echo "=============================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "=============================="

# Run training
SCRIPT=${1:-src.ml.train_fusion}
shift || true
python -m "$SCRIPT" --config configs/training_config.yaml "$@"

echo "=============================="
echo "End: $(date)"
echo "=============================="
