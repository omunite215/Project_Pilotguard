#!/bin/bash
#SBATCH --job-name=pg-features
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=backend/logs/features_%j.out
#SBATCH --error=backend/logs/features_%j.err
#SBATCH --mail-type=END,FAIL

# ============================================================
# PilotGuard — DINOv2 Feature Pre-computation Only
#
# Run this first to cache features, then submit training jobs.
# Features are saved to data/features/<dataset>/dinov2_features.npz
#
# Usage: sbatch scripts/slurm_features_only.sh
# ============================================================

set -euo pipefail

module load python/3.13.5 cuda/12.8.0

cd "$SLURM_SUBMIT_DIR/backend" || cd "$HOME/pilotguard/backend"
source venv/bin/activate
mkdir -p logs

echo "============================================================"
echo "DINOv2 Feature Pre-computation"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"
echo "============================================================"

python -m scripts.train_hpc --profile all --stage features

echo "============================================================"
echo "Feature extraction completed: $(date)"
echo "============================================================"
