#!/bin/bash
#SBATCH --job-name=pg-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=backend/logs/train_%x_%j.out
#SBATCH --error=backend/logs/train_%x_%j.err
#SBATCH --mail-type=END,FAIL

# ============================================================
# PilotGuard — Single Profile Training (H200 GPU)
#
# Trains one specific dataset profile.
#
# Usage:
#   sbatch --job-name=pg-nthu   scripts/slurm_train_profile.sh nthu_drowsiness
#   sbatch --job-name=pg-uta    scripts/slurm_train_profile.sh uta_drowsiness
#   sbatch --job-name=pg-affect scripts/slurm_train_profile.sh affectnet_emotion
#   sbatch --job-name=pg-disfa  scripts/slurm_train_profile.sh disfa_au
# ============================================================

set -euo pipefail

PROFILE=${1:?"Usage: sbatch slurm_train_profile.sh <profile_name>"}

module load python/3.13.5 cuda/12.8.0

cd "$SLURM_SUBMIT_DIR/backend" || cd "$HOME/pilotguard/backend"
source venv/bin/activate
mkdir -p logs

echo "============================================================"
echo "Training profile: $PROFILE"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo "============================================================"

# Stage 1: Pre-compute features for this profile
python -m scripts.train_hpc --profile "$PROFILE" --stage features

# Stage 2: Train head
python -m scripts.train_hpc --profile "$PROFILE" --stage dinov2

echo "============================================================"
echo "Profile $PROFILE completed: $(date)"
echo "============================================================"
