# PilotGuard — HPC Training Guide (NEU Discovery)

## Overview

```
Local PC (Windows) ──(MobaXterm)──> NEU Discovery login node ──(SLURM)──> H200 GPU node
```

**What gets trained:**
- XGBoost geometric drowsiness classifier (CPU, fast)
- DINOv2 classification heads × 4 datasets (GPU, warmup + cosine schedule)

---

## Step-by-Step Instructions

### Step 1: Connect to HPC

1. Open **MobaXterm**
2. Click your saved SSH session (or: Session → SSH → `login.discovery.neu.edu`, user: `patel.omm`)
3. Enter password when prompted

**Expected output:**
```
Last login: Wed Apr  2 10:00:00 2026 from ...
[patel.omm@login-00 ~]$
```

---

### Step 2: Create project directory (first time only)

```bash
mkdir -p /patel.omm/pilotguard
cd /work/patel.omm/pilotguard
```

**Expected output:** No output (silent success).

---

### Step 3: Transfer files from local PC

Use MobaXterm's **left panel file browser**:

1. In the left panel, navigate to `/work/patel.omm/pilotguard/`
2. On your Windows PC, **zip the code** (PowerShell):
   ```powershell
   Compress-Archive -Path D:\PilotGuard\backend, D:\PilotGuard\configs, D:\PilotGuard\scripts -DestinationPath D:\PilotGuard\pilotguard_code.zip -Force
   ```
3. Drag `D:\PilotGuard\pilotguard_code.zip` into MobaXterm left panel
4. Extract on HPC:
   ```bash
   cd /work/patel.omm/pilotguard
   unzip -o pilotguard_code.zip
   rm pilotguard_code.zip
   ```

**Expected output:**
```
Archive:  pilotguard_code.zip
   creating: backend/
   creating: configs/
   creating: scripts/
  inflating: backend/pyproject.toml
  inflating: configs/training_config.yaml
  ...
```

5. **Transfer data the same way.** The raw datasets must be at these paths on HPC:
   ```
   /work/patel.omm/pilotguard/data/raw/nthu-ddd/          # NTHU-DDD images
   /work/patel.omm/pilotguard/data/raw/affectnet/          # AffectNet images
   /work/patel.omm/pilotguard/data/raw/disfa/              # DISFA images
   /work/patel.omm/pilotguard/data/processed/uta_frames/   # UTA extracted frames
   /work/patel.omm/pilotguard/data/processed/*_manifest.csv  # All 4 manifests
   ```

   > If datasets are already somewhere on Discovery (e.g. `/scratch/` or a shared folder),
   > create symlinks instead: `ln -s /path/to/nthu-ddd data/raw/nthu-ddd`

6. Verify the data structure:
   ```bash
   ls data/processed/*_manifest.csv
   head -2 data/processed/nthu_ddd_manifest.csv
   ls data/raw/
   ```

**Expected output:**
```
data/processed/affectnet_manifest.csv  data/processed/disfa_manifest.csv
data/processed/nthu_ddd_manifest.csv   data/processed/uta_rldd_manifest.csv

path,label,subject_id,split,quality_score
raw/nthu-ddd/drowsy/001_glasses_sleepyCombination_1000_drowsy.jpg,drowsy,nthu_001,test,1.0000

affectnet  disfa  nthu-ddd
```

---

### Step 4: Set up Python environment (first time only)

```bash
cd /work/patel.omm/pilotguard/backend
module load python/3.13.5 cuda/12.8.0
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

**Expected output** (last few lines):
```
Successfully installed ... torch-2.5.0 ... xgboost-2.1.0 ...
```

Verify GPU + PyTorch:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

**Expected output:**
```
PyTorch 2.5.x, CUDA: False
```
> CUDA shows False on the login node — that's expected. GPUs are only on compute nodes.

---

### Step 5: Create logs directory

```bash
mkdir -p /work/patel.omm/pilotguard/backend/logs
```

---

### Step 6: Submit training jobs

Make SLURM scripts executable:
```bash
chmod +x /work/patel.omm/pilotguard/scripts/*.sh
```

#### Option A: Full pipeline (recommended first time, ~6-8 hours)

```bash
cd /work/patel.omm/pilotguard
sbatch scripts/slurm_train_all.sh
```

**Expected output:**
```
Submitted batch job 1234567
```

#### Option B: Step-by-step with dependencies (best for monitoring)

```bash
cd /work/patel.omm/pilotguard

# Step 1: Extract DINOv2 features first (~2-3 hours)
JOB1=$(sbatch --parsable scripts/slurm_features_only.sh)
echo "Feature extraction job: $JOB1"

# Step 2: Submit training jobs that wait for features to finish
sbatch --dependency=afterok:$JOB1 --job-name=pg-nthu   scripts/slurm_train_profile.sh nthu_drowsiness
sbatch --dependency=afterok:$JOB1 --job-name=pg-uta    scripts/slurm_train_profile.sh uta_drowsiness
sbatch --dependency=afterok:$JOB1 --job-name=pg-affect scripts/slurm_train_profile.sh affectnet_emotion
sbatch --dependency=afterok:$JOB1 --job-name=pg-disfa  scripts/slurm_train_profile.sh disfa_au
```

**Expected output:**
```
Feature extraction job: 1234567
Submitted batch job 1234568
Submitted batch job 1234569
Submitted batch job 1234570
Submitted batch job 1234571
```

#### Option C: Single profile only

```bash
sbatch --job-name=pg-nthu scripts/slurm_train_profile.sh nthu_drowsiness
```

---

### Step 7: Monitor jobs

#### Check job status
```bash
squeue -u patel.omm
```

**Expected output:**
```
JOBID   PARTITION  NAME        STATE    TIME     NODES  NODELIST
1234567 gpu        pg-features RUNNING  0:15:32  1      gpu-0042
1234568 gpu        pg-nthu     PENDING  0:00:00  1      (Dependency)
```

States: `PENDING` → waiting for resources/dependencies, `RUNNING` → active, `COMPLETED` → done

#### Watch live training output
```bash
tail -f backend/logs/train_all_1234567.out
```

**Expected output during feature extraction:**
```
============================================================
PilotGuard Training Pipeline
============================================================
Job ID:     1234567
Node:       gpu-0042
GPU:        NVIDIA H200 80GB HBM3, 80 GB
CUDA:       NVIDIA H200 80GB HBM3
PyTorch:    2.5.0
bf16:       True
Start:      Wed Apr  2 14:30:00 EDT 2026
============================================================

============================================================
STAGE 1: Pre-computing DINOv2 features
============================================================
Pre-computing DINOv2 features for nthu_drowsiness...
Loading DINOv2 model: dinov2_vits14 on cuda
Processed 500/66521 images
Processed 1000/66521 images
...
Saved 64000 features to /work/patel.omm/pilotguard/data/features/nthu_ddd/dinov2_features.npz
DINOv2 feature extraction: 1200.0s (55.4 img/s)
```

**Expected output during DINOv2 head training:**
```
============================================================
Training DINOv2 head: nthu_drowsiness
  Dataset:  nthu_ddd
  Task:     binary (2 classes)
  Head:     fusion
  Epochs:   50 (warmup: 5)
  Batch:    128, LR: 3.00e-04
============================================================
============================================================
Training config:
  Model params:    55042
  Train samples:   25000
  Val samples:     5500
  Epochs:          50 (warmup: 5)
  Batch size:      128
  Steps/epoch:     196
  Peak LR:         3.00e-04
  Device:          cuda
  Mixed precision: True (torch.bfloat16)
============================================================
[warmup] Epoch 1/50: loss=0.6821, val_f1=0.5234, val_acc=0.5400, lr=1.34e-05
[warmup] Epoch 2/50: loss=0.6543, val_f1=0.6012, val_acc=0.6100, lr=5.88e-05
[warmup] Epoch 3/50: loss=0.5987, val_f1=0.6845, val_acc=0.7000, lr=1.18e-04
[warmup] Epoch 4/50: loss=0.5234, val_f1=0.7523, val_acc=0.7600, lr=2.02e-04
[warmup] Epoch 5/50: loss=0.4567, val_f1=0.8012, val_acc=0.8100, lr=2.94e-04
[cosine] Epoch 10/50: loss=0.2345, val_f1=0.8934, val_acc=0.9000, lr=2.78e-04
[cosine] Epoch 15/50: loss=0.1876, val_f1=0.9123, val_acc=0.9200, lr=2.34e-04
[cosine] Epoch 20/50: loss=0.1543, val_f1=0.9234, val_acc=0.9300, lr=1.78e-04
...
Best val F1: 0.9300 at epoch 25

Test Results:
  Accuracy:  0.9250
  F1:        0.9260
  Precision: 0.9270
  Recall:    0.9250
```

**Expected output at the end:**
```
============================================================
ALL TRAINING COMPLETE — Total time: 245.3 min
============================================================
```

---

### Step 8: Check results after completion

```bash
# Check if job finished
squeue -u patel.omm  # Should be empty when done

# Check for errors
grep -i "error\|traceback" backend/logs/train_all_*.err

# List trained models
find backend/models -name "*.pt" -o -name "*.json" | head -20
```

**Expected output:**
```
backend/models/drowsiness/geometric_xgb_v1.json
backend/models/drowsiness/geometric_xgb_v1_metadata.json
backend/models/nthu_ddd/nthu_drowsiness_best.pt
backend/models/nthu_ddd/nthu_drowsiness_metadata.json
backend/models/uta_rldd/uta_drowsiness_best.pt
backend/models/uta_rldd/uta_drowsiness_metadata.json
backend/models/affectnet/affectnet_emotion_best.pt
backend/models/affectnet/affectnet_emotion_metadata.json
backend/models/disfa/disfa_au_best.pt
backend/models/disfa/disfa_au_metadata.json
```

Check model metrics:
```bash
cat backend/models/nthu_ddd/nthu_drowsiness_metadata.json
```

**Expected output:**
```json
{
  "profile": "nthu_drowsiness",
  "dataset": "nthu_ddd",
  "task": "binary",
  "head": "fusion",
  "num_classes": 2,
  "label_map": {"alert": 0, "drowsy": 1},
  "epochs_trained": 25,
  "best_val_f1": 0.93,
  "test_metrics": {
    "accuracy": 0.925,
    "f1": 0.926,
    "precision": 0.927,
    "recall": 0.925
  },
  "train_samples": 25000,
  "val_samples": 5500,
  "test_samples": 5500
}
```

---

### Step 9: Download models to local PC

In MobaXterm left panel, navigate to `/work/patel.omm/pilotguard/backend/models/` and drag the folder to your local `D:\PilotGuard\backend\models\`.

Or zip and download:
```bash
cd /work/patel.omm/pilotguard
zip -r pilotguard_models.zip backend/models/
```
Then drag `pilotguard_models.zip` from MobaXterm left panel to your PC.

---

### Step 10: TensorBoard (optional)

On HPC:
```bash
source /work/patel.omm/pilotguard/backend/venv/bin/activate
tensorboard --logdir=/work/patel.omm/pilotguard/backend/runs --port=6006 --bind_all &
```

On your local PC (new PowerShell window):
```powershell
ssh -L 6006:localhost:6006 patel.omm@login.discovery.neu.edu
```

Then open http://localhost:6006 in your browser.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | `source venv/bin/activate` |
| `CUDA out of memory` | Reduce `batch_size` in `configs/training_config.yaml` (try 64) |
| `No GPU detected` | You're on a login node. GPUs are only available via `sbatch` |
| `Permission denied` on scripts | `chmod +x scripts/*.sh` |
| Job stuck in `PENDING` | `squeue -u patel.omm` — may be waiting for GPU resources |
| `face_landmarker.task not found` | Verify `backend/models/face_landmarker.task` exists |
| H200 not available | Change `--gres=gpu:h200:1` to `--gres=gpu:a100:1` in SLURM scripts |
| `No training data extracted` | Raw dataset images missing. Check `ls data/raw/nthu-ddd/` |
| `sbatch: error: Batch job submission failed` | Check script line endings: `dos2unix scripts/*.sh` |

---

## File Reference

| File | Purpose |
|------|---------|
| `scripts/slurm_train_all.sh` | SLURM: Full pipeline (features + XGBoost + all DINOv2 heads) |
| `scripts/slurm_train_profile.sh` | SLURM: Single dataset profile |
| `scripts/slurm_features_only.sh` | SLURM: DINOv2 feature extraction only |
| `backend/scripts/train_hpc.py` | Python: Training orchestrator |
| `configs/training_config.yaml` | All hyperparameters + per-dataset profiles |
| `backend/src/ml/train_dinov2_head.py` | Warmup + cosine LR training loop |
| `backend/src/ml/train_geometric.py` | XGBoost training |

## Training Schedule

| Profile | Dataset | Train | Epochs | Warmup | LR | Est. Time (H200) |
|---------|---------|-------|--------|--------|----|-------------------|
| nthu_drowsiness | NTHU-DDD | 28,672 | 50 | 5 | 3e-4 | ~20 min |
| uta_drowsiness | UTA-RLDD | 125,490 | 40 | 3 | 2e-4 | ~45 min |
| affectnet_emotion | AffectNet | 19,328 | 60 | 5 | 2e-4 | ~15 min |
| disfa_au | DISFA | 36,080 | 50 | 5 | 2e-4 | ~25 min |

> Feature extraction: ~2-3 hours. Head training: ~1-2 hours total. XGBoost: ~30-60 min.
