# PilotGuard Data Directory

> **This directory is git-ignored.** Raw data, processed data, and synthetic data are never committed.

## Datasets

| Dataset | Description | Status | License |
|---------|-------------|--------|---------|
| **NTHU-DDD** | NTHU Driver Drowsiness Detection - 36 subjects, IR + visible light | Pending access request | Academic |
| **UTA-RLDD** | Real-Life Drowsiness Dataset - 60 subjects, 3 levels | Pending access request | Academic |
| **300W** | 300 Faces in-the-Wild - 68-landmark annotations | Public download | Research |
| **AffectNet** | 1M+ facial images, 8 emotion labels + valence/arousal | Pending access request | Academic |
| **DISFA** | Denver Intensity of Spontaneous Facial Actions - 12 AUs | Pending access request | Academic |
| **BP4D** | Binghamton-Pittsburgh 4D - spontaneous expressions + AUs | Pending access request | Academic |

## Directory Structure

```
data/
├── raw/           # Original downloaded datasets (untouched)
├── processed/     # Cleaned, normalized, split datasets
│   └── YYYY-MM-DD/  # Versioned by processing date
└── synthetic/     # Augmented / generated data
```

## Data Processing Log

| Date | Action | Dataset | Details |
|------|--------|---------|---------|
| *No entries yet* | | | |

## Rules

1. Never commit raw data or processed data to git
2. Document every transformation in this file
3. Use subject-stratified splits (seed=42) - no subject overlap between train/val/test
4. Version processed data by date folder
5. Validate data shape and distributions before training
