# PD Speech Classification (LOSO by subject) + Leak/Confounding Diagnostics

This repository runs a **Leave-One-Subject-Out (LOSO)** evaluation for PD speech classification,
with optional SMOTE, multi-view ensembling (Original + PCA (+ optional LDA)), and a diagnostics pack
to detect data leakage / dataset confounding.

## What this project produces

Primary (publication-facing) metrics:
- **Subject-level** metrics (primary): Acc, BalAcc, F1(w), ROC AUC, AP (AUPRC), MCC, Recall0/1
- 95% bootstrap CI for **ROC AUC** and **AP** at the **subject level**
- Confusion matrices and ROC/PR curves for both sample-level and subject-level

Diagnostics (runs before training):
- Dataset integrity report (label consistency per subject, duplicates, high-cardinality ID-like columns)
- Subject-ID collision checks across datasets (when datasets were merged)
- Dataset predictability test (can features predict which dataset a sample came from?)
- Label permutation sanity test (AUC should drop near chance)

## Recommended environment

- Python: **3.10–3.12** recommended (Python 3.14 is bleeding-edge and can trigger incompatibilities).
- Install:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Data

Place your unified CSV at:

`data/raw/UNIFIED_RECONSTRUCTED.csv`

Expected columns (case-insensitive):
- `label` (0/1)
- `subject_id` (for LOSO grouping)
- `dataset_source` (optional but strongly recommended)

## Run

```bash
python scripts/run_loso_with_diagnostics.py --input data/raw/UNIFIED_RECONSTRUCTED.csv --out outputs
```

## Reporting in the paper

Use **Subject-level (LOSO, mean prob)** as the **primary** result.
Sample-level is secondary (more optimistic due to repeated samples per subject).

## Reproducibility notes

- Preprocessing is fit **only on training folds**
- SMOTE (when enabled) is applied **only on training folds**
