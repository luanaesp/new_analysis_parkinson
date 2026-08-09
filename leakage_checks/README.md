> ## ⚠️ SUPERSEDED — DO NOT CITE THESE NUMBERS
>
> This folder is an **earlier, exploratory pipeline** that is **not** the one reported in the
> manuscript. It uses a different model (multi-view ensembling over Original + PCA + optional
> LDA views) and therefore produces **different predictions** from the pipeline of record.
>
> Concretely, `confusion_matrix_subject_loso.png` in this folder reports
> TN=73, FP=31, FN=26, TP=202 (sensitivity 88.60%, specificity 70.19%), whereas the
> manuscript's subject-level LOSO result is
> TN=72, FP=32, FN=37, TP=191 (sensitivity 83.77%, specificity 69.23%).
> Both describe the same 332 subjects (104 HC / 228 PD); the difference is the model, not the data.
>
> `confusion_matrix_sample_loso.png` (TN=217, FP=95, FN=95, TP=589, N=996) is a
> **recording-level** result and is not comparable to any subject-level number in the paper.
>
> **The pipeline of record is [`../run_all_analyses.py`](../run_all_analyses.py)**, whose outputs
> live in [`../outputs/`](../outputs/) and match the manuscript exactly. This folder is retained
> only for provenance of the initial leakage investigation.

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
Links for the datasets used in the project:
“Parkinson's Speech with Multiple Types of Sound Recordings - UCI Machine Learning Repository”  ; “Parkinson Dataset with replicated acoustic features - UCI Machine Learning Repository“ ; “Parkinson's Disease Classification - UCI Machine Learning Repository“

## Reproducibility notes 

- Preprocessing is fit **only on training folds**
- SMOTE (when enabled) is applied **only on training folds**
