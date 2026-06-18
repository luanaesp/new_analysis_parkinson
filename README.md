# Leakage-Audited Benchmark for Voice-Based Parkinson's Disease Detection

Reproducible code for the analyses in "An Interpretable, Leakage-Audited Pipeline and Honest Cross-Cohort Benchmark for Voice-Based Detection of Parkinson's Disease."

The pipeline establishes an honest, externally validated benchmark across multiple public PD-voice cohorts, with explicit auditing for data leakage and batch-effect confounding. It is not a deployable diagnostic tool; it is a methodological/empirical benchmark and a reusable auditing protocol.

## What the code does

1. Reproduces within-cohort Leave-One-Subject-Out (LOSO) performance (798 features) with bootstrap CIs, and Wilson CIs for sensitivity/specificity.
2. Harmonises three cohorts into a shared acoustic core (19 features).
3. Leave-one-laboratory-out external validation (+ pairwise cross-cohort) with CIs.
4. Dataset/laboratory predictability test (batch-effect confound).
5. Label-permutation test (empirical p-value).
6. Model-agnostic permutation importance.
7. Probability averaging vs. majority voting.
8. Demographic table (sex per cohort/class; age where available).
9. Sakar-2013 vs. Sakar-2019 acoustic-distance de-duplication / sensitivity.
10. Leaky k-fold vs. honest LOSO comparison (quantifies leakage inflation).
11. Feature-provenance table for the 798 native features.
12. All figures + results_summary.json.

## Data (publicly available)

* Istanbul PD Classification Dataset (UCI 470) and Parkinson Speech Dataset (UCI 301)
* Parkinson Dataset with Replicated Acoustic Features (Naranjo et al., 2016)
* Mobile-assisted cohort (Carron et al., 2021; Mendeley Data)

Place the input files in the repository root (or edit the `PATHS` block at the top of `run_all_analyses.py`): `UNIFIED_RECONSTRUCTED (1).csv`, `PD-Dataset.csv`, `train_data.txt`, `test_data.txt`.

## How to run

```bash
pip install -r requirements.txt
python run_all_analyses.py
```

Outputs are written to `./outputs/`.

## Reproducibility

* Fixed random seed: 42
* Commit used to produce the reported results: `f7c00e5f0138dc6c80d9e5802d5b58957c0b0764`
* Environment used for the reported results: Python 3.14, scikit-learn 1.8.0, imbalanced-learn 0.14.1, NumPy 2.4.2, pandas 3.0.1 (16-core workstation, ~3 min runtime).
* Headline numbers: within-cohort LOSO AUC = 0.875 (95% CI 0.832-0.911); leave-one-laboratory-out AUC = 0.650 (95% CI 0.555-0.738); laboratory predictability = 100% (chance 64%); label-permutation null AUC = 0.502 (p = 0.005).

## License

Code released under the MIT License (see LICENSE).
