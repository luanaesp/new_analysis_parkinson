# -*- coding: utf-8 -*-
"""
PD Speech Classification – LOSO (subject-level) + SMOTE + Multi-view
+ Leak/Confounding Diagnostics Pack (runs before training)

Primary reporting: SUBJECT-LEVEL metrics (aggregate per subject, mean probability).
"""

import os
import warnings
import time
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve, ConfusionMatrixDisplay,
    average_precision_score, balanced_accuracy_score, matthews_corrcoef,
)
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning

from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_STATE = 42

TARGET_CANDIDATES = ["label", "class", "target", "y", "pd", "diagnosis"]
GROUP_CANDIDATES = ["subject_id", "subject", "id", "speaker", "patient_id", "name", "recording", "filename"]
DATASET_CANDIDATES = ["dataset_source", "dataset", "source", "corpus", "study", "origin"]
RECORDING_CANDIDATES = ["recording_id", "recording", "file", "filename", "utterance_id", "sample_id"]

def choose_column(cands, cols):
    lower = {c.lower(): c for c in cols}
    for k in cands:
        if k in cols:
            return k
        if k.lower() in lower:
            return lower[k.lower()]
    return None

def safe_smote_fit_resample(X, y, smote_ratio=1.0, max_k=5):
    cnt0, cnt1 = np.sum(y == 0), np.sum(y == 1)
    n_min = min(cnt0, cnt1)
    if n_min < 2:
        return X, y
    k = max(1, min(max_k, n_min - 1))
    maj = 1 if cnt1 > cnt0 else 0
    min_cl = 1 - maj
    n_maj = max(cnt0, cnt1)
    target_min = int(smote_ratio * n_maj)
    sm = SMOTE(sampling_strategy={min_cl: target_min}, k_neighbors=k, random_state=RANDOM_STATE)
    return sm.fit_resample(X, y)

def probs_softvote(list_of_prob_arrays, weights=None):
    P = np.vstack(list_of_prob_arrays)
    if weights is None:
        return P.mean(axis=0)
    w = np.array(weights).reshape(-1, 1)
    return (P * w).sum(axis=0) / np.sum(w)

def evaluate(y_true, y_prob, threshold=0.5, label="Model"):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    if len(np.unique(y_true)) >= 2:
        roc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
    else:
        roc = np.nan
        ap = np.nan

    rpt = classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=True)
    rec0 = rpt.get("0", {}).get("recall", np.nan)
    rec1 = rpt.get("1", {}).get("recall", np.nan)

    print(
        f"\n[{label}] Acc={acc:.4f} | BalAcc={bal:.4f} | F1(w)={f1w:.4f} | "
        f"ROC AUC={roc:.4f} | AP={ap:.4f} | MCC={mcc:.4f} | Recall0={rec0:.4f} | Recall1={rec1:.4f}"
    )
    return dict(model=label, accuracy=acc, balanced_accuracy=bal, f1_weighted=f1w,
                roc_auc=roc, ap=ap, mcc=mcc, recall_0=rec0, recall_1=rec1)

def aggregate_subject_level(subj, y_true, y_prob, agg="mean"):
    df_ = pd.DataFrame({"subj": subj, "y": y_true, "p": y_prob})
    y_nunique = df_.groupby("subj")["y"].nunique()
    if (y_nunique > 1).any():
        y_subj = df_.groupby("subj")["y"].agg(lambda s: int(s.mode().iloc[0]))
    else:
        y_subj = df_.groupby("subj")["y"].first().astype(int)

    p_subj = df_.groupby("subj")["p"].mean() if agg == "mean" else df_.groupby("subj")["p"].median()
    return y_subj.values, p_subj.values, y_subj.index.values

def bootstrap_ci_metric(y_true, y_prob, metric_fn, n_boot=2000, seed=42):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_fn(y_true[idx], y_prob[idx]))
    if not vals:
        return (np.nan, np.nan)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def plot_confusion_matrix(y_true, y_pred, out_path, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    disp.plot(cmap="Blues", ax=ax)
    ax.set_title(title)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_roc_pr(y_true, y_prob, out_roc, out_pr, title_prefix):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    fig = plt.figure(figsize=(7.5, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} – ROC"); plt.legend(loc="lower right")
    fig.savefig(out_roc, dpi=300, bbox_inches="tight"); plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig = plt.figure(figsize=(7.5, 6))
    plt.plot(recall, precision, lw=2, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{title_prefix} – Precision-Recall"); plt.legend(loc="lower left")
    fig.savefig(out_pr, dpi=300, bbox_inches="tight"); plt.close(fig)

def build_preprocessor(num_cols, cat_cols, X_tr_df):
    low_card = [c for c in cat_cols if X_tr_df[c].nunique() <= 20] if cat_cols else []
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if low_card:
        transformers.append(("cat", cat_pipe, low_card))
    return ColumnTransformer(transformers=transformers, remainder="drop"), low_card

def make_base_models(class_weight=None, light=False):
    lr = LogisticRegression(max_iter=4000, solver="liblinear", class_weight=class_weight, random_state=RANDOM_STATE)
    svm = SVC(kernel="rbf", probability=True, gamma="scale", C=1.0,
              class_weight=class_weight, random_state=RANDOM_STATE, cache_size=1000)
    if light:
        return {"LR": lr}
    rf = RandomForestClassifier(n_estimators=300, n_jobs=1, class_weight=class_weight, random_state=RANDOM_STATE)
    return {"LR": lr, "SVM": svm, "RF": rf}

def load_data(input_path, force_global_subject_id=True):
    print(f"Lendo: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Shape bruto: {df.shape}")

    y_col = choose_column(TARGET_CANDIDATES, df.columns)
    g_col = choose_column(GROUP_CANDIDATES, df.columns)
    d_col = choose_column(DATASET_CANDIDATES, df.columns)
    r_col = choose_column(RECORDING_CANDIDATES, df.columns)

    if y_col is None or g_col is None:
        raise ValueError(f"Missing required columns. target={y_col} group={g_col}")

    df = df.dropna(subset=[y_col]).copy()
    df[g_col] = df[g_col].astype(str)

    if d_col is not None:
        df[d_col] = df[d_col].astype(str).fillna("UNKNOWN")
    if force_global_subject_id and (d_col is not None):
        df[g_col] = df[d_col].astype(str) + "::" + df[g_col].astype(str)

    meta_drop = ["dataset_source","recording_id","ID__replicated","label_original","class_pd_speech","Status_replicated"]
    drop_set = {c for c in meta_drop if c in df.columns}
    drop_set -= {y_col, g_col}
    if d_col is not None: drop_set -= {d_col}
    if r_col is not None: drop_set -= {r_col}

    non_feats = {y_col, g_col} | drop_set
    if d_col is not None: non_feats.add(d_col)
    if r_col is not None: non_feats.add(r_col)

    for c in df.columns:
        if c in non_feats:
            continue
        if df[c].dtype == "object":
            cand = pd.to_numeric(df[c], errors="coerce")
            if cand.notna().mean() > 0.8:
                df[c] = cand
            else:
                non_feats.add(c)

    all_nan_cols = [c for c in df.columns if c not in non_feats and df[c].isna().all()]
    if all_nan_cols:
        df = df.drop(columns=all_nan_cols)
        print(f"Removidas colunas 100% NaN: {len(all_nan_cols)}")

    keep = [c for c in df.columns if c not in non_feats] + [y_col, g_col]
    if d_col is not None: keep.append(d_col)
    if r_col is not None: keep.append(r_col)
    seen = set(); keep2 = []
    for c in keep:
        if c in df.columns and c not in seen:
            keep2.append(c); seen.add(c)
    df = df[keep2].copy()

    feat_cols = [c for c in df.columns if c not in [y_col, g_col] + ([d_col] if d_col else []) + ([r_col] if r_col else [])]
    print(f"Shape após limpeza: {df.shape}")
    print(f"Colunas de features utilizadas: {len(feat_cols)}")
    print(f"Alvo: {y_col} | Grupo: {g_col} | Dataset: {d_col} | Recording: {r_col}")
    return df, y_col, g_col, d_col, r_col

def dataset_integrity_report(df, y_col, g_col, out_dir, d_col=None, r_col=None, force_global_subject_id=True):
    diag_dir = ensure_dir(os.path.join(out_dir, "leakage_checks"))
    print("\n" + "=" * 80)
    print("INTEGRITY REPORT (dataset audit)")
    print("=" * 80)

    feat_cols = [c for c in df.columns if c not in [y_col, g_col] + ([d_col] if d_col else []) + ([r_col] if r_col else [])]
    print(f"Rows: {len(df)} | Subjects: {df[g_col].nunique()} | Features: {len(feat_cols)}")
    print("\nLabel distribution:")
    print(df[y_col].astype(int).value_counts().to_string())

    subj_nuniq = df.groupby(g_col)[y_col].nunique()
    n_inconsistent = int((subj_nuniq > 1).sum())
    if n_inconsistent:
        print(f"\n⚠️ Subjects with inconsistent label: {n_inconsistent}")
        print(subj_nuniq[subj_nuniq > 1].head(30).to_string())
    else:
        print("\n✅ All subjects have consistent label.")

    if d_col is not None:
        print("\nBy dataset_source:")
        tmp = df.groupby(d_col).agg(
            n_samples=(y_col, "size"),
            n_subjects=(g_col, "nunique"),
            prevalence_pos=(y_col, lambda s: float(np.mean(s.astype(int) == 1))),
        ).sort_values("n_samples", ascending=False)
        print(tmp.to_string())
        tmp.to_csv(os.path.join(diag_dir, "summary_by_dataset_source.csv"))

    dup_rows = int(df.duplicated().sum())
    print(f"\nExact duplicate rows: {dup_rows}")
    if dup_rows:
        df[df.duplicated(keep=False)].head(2000).to_csv(os.path.join(diag_dir, "duplicate_rows_head.csv"), index=False)

    X = df[feat_cols].copy()
    X = X.apply(lambda s: s.fillna(s.median()) if pd.api.types.is_numeric_dtype(s) else s.fillna("NA"))
    feat_hash = pd.util.hash_pandas_object(X, index=False)
    dup_feat = int(feat_hash.duplicated().sum())
    print(f"Duplicate FEATURE vectors: {dup_feat}")
    if dup_feat:
        idx = feat_hash[feat_hash.duplicated(keep=False)].index[:2000]
        meta_cols = [y_col, g_col] + ([d_col] if d_col else []) + ([r_col] if r_col else [])
        df.loc[idx, meta_cols].to_csv(os.path.join(diag_dir, "duplicate_feature_vectors_meta_head.csv"), index=False)

    print(f"\nReports saved to: {diag_dir}")
    print("=" * 80 + "\n")

def dataset_predictability_test(df, y_col, g_col, d_col, out_dir, fast_checks=True):
    print("\n" + "=" * 80)
    print("DATASET PREDICTABILITY TEST (confounding)")
    print("=" * 80)

    if d_col is None:
        print("No dataset_source column; skipping.")
        return

    ds = df[d_col].astype(str).values
    ds_vals, ds_enc = np.unique(ds, return_inverse=True)
    if len(ds_vals) < 2:
        print("Only one dataset_source detected; skipping.")
        return

    feat_cols = [c for c in df.columns if c not in [y_col, g_col, d_col]]
    X_df = df[feat_cols].copy()
    groups = df[g_col].astype(str).values

    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    logo = LeaveOneGroupOut()
    pred_all = np.zeros(len(df), dtype=int)
    pmax_all = np.zeros(len(df), dtype=float)

    clf = LogisticRegression(max_iter=4000, solver="liblinear", random_state=RANDOM_STATE)

    for i, (tr, te) in enumerate(logo.split(X_df, ds_enc, groups)):
        X_tr_df, X_te_df = X_df.iloc[tr], X_df.iloc[te]
        y_tr = ds_enc[tr]

        pre, _ = build_preprocessor(num_cols, cat_cols, X_tr_df)
        X_tr = np.nan_to_num(np.asarray(pre.fit_transform(X_tr_df), dtype=np.float32), nan=0.0)
        X_te = np.nan_to_num(np.asarray(pre.transform(X_te_df), dtype=np.float32), nan=0.0)

        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_te)
        pred = np.argmax(p, axis=1)

        pred_all[te] = pred
        pmax_all[te] = np.max(p, axis=1)

        if fast_checks and i >= 200:
            break

    acc = float((pred_all == ds_enc).mean())
    print(f"Dataset predictability (LOSO by subject) – Accuracy ≈ {acc:.4f}")
    print("If very high (e.g., >0.85), features carry strong dataset signature (confounding risk).")

    diag_dir = ensure_dir(os.path.join(out_dir, "leakage_checks"))
    pd.DataFrame({"dataset_true": ds, "dataset_pred": ds_vals[pred_all], "pmax": pmax_all}).to_csv(
        os.path.join(diag_dir, "dataset_predictability_predictions.csv"), index=False
    )

def label_permutation_test(df, y_col, g_col, d_col, out_dir, seed=42):
    print("\n" + "=" * 80)
    print("LABEL PERMUTATION TEST (sanity/leakage)")
    print("=" * 80)

    dfp = df.copy()
    rng = np.random.default_rng(seed)

    if d_col is not None and d_col in dfp.columns:
        def shuffle_group(s):
            arr = np.array(s.to_numpy(), copy=True)
            rng.shuffle(arr)
            return arr
        dfp[y_col] = dfp.groupby(d_col)[y_col].transform(shuffle_group).astype(int)
        print(f"Labels permuted within each {d_col}.")
    else:
        arr = np.array(dfp[y_col].to_numpy(), copy=True)
        rng.shuffle(arr)
        dfp[y_col] = arr.astype(int)
        print("Labels permuted globally.")

    cols = [c for c in dfp.columns if c not in [y_col, g_col] + ([d_col] if d_col else [])]
    y = dfp[y_col].astype(int).values
    groups = dfp[g_col].astype(str).values
    X_df = dfp[cols].copy()

    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    logo = LeaveOneGroupOut()
    probs_all, y_all = [], []

    for tr, te in logo.split(X_df, y, groups):
        X_tr_df, X_te_df = X_df.iloc[tr], X_df.iloc[te]
        y_tr, y_te = y[tr], y[te]

        pre, _ = build_preprocessor(num_cols, cat_cols, X_tr_df)
        X_tr = np.nan_to_num(np.asarray(pre.fit_transform(X_tr_df), dtype=np.float32), nan=0.0)
        X_te = np.nan_to_num(np.asarray(pre.transform(X_te_df), dtype=np.float32), nan=0.0)

        selector = VarianceThreshold(threshold=0.01)
        X_tr = selector.fit_transform(X_tr)
        X_te = selector.transform(X_te)

        clf = LogisticRegression(max_iter=4000, solver="liblinear", random_state=RANDOM_STATE)
        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_te)[:, 1]

        probs_all.extend(p.tolist())
        y_all.extend(y_te.tolist())

    m = evaluate(np.array(y_all), np.array(probs_all), label="Permutation sanity (sample-level)")
    diag_dir = ensure_dir(os.path.join(out_dir, "leakage_checks"))
    pd.DataFrame([m]).to_csv(os.path.join(diag_dir, "perm_test_metrics_sample.csv"), index=False)
    return m

def process_fold(fold_data):
    (i, (tr, te), groups, X_df, y, num_cols, cat_cols, n_subj, t0,
     balance_mode, smote_ratio, max_smote_k,
     use_original, use_pca, use_lda, pca_variance,
     n_jobs_local) = fold_data

    if (i == 0) or ((i + 1) % 10 == 0):
        print(f"[Fold {i+1}/{n_subj}] test subject: {groups[te][0]} | elapsed {(time.time()-t0)/60:.1f} min")

    X_tr_df, X_te_df = X_df.iloc[tr], X_df.iloc[te]
    y_tr, y_te = y[tr], y[te]

    pre, _ = build_preprocessor(num_cols, cat_cols, X_tr_df)
    X_tr = np.nan_to_num(np.asarray(pre.fit_transform(X_tr_df), dtype=np.float32), nan=0.0)
    X_te = np.nan_to_num(np.asarray(pre.transform(X_te_df), dtype=np.float32), nan=0.0)

    selector = VarianceThreshold(threshold=0.01)
    X_tr = selector.fit_transform(X_tr)
    X_te = selector.transform(X_te)

    class_weight = "balanced" if balance_mode == "weights" else None
    base_models = make_base_models(class_weight=class_weight, light=False)

    def fit_predict(Xtr, Xte, ytr):
        pv = []
        for _, clf in base_models.items():
            clf.fit(Xtr, ytr)
            pv.append(clf.predict_proba(Xte)[:, 1])
        return probs_softvote(pv)

    all_probs = []

    if use_original:
        X_tr_o, y_tr_o = X_tr, y_tr
        if balance_mode == "smote":
            X_tr_o, y_tr_o = safe_smote_fit_resample(X_tr_o, y_tr_o, smote_ratio=smote_ratio, max_k=max_smote_k)
        all_probs.append(fit_predict(X_tr_o, X_te, y_tr_o))

    if use_pca and X_tr.shape[1] > 1:
        pca = PCA(n_components=pca_variance, svd_solver="full", random_state=RANDOM_STATE)
        X_tr_p = pca.fit_transform(X_tr)
        X_te_p = pca.transform(X_te)
        y_tr_p = y_tr.copy()
        if balance_mode == "smote":
            X_tr_p, y_tr_p = safe_smote_fit_resample(X_tr_p, y_tr_p, smote_ratio=smote_ratio, max_k=max_smote_k)
        all_probs.append(fit_predict(X_tr_p, X_te_p, y_tr_p))

    if use_lda and len(np.unique(y_tr)) >= 2 and X_tr.shape[1] > 1:
        lda = LDA(solver="eigen", shrinkage="auto")
        X_tr_l = lda.fit_transform(X_tr, y_tr)
        X_te_l = lda.transform(X_te)
        y_tr_l = y_tr.copy()
        if balance_mode == "smote":
            X_tr_l, y_tr_l = safe_smote_fit_resample(X_tr_l, y_tr_l, smote_ratio=smote_ratio, max_k=max_smote_k)
        all_probs.append(fit_predict(X_tr_l, X_te_l, y_tr_l))

    p_final = probs_softvote(all_probs) if all_probs else np.zeros(len(y_te), dtype=float)
    return (groups[te].tolist(), y_te.tolist(), p_final.tolist())

def run_loso(df, y_col, g_col, out_dir,
             balance_mode="smote", smote_ratio=1.0, max_smote_k=5,
             use_original=True, use_pca=True, use_lda=False, pca_variance=0.95,
             n_jobs=-1, bootstrap_n=2000, plot=True):

    feat_cols = [c for c in df.columns if c not in [y_col, g_col]]
    y = df[y_col].astype(int).values
    groups = df[g_col].astype(str).values
    X_df = df[feat_cols].copy()

    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    logo = LeaveOneGroupOut()
    n_subj = len(np.unique(groups))
    print(f"Iniciando LOSO com {n_subj} sujeitos...")
    print(f"Config: Original={use_original}, PCA={use_pca}, LDA={use_lda} | Balance={balance_mode}")
    t0 = time.time()

    fold_data = []
    for i, (tr, te) in enumerate(logo.split(X_df, y, groups)):
        fold_data.append((i, (tr, te), groups, X_df, y, num_cols, cat_cols, n_subj, t0,
                          balance_mode, smote_ratio, max_smote_k,
                          use_original, use_pca, use_lda, pca_variance,
                          n_jobs))

    results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(process_fold)(d) for d in fold_data)

    subj_all, y_true_all, proba_all = [], [], []
    for subj_te, y_te, p_final in results:
        subj_all.extend(subj_te); y_true_all.extend(y_te); proba_all.extend(p_final)

    subj_all = np.array(subj_all)
    y_true_all = np.array(y_true_all, dtype=int)
    proba_all = np.array(proba_all, dtype=float)

    sample_metrics = evaluate(y_true_all, proba_all, label="Sample-level (LOSO)")
    y_subj, p_subj, _ = aggregate_subject_level(subj_all, y_true_all, proba_all, agg="mean")
    subject_metrics = evaluate(y_subj, p_subj, label="Subject-level (LOSO, mean prob)")

    roc_ci = bootstrap_ci_metric(y_subj, p_subj, roc_auc_score, n_boot=bootstrap_n, seed=RANDOM_STATE)
    ap_ci = bootstrap_ci_metric(y_subj, p_subj, average_precision_score, n_boot=bootstrap_n, seed=RANDOM_STATE)
    print(f"\nSubject-level ROC AUC 95% CI: {roc_ci[0]:.4f} – {roc_ci[1]:.4f}")
    print(f"Subject-level AP      95% CI: {ap_ci[0]:.4f} – {ap_ci[1]:.4f}")

    ensure_dir(out_dir)
    pd.DataFrame([sample_metrics]).to_csv(os.path.join(out_dir, "metrics_sample.csv"), index=False)
    pd.DataFrame([subject_metrics]).to_csv(os.path.join(out_dir, "metrics_subject.csv"), index=False)

    if plot:
        plot_confusion_matrix(y_true_all, (proba_all >= 0.5).astype(int),
                              os.path.join(out_dir, "confusion_matrix_sample_loso.png"),
                              "Confusion Matrix – Sample-level (LOSO)")
        plot_confusion_matrix(y_subj, (p_subj >= 0.5).astype(int),
                              os.path.join(out_dir, "confusion_matrix_subject_loso.png"),
                              "Confusion Matrix – Subject-level (LOSO)")
        plot_roc_pr(y_true_all, proba_all,
                    os.path.join(out_dir, "roc_sample_loso.png"),
                    os.path.join(out_dir, "pr_sample_loso.png"),
                    "Sample-level (LOSO)")
        plot_roc_pr(y_subj, p_subj,
                    os.path.join(out_dir, "roc_subject_loso.png"),
                    os.path.join(out_dir, "pr_subject_loso.png"),
                    "Subject-level (LOSO)")

    return {"sample": sample_metrics, "subject": subject_metrics, "subject_ci": {"roc_auc_95ci": roc_ci, "ap_95ci": ap_ci}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--no-diagnostics", action="store_true")
    ap.add_argument("--no-global-subject", action="store_true")
    ap.add_argument("--balance", default="smote", choices=["smote", "weights", "none"])
    ap.add_argument("--no-pca", action="store_true")
    ap.add_argument("--use-lda", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out)
    df, y_col, g_col, d_col, r_col = load_data(args.input, force_global_subject_id=(not args.no_global_subject))

    if not args.no_diagnostics:
        dataset_integrity_report(df, y_col, g_col, out_dir, d_col=d_col, r_col=r_col,
                                 force_global_subject_id=(not args.no_global_subject))
        if d_col is not None:
            dataset_predictability_test(df, y_col, g_col, d_col, out_dir, fast_checks=True)

    run_loso(df, y_col, g_col, out_dir,
             balance_mode=args.balance,
             use_original=True,
             use_pca=(not args.no_pca),
             use_lda=args.use_lda,
             bootstrap_n=args.bootstrap,
             n_jobs=args.jobs,
             plot=True)

    if not args.no_diagnostics:
        label_permutation_test(df, y_col, g_col, d_col=d_col, out_dir=out_dir, seed=RANDOM_STATE)

    print("\\nDone. Outputs in:", os.path.abspath(out_dir))

if __name__ == "__main__":
    main()
