# -*- coding: utf-8 -*-
"""
PD Speech Classification – LOSO (subject-level) + SMOTE + Multi-view
+ Leak/Confounding Diagnostics Pack (RUNS BEFORE TRAINING)

O que este script faz (além do seu pipeline):
1)  Auditoria do dataset (UNIFIED_RECONSTRUCTED):
   - Checa se existe dataset_source (ou tenta inferir)
   - Checa colisão de subject_id entre datasets (mesmo id aparecendo em fontes diferentes)
   - Checa sujeitos com label inconsistente
   - Checa duplicatas exatas de linhas / duplicatas de features
   - Checa se há "features que são praticamente IDs" (alta cardinalidade)
   - Resumo por dataset_source (n sujeitos, n amostras, prevalência)

2)  Testes de confounding (dataset bias):
   - "Dataset Predictability Test": treina um classificador para prever dataset_source a partir das features.
     Se der muito alto → seu modelo pode estar aprendendo “qual dataset é”, não “PD”.

3)  Teste de sanity/leakage:
   - Permutation test: embaralha labels (mantendo proporção por dataset_source) e re-roda versão rápida.
     Se AUC continuar alto → cheiro de leakage/artefato.

4)  Opção de "subject_id global seguro":
   - Se existir dataset_source, você pode forçar um subject_id global = dataset_source + "::" + subject_id
   - Isso elimina colisão de ids entre datasets diferentes (muito importante quando você unifica bases).

Obs: Para economizar tempo nos testes, existe configuração FAST_CHECKS.
"""

import os
import re
import warnings
import time
import numpy as np
import pandas as pd

# matplotlib sem GUI
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
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from sklearn.inspection import permutation_importance
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning

from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed

# ==============================
# Warnings: não “matar tudo”, só o barulho comum
# ==============================
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ==============================
# CONFIG
# ==============================
INPUT_PATH = r"C:\Users\mdant\Downloads\UNIFIED_RECONSTRUCTED.csv"
OUT_DIR = "outputs_loso"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# Balanceamento
BALANCE_MODE = "smote"  # "smote" | "weights" | "none"
SMOTE_RATIO = 1.0
MAX_SMOTE_K = 5

# Execução
VERBOSE_EVERY = 10
LIMIT_SUBJECTS = None
N_JOBS = -1

# Multi-view
USE_ORIGINAL_VIEW = True
USE_PCA_VIEW = True
USE_LDA_VIEW = False
PCA_VARIANCE = 0.95

# Plots
PLOT_CONFUSION_MATRIX = True
PLOT_ROC_PR_CURVES = True

# EDA (exploratório)
PLOT_EDA_TSNE = False
PLOT_EDA_PCA = False
PLOT_EDA_LDA = False

# Feature importance "publicável"
DO_PERMUTATION_IMPORTANCE = True
PERM_IMPORTANCE_N_REPEATS = 10
PERM_IMPORTANCE_TOPN = 30
PERM_IMPORTANCE_MODEL = "LR"  # "LR" ou "RF"

# Bootstrap CI (subject-level)
BOOTSTRAP_N = 2000

# ==============================
# NOVO: Leak/Confounding tests
# ==============================
RUN_INTEGRITY_CHECKS = True
RUN_DATASET_PREDICTABILITY_TEST = True
RUN_LABEL_PERMUTATION_TEST = True

FAST_CHECKS = True
# FAST_CHECKS=True deixa os testes mais leves:
# - reduz modelos e folds (ainda informativo)
# - não roda permutation importance nesses testes

# Se existir dataset_source: crie subject_id global (recomendado quando unifica bases)
FORCE_GLOBAL_SUBJECT_ID = True

# Nome/possíveis nomes
TARGET_CANDIDATES = ["label", "class", "target", "y", "pd", "diagnosis"]
GROUP_CANDIDATES = ["subject_id", "subject", "id", "speaker", "patient_id", "name", "recording", "filename"]
DATASET_CANDIDATES = ["dataset_source", "dataset", "source", "corpus", "study", "origin"]
RECORDING_CANDIDATES = ["recording_id", "recording", "file", "filename", "utterance_id", "sample_id"]

# ==============================
# Utils
# ==============================
def choose_column(cands, cols):
    lower = {c.lower(): c for c in cols}
    for k in cands:
        if k in cols:
            return k
        if k.lower() in lower:
            return lower[k.lower()]
    return None


def safe_smote_fit_resample(X, y):
    """SMOTE com k vizinhos seguro e fallback quando a minoria é muito pequena."""
    cnt0, cnt1 = np.sum(y == 0), np.sum(y == 1)
    n_min = min(cnt0, cnt1)
    if n_min < 2:
        return X, y

    k = max(1, min(MAX_SMOTE_K, n_min - 1))
    maj = 1 if cnt1 > cnt0 else 0
    min_cl = 1 - maj
    n_maj = max(cnt0, cnt1)
    target_min = int(SMOTE_RATIO * n_maj)
    sampling_strategy = {min_cl: target_min}

    sm = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k, random_state=RANDOM_STATE)
    return sm.fit_resample(X, y)


def probs_softvote(list_of_prob_arrays, weights=None):
    P = np.vstack(list_of_prob_arrays)  # (n_models, n_samples)
    if weights is None:
        return P.mean(axis=0)
    w = np.array(weights).reshape(-1, 1)
    return (P * w).sum(axis=0) / np.sum(w)


def evaluate(y_true, y_prob, threshold=0.5, label="Model"):
    """Avaliação robusta sem warnings quando houver classe única."""
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

    return dict(
        model=label,
        accuracy=acc,
        balanced_accuracy=bal,
        f1_weighted=f1w,
        roc_auc=roc,
        ap=ap,
        mcc=mcc,
        recall_0=rec0,
        recall_1=rec1,
    )


def aggregate_subject_level(subj, y_true, y_prob, agg="mean"):
    df_ = pd.DataFrame({"subj": subj, "y": y_true, "p": y_prob})

    # rótulo do sujeito (se inconsistente: modo)
    y_nunique = df_.groupby("subj")["y"].nunique()
    if (y_nunique > 1).any():
        y_subj = df_.groupby("subj")["y"].agg(lambda s: int(s.mode().iloc[0]))
    else:
        y_subj = df_.groupby("subj")["y"].first().astype(int)

    if agg == "mean":
        p_subj = df_.groupby("subj")["p"].mean()
    elif agg == "median":
        p_subj = df_.groupby("subj")["p"].median()
    else:
        raise ValueError("agg deve ser 'mean' ou 'median'")

    return y_subj.values, p_subj.values, y_subj.index.values


def bootstrap_ci_metric(y_true, y_prob, metric_fn, n_boot=2000, seed=42):
    """IC 95% via bootstrap (reamostra sujeitos) evitando reamostras mono-classe."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

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

    if len(vals) == 0:
        return (np.nan, np.nan)

    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ==============================
# Plot helpers
# ==============================
def save_path(name):
    return os.path.join(OUT_DIR, name)


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    disp.plot(cmap="Blues", ax=ax)
    ax.set_title(title)
    fig.savefig(save_path(filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(y_true, y_prob, title_prefix="LOSO", roc_file="roc.png", pr_file="pr.png"):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        print(f"{title_prefix}: apenas uma classe presente — ROC/PR não calculável.")
        return

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    fig = plt.figure(figsize=(7.5, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} – ROC")
    plt.legend(loc="lower right")
    fig.savefig(save_path(roc_file), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig = plt.figure(figsize=(7.5, 6))
    plt.plot(recall, precision, lw=2, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} – Precision-Recall")
    plt.legend(loc="lower left")
    fig.savefig(save_path(pr_file), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==============================
# Data loading + cleaning
# ==============================
def load_data():
    print(f"Lendo: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"Shape bruto: {df.shape}")

    y_col = choose_column(TARGET_CANDIDATES, df.columns)
    g_col = choose_column(GROUP_CANDIDATES, df.columns)
    d_col = choose_column(DATASET_CANDIDATES, df.columns)
    r_col = choose_column(RECORDING_CANDIDATES, df.columns)

    if y_col is None or g_col is None:
        raise ValueError(f"Colunas não encontradas — alvo: {y_col} | grupo: {g_col}")

    df.dropna(subset=[y_col], inplace=True)
    df[g_col] = df[g_col].astype(str)

    if d_col is not None:
        df[d_col] = df[d_col].astype(str).fillna("UNKNOWN")

    # Se você unificou datasets, isso é IMPORTANTÍSSIMO:
    # transforma subject_id em “global id” para evitar colisões entre datasets
    if FORCE_GLOBAL_SUBJECT_ID and (d_col is not None):
        df[g_col] = df[d_col].astype(str) + "::" + df[g_col].astype(str)

    meta_and_label_cols_to_drop = [
        # cuidado: só dropa se existir
        "dataset_source",
        "recording_id",
        "ID__replicated",
        "label_original",
        "class_pd_speech",
        "Status_replicated",
        "f1_uci2013_train",
        "f28_uci2013_train",
        "f29_uci2013_train",
        "f1_uci2013_test",
        "f28_uci2013_test",
        "f29_uci2013_test",
    ]

    # não dropa g_col/y_col/d_col/r_col
    drop_set = set([c for c in meta_and_label_cols_to_drop if c in df.columns])
    drop_set -= {y_col, g_col}
    if d_col is not None:
        drop_set -= {d_col}
    if r_col is not None:
        drop_set -= {r_col}

    non_feats = {y_col, g_col}.union(drop_set)
    if d_col is not None:
        non_feats.add(d_col)
    if r_col is not None:
        non_feats.add(r_col)

    # tenta converter objects numéricas (mas sem transformar IDs/strings úteis)
    for c in df.columns:
        if c in non_feats:
            continue
        if df[c].dtype == "object":
            cand = pd.to_numeric(df[c], errors="coerce")
            if cand.notna().mean() > 0.8:
                df[c] = cand
            else:
                non_feats.add(c)

    # remove colunas 100% NaN (somente features)
    all_nan_cols = [c for c in df.columns if c not in non_feats and df[c].isna().all()]
    if all_nan_cols:
        df = df.drop(columns=all_nan_cols)
        print(f"Removidas colunas 100% NaN: {len(all_nan_cols)}")

    # mantém features + (y,g,d,r)
    keep_cols = [c for c in df.columns if (c not in non_feats)] + [y_col, g_col]
    if d_col is not None:
        keep_cols.append(d_col)
    if r_col is not None:
        keep_cols.append(r_col)

    # remove duplicatas na lista mantendo ordem
    seen = set()
    keep_cols2 = []
    for c in keep_cols:
        if c not in seen and c in df.columns:
            keep_cols2.append(c)
            seen.add(c)

    df = df[keep_cols2].copy()

    feature_cols = [c for c in df.columns if c not in [y_col, g_col] + ([d_col] if d_col else []) + ([r_col] if r_col else [])]
    print(f"Shape após limpeza: {df.shape}")
    print(f"Colunas de features utilizadas: {len(feature_cols)}")
    print(f"Alvo: {y_col} | Grupo: {g_col} | Dataset: {d_col} | Recording: {r_col}")

    return df, y_col, g_col, d_col, r_col


# ==============================
# Integrity / Leakage diagnostics
# ==============================
def dataset_integrity_report(df, y_col, g_col, d_col=None, r_col=None):
    diag_dir = os.path.join(OUT_DIR, "leakage_checks")
    os.makedirs(diag_dir, exist_ok=True)

    def wprint(s):
        print(s)

    wprint("\n" + "=" * 80)
    wprint("INTEGRITY REPORT (dataset audit)")
    wprint("=" * 80)

    wprint(f"Rows: {len(df)} | Subjects: {df[g_col].nunique()} | Features: {len([c for c in df.columns if c not in [y_col,g_col] + ([d_col] if d_col else []) + ([r_col] if r_col else [])])}")

    # 1) label distribution
    vc = df[y_col].astype(int).value_counts(dropna=False)
    wprint("\nLabel distribution:")
    wprint(vc.to_string())

    # 2) subject label consistency
    subj_nuniq = df.groupby(g_col)[y_col].nunique()
    n_inconsistent = int((subj_nuniq > 1).sum())
    if n_inconsistent > 0:
        wprint(f"\n⚠️ Subjects com label inconsistente: {n_inconsistent}")
        bad = subj_nuniq[subj_nuniq > 1].index.astype(str).tolist()[:30]
        wprint(f"Exemplos (até 30): {bad}")
    else:
        wprint("\n✅ Todos os subjects têm label consistente (por grupo).")

    # 3) dataset_source summary
    if d_col is not None:
        wprint("\nPor dataset_source:")
        tmp = df.groupby(d_col).agg(
            n_samples=(y_col, "size"),
            n_subjects=(g_col, "nunique"),
            prevalence_pos=(y_col, lambda s: float(np.mean(s.astype(int) == 1))),
        ).sort_values("n_samples", ascending=False)
        wprint(tmp.to_string())
        tmp.to_csv(os.path.join(diag_dir, "summary_by_dataset_source.csv"))

        # 4) subject_id collision check (ANTES do global id, se você quiser ver colisão)
        # Aqui não dá pra “desfazer” se você já fez global id; então checamos pelo padrão "ds::id"
        if FORCE_GLOBAL_SUBJECT_ID:
            # tenta recuperar "id local"
            local_id = df[g_col].astype(str).str.split("::", n=1, expand=True)
            if local_id.shape[1] == 2:
                df_local = df[[d_col, y_col]].copy()
                df_local["local_subject_id"] = local_id[1].astype(str)
                # colisão: mesmo local_subject_id aparecendo em mais de um dataset_source
                col = df_local.groupby("local_subject_id")[d_col].nunique()
                collisions = col[col > 1].sort_values(ascending=False)
                wprint(f"\nChecagem de colisão (local subject_id em múltiplos datasets): {len(collisions)}")
                if len(collisions) > 0:
                    wprint("⚠️ Exemplo de ids locais colidindo (até 30):")
                    wprint(collisions.head(30).to_string())
                    collisions.head(2000).to_csv(os.path.join(diag_dir, "subject_id_collisions_local.csv"))
                else:
                    wprint("✅ Nenhuma colisão detectada em ids locais (pelo parse ds::id).")
            else:
                wprint("\n⚠️ Não consegui parsear ds::id para checar colisão local.")
        else:
            # sem global: colisão direta (mesmo subject_id em múltiplos datasets)
            col = df.groupby(g_col)[d_col].nunique()
            collisions = col[col > 1].sort_values(ascending=False)
            wprint(f"\nChecagem de colisão (subject_id em múltiplos datasets): {len(collisions)}")
            if len(collisions) > 0:
                wprint("⚠️ Exemplo de ids colidindo (até 30):")
                wprint(collisions.head(30).to_string())
                collisions.head(2000).to_csv(os.path.join(diag_dir, "subject_id_collisions.csv"))
            else:
                wprint("✅ Nenhuma colisão detectada.")

    # 5) duplicates exact rows
    dup_rows = df.duplicated().sum()
    wprint(f"\nDuplicatas exatas de linha: {int(dup_rows)}")
    if dup_rows > 0:
        df[df.duplicated(keep=False)].head(2000).to_csv(os.path.join(diag_dir, "duplicate_rows_head.csv"), index=False)

    # 6) duplicates by feature vector (ignoring y/g/d/r)
    feat_cols = [c for c in df.columns if c not in [y_col, g_col] + ([d_col] if d_col else []) + ([r_col] if r_col else [])]
    # hashes: mais rápido que comparar float a float
    X = df[feat_cols].copy()
    # imputação simples para hash consistente
    X = X.apply(lambda s: s.fillna(s.median()) if pd.api.types.is_numeric_dtype(s) else s.fillna("NA"))
    feat_hash = pd.util.hash_pandas_object(X, index=False)
    dup_feat = feat_hash.duplicated().sum()
    wprint(f"Duplicatas de FEATURES (mesmo vetor de features): {int(dup_feat)}")
    if dup_feat > 0:
        idx = feat_hash[feat_hash.duplicated(keep=False)].index[:2000]
        df.loc[idx, [y_col, g_col] + ([d_col] if d_col else []) + ([r_col] if r_col else [])].to_csv(
            os.path.join(diag_dir, "duplicate_feature_vectors_meta_head.csv"), index=False
        )

    # 7) suspicious high-cardinality columns
    susp = []
    for c in feat_cols:
        if df[c].dtype == "object":
            nun = df[c].nunique(dropna=False)
            if nun > 0.9 * len(df):
                susp.append((c, nun))
    if susp:
        wprint("\n⚠️ Colunas suspeitas (object com cardinalidade ~N; podem ser IDs mascarados):")
        for c, nun in susp[:30]:
            wprint(f"  - {c}: nunique={nun}")
        pd.DataFrame(susp, columns=["column", "nunique"]).to_csv(os.path.join(diag_dir, "high_cardinality_object_cols.csv"), index=False)
    else:
        wprint("\n✅ Nenhuma coluna object com cardinalidade ~N detectada nas features.")

    wprint(f"\nRelatórios salvos em: {diag_dir}")
    wprint("=" * 80 + "\n")


def build_preprocessor(num_cols, cat_cols, X_tr_df):
    low_card = [c for c in cat_cols if X_tr_df[c].nunique() <= 20] if cat_cols else []

    num_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if low_card:
        transformers.append(("cat", cat_pipe, low_card))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return pre, low_card


def make_base_models(class_weight=None, light=False):
    cw = class_weight if (BALANCE_MODE != "smote") else None

    lr = LogisticRegression(
        max_iter=4000,
        solver="liblinear",
        penalty="l2",
        class_weight=cw,
        random_state=RANDOM_STATE,
    )

    # ✅ pequeno ajuste no SVM: cache_size e class_weight; (probability=True é caro, mas você já usa)
    svm = SVC(
        kernel="rbf",
        probability=True,
        gamma="scale",
        C=1.0,
        class_weight=cw,
        random_state=RANDOM_STATE,
        cache_size=1000,  # ajuda estabilidade/perf
    )

    if light:
        return {"LR": lr}  # para checks rápidos

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=1,
        class_weight=cw,
        random_state=RANDOM_STATE,
    )

    return {"LR": lr, "SVM": svm, "RF": rf}


def process_fold(fold_data):
    (
        i,
        (tr, te),
        groups,
        X_df,
        y,
        num_cols,
        cat_cols,
        n_subj,
        t0,
        light_models,
        do_perm_importance,
    ) = fold_data

    subj_id = groups[te][0]
    if (i == 0) or ((i + 1) % VERBOSE_EVERY == 0):
        print(f"[Fold {i+1}/{n_subj}] sujeito teste: {subj_id} | elapsed {(time.time() - t0) / 60:.1f} min")

    X_tr_df, X_te_df = X_df.iloc[tr], X_df.iloc[te]
    y_tr, y_te = y[tr], y[te]

    pre, low_card = build_preprocessor(num_cols, cat_cols, X_tr_df)
    X_tr = pre.fit_transform(X_tr_df)
    X_te = pre.transform(X_te_df)

    X_tr = np.asarray(X_tr, dtype=np.float32)
    X_te = np.asarray(X_te, dtype=np.float32)
    X_tr = np.nan_to_num(X_tr, nan=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0)

    selector = VarianceThreshold(threshold=0.01)
    X_tr = selector.fit_transform(X_tr)
    X_te = selector.transform(X_te)

    all_probs = []

    class_weight = "balanced" if BALANCE_MODE == "weights" else None
    base_models = make_base_models(class_weight=class_weight, light=light_models)

    # -------- (1) Original view --------
    if USE_ORIGINAL_VIEW:
        X_tr_o, y_tr_o = X_tr, y_tr
        if BALANCE_MODE == "smote":
            X_tr_o, y_tr_o = safe_smote_fit_resample(X_tr_o, y_tr_o)

        pv = []
        for _, clf in base_models.items():
            clf.fit(X_tr_o, y_tr_o)
            pv.append(clf.predict_proba(X_te)[:, 1])
        all_probs.append(probs_softvote(pv))

    # -------- (2) PCA view --------
    if USE_PCA_VIEW and X_tr.shape[1] > 1:
        pca = PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=RANDOM_STATE)
        X_tr_p = pca.fit_transform(X_tr)
        X_te_p = pca.transform(X_te)

        y_tr_p = y_tr.copy()
        if BALANCE_MODE == "smote":
            X_tr_p, y_tr_p = safe_smote_fit_resample(X_tr_p, y_tr_p)

        pv = []
        for _, clf in base_models.items():
            clf.fit(X_tr_p, y_tr_p)
            pv.append(clf.predict_proba(X_te_p)[:, 1])
        all_probs.append(probs_softvote(pv))

    # -------- (3) LDA view --------
    if USE_LDA_VIEW and len(np.unique(y_tr)) >= 2 and X_tr.shape[1] > 1:
        lda = LDA(solver="eigen", shrinkage="auto")
        X_tr_l = lda.fit_transform(X_tr, y_tr)
        X_te_l = lda.transform(X_te)

        y_tr_l = y_tr.copy()
        if BALANCE_MODE == "smote":
            X_tr_l, y_tr_l = safe_smote_fit_resample(X_tr_l, y_tr_l)

        pv = []
        for _, clf in base_models.items():
            clf.fit(X_tr_l, y_tr_l)
            pv.append(clf.predict_proba(X_te_l)[:, 1])
        all_probs.append(probs_softvote(pv))

    if len(all_probs) > 0:
        p_final = probs_softvote(all_probs)
    else:
        p_final = np.zeros(len(y_te), dtype=float)

    # (mantém o retorno do seu pipeline, mas desliga perm_importance em modos rápidos para evitar custo/bugs)
    perm_imp = None
    if do_perm_importance and DO_PERMUTATION_IMPORTANCE:
        # (mantém sua lógica original, mas só quando habilitado)
        try:
            # modelo representativo: LR na visão original
            X_tr_imp, y_tr_imp = X_tr, y_tr
            if BALANCE_MODE == "smote":
                X_tr_imp, y_tr_imp = safe_smote_fit_resample(X_tr_imp, y_tr_imp)
            clf_imp = LogisticRegression(
                max_iter=4000, solver="liblinear", class_weight=("balanced" if BALANCE_MODE == "weights" else None),
                random_state=RANDOM_STATE,
            )
            clf_imp.fit(X_tr_imp, y_tr_imp)
            if len(np.unique(y_te)) >= 2:
                r = permutation_importance(
                    clf_imp, X_te, y_te, n_repeats=PERM_IMPORTANCE_N_REPEATS,
                    random_state=RANDOM_STATE, scoring="roc_auc",
                )
                perm_imp = (r.importances_mean,)
        except Exception:
            perm_imp = None

    return (groups[te].tolist(), y_te.tolist(), p_final.tolist(), perm_imp)


def run_loso_parallel(df, y_col, g_col, light_models=False, do_perm_importance=True):
    cols = [c for c in df.columns if c not in [y_col, g_col]]
    y = df[y_col].astype(int).values
    groups = df[g_col].astype(str).values
    X_df = df[cols].copy()

    if LIMIT_SUBJECTS:
        keep = []
        for s in pd.unique(groups):
            keep.append(s)
            if len(keep) >= LIMIT_SUBJECTS:
                break
        mask = np.isin(groups, keep)
        X_df, y, groups = X_df.loc[mask], y[mask], groups[mask]

    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    logo = LeaveOneGroupOut()
    n_subj = len(np.unique(groups))
    print(f"Iniciando LOSO com {n_subj} sujeitos...")
    print(f"Config: Original={USE_ORIGINAL_VIEW}, PCA={USE_PCA_VIEW}, LDA={USE_LDA_VIEW} | Balance={BALANCE_MODE}")
    t0 = time.time()

    fold_data = []
    for i, (tr, te) in enumerate(logo.split(X_df, y, groups)):
        fold_data.append((i, (tr, te), groups, X_df, y, num_cols, cat_cols, n_subj, t0, light_models, do_perm_importance))

    results = Parallel(n_jobs=N_JOBS, verbose=10)(delayed(process_fold)(data) for data in fold_data)

    subj_all, y_true_all, proba_all = [], [], []
    perm_imps = []

    for subj_te, y_te, p_final, perm_imp in results:
        subj_all.extend(subj_te)
        y_true_all.extend(y_te)
        proba_all.extend(p_final)
        if perm_imp is not None:
            perm_imps.append(perm_imp)

    subj_all = np.array(subj_all)
    y_true_all = np.array(y_true_all, dtype=int)
    proba_all = np.array(proba_all, dtype=float)

    sample_metrics = evaluate(y_true_all, proba_all, threshold=0.5, label="Sample-level (LOSO)")

    y_subj, p_subj, subj_ids = aggregate_subject_level(subj_all, y_true_all, proba_all, agg="mean")
    subject_metrics = evaluate(y_subj, p_subj, threshold=0.5, label="Subject-level (LOSO, mean prob)")

    roc_ci = bootstrap_ci_metric(y_subj, p_subj, roc_auc_score, n_boot=BOOTSTRAP_N, seed=RANDOM_STATE)
    ap_ci = bootstrap_ci_metric(y_subj, p_subj, average_precision_score, n_boot=BOOTSTRAP_N, seed=RANDOM_STATE)
    print(f"\nSubject-level ROC AUC 95% CI: {roc_ci[0]:.4f} – {roc_ci[1]:.4f}")
    print(f"Subject-level AP      95% CI: {ap_ci[0]:.4f} – {ap_ci[1]:.4f}")

    if PLOT_CONFUSION_MATRIX:
        y_pred_sample = (proba_all >= 0.5).astype(int)
        plot_confusion_matrix(y_true_all, y_pred_sample, "Matriz de Confusão – Sample-level (LOSO)", "confusion_matrix_sample_loso.png")

        y_pred_subj = (p_subj >= 0.5).astype(int)
        plot_confusion_matrix(y_subj, y_pred_subj, "Matriz de Confusão – Subject-level (LOSO)", "confusion_matrix_subject_loso.png")

    if PLOT_ROC_PR_CURVES:
        plot_roc_pr(y_true_all, proba_all, "Sample-level (LOSO)", "roc_sample_loso.png", "pr_sample_loso.png")
        plot_roc_pr(y_subj, p_subj, "Subject-level (LOSO)", "roc_subject_loso.png", "pr_subject_loso.png")

    metrics_out = {
        "sample": sample_metrics,
        "subject": subject_metrics,
        "subject_ci": {"roc_auc_95ci": roc_ci, "ap_95ci": ap_ci},
    }
    pd.DataFrame([sample_metrics]).to_csv(save_path("metrics_sample.csv"), index=False)
    pd.DataFrame([subject_metrics]).to_csv(save_path("metrics_subject.csv"), index=False)

    return dict(
        subj_all=subj_all,
        y_true=y_true_all,
        y_prob=proba_all,
        y_subj=y_subj,
        p_subj=p_subj,
        metrics=metrics_out,
        perm_imps=perm_imps,
    )


# ==============================
# Confounding checks
# ==============================
def dataset_predictability_test(df, y_col, g_col, d_col):
    """
    Treina um modelo para prever dataset_source a partir das features.
    Se performance for alta, há risco de "dataset confounding".
    """
    diag_dir = os.path.join(OUT_DIR, "leakage_checks")
    os.makedirs(diag_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("DATASET PREDICTABILITY TEST (confounding)")
    print("=" * 80)

    # precisa de pelo menos 2 datasets
    if d_col is None:
        print("Sem dataset_source: não dá para rodar este teste.")
        return

    ds = df[d_col].astype(str).values
    # encode datasets para 0..K-1
    ds_vals, ds_enc = np.unique(ds, return_inverse=True)
    if len(ds_vals) < 2:
        print("Apenas 1 dataset_source detectado; teste não aplicável.")
        return

    # features: tudo exceto y, group, dataset
    feat_cols = [c for c in df.columns if c not in [y_col, g_col, d_col]]
    X_df = df[feat_cols].copy()
    groups = df[g_col].astype(str).values

    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    logo = LeaveOneGroupOut()
    t0 = time.time()
    probs_all = np.zeros(len(df), dtype=float)
    pred_all = np.zeros(len(df), dtype=int)

    # modelo simples e rápido
    clf = LogisticRegression(max_iter=4000, solver="liblinear", random_state=RANDOM_STATE)

    # vamos fazer LOSO por subject também (para não vazar sujeito entre train/test)
    for i, (tr, te) in enumerate(logo.split(X_df, ds_enc, groups)):
        X_tr_df, X_te_df = X_df.iloc[tr], X_df.iloc[te]
        y_tr, y_te = ds_enc[tr], ds_enc[te]

        pre, _ = build_preprocessor(num_cols, cat_cols, X_tr_df)
        X_tr = pre.fit_transform(X_tr_df)
        X_te = pre.transform(X_te_df)

        X_tr = np.nan_to_num(np.asarray(X_tr, dtype=np.float32), nan=0.0)
        X_te = np.nan_to_num(np.asarray(X_te, dtype=np.float32), nan=0.0)

        # multi-class: usa predict_proba e pega argmax
        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_te)
        pred = np.argmax(p, axis=1)

        pred_all[te] = pred
        # “probabilidade do acerto” (máx), só para diagnóstico
        probs_all[te] = np.max(p, axis=1)

        if FAST_CHECKS and (i >= 200):  # corta cedo para diagnóstico rápido
            break

    acc = (pred_all == ds_enc).mean()
    print(f"Dataset predictability (LOSO by subject) – Accuracy ≈ {acc:.4f}")
    print("Interpretação: se isso for MUITO alto (ex.: >0.85), seu modelo pode estar aprendendo assinatura do dataset.")

    # salva
    out = pd.DataFrame({"dataset_true": ds, "dataset_pred": ds_vals[pred_all], "pmax": probs_all})
    out.to_csv(os.path.join(diag_dir, "dataset_predictability_predictions.csv"), index=False)


def label_permutation_test(df, y_col, g_col, d_col=None, seed=42):
    """
    Sanity check: embaralha labels e re-roda uma avaliação simples LOSO.
    Se o AUC continuar alto após permutar, pode haver vazamento/confounding forte.
    """
    print("\n" + "=" * 80)
    print("LABEL PERMUTATION TEST (sanity/leakage)")
    print("=" * 80)

    dfp = df.copy()
    rng = np.random.default_rng(seed)

    # shuffle dentro de cada dataset (se existir d_col), senão shuffle global
    if d_col is not None and d_col in dfp.columns:
        def shuffle_group(s):
            arr = np.array(s.to_numpy(), copy=True)  # garante não-readonly
            rng.shuffle(arr)
            return arr

        dfp[y_col] = dfp.groupby(d_col)[y_col].transform(shuffle_group).astype(int)
        print(f"Labels permutados dentro de cada {d_col}.")
    else:
        arr = np.array(dfp[y_col].to_numpy(), copy=True)
        rng.shuffle(arr)
        dfp[y_col] = arr.astype(int)
        print("Labels permutados globalmente.")

    # Rode um modelo simples e barato só para sanity (LR, sem multiview)
    # Usa as MESMAS regras do seu pipeline: drop label e subject do treino
    cols = [c for c in dfp.columns if c not in [y_col, g_col]]
    y = dfp[y_col].astype(int).values
    groups = dfp[g_col].astype(str).values
    X_df = dfp[cols].copy()

    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    logo = LeaveOneGroupOut()
    probs_all = []
    y_all = []

    for tr, te in logo.split(X_df, y, groups):
        X_tr_df, X_te_df = X_df.iloc[tr], X_df.iloc[te]
        y_tr, y_te = y[tr], y[te]

        pre, low_card = build_preprocessor(num_cols, cat_cols, X_tr_df)
        X_tr = pre.fit_transform(X_tr_df)
        X_te = pre.transform(X_te_df)

        X_tr = np.asarray(X_tr, dtype=np.float32)
        X_te = np.asarray(X_te, dtype=np.float32)
        X_tr = np.nan_to_num(X_tr, nan=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0)

        selector = VarianceThreshold(threshold=0.01)
        X_tr = selector.fit_transform(X_tr)
        X_te = selector.transform(X_te)

        # Sem SMOTE aqui: sanity check (deixa simples e rápido)
        clf = LogisticRegression(max_iter=4000, solver="liblinear", random_state=RANDOM_STATE)
        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_te)[:, 1]

        probs_all.extend(p.tolist())
        y_all.extend(y_te.tolist())

    probs_all = np.array(probs_all, dtype=float)
    y_all = np.array(y_all, dtype=int)

    m = evaluate(y_all, probs_all, threshold=0.5, label="Permutation sanity (sample-level)")
    print("\nInterpretação esperada:")
    print("- ROC AUC deve cair para ~0.5 (ou perto disso).")
    print("- Acc/BalAcc devem cair perto do chute (dependendo do desbalanceamento).")
    print("Se continuar muito alto, suspeite vazamento/confounding.")
    return m


# ==============================
# Optional EDA (Exploratory only)
# ==============================
def run_eda_plots(df, y_col, g_col):
    from sklearn.manifold import TSNE

    X_vis = df.drop(columns=[y_col, g_col])
    y_vis = df[y_col].astype(int).values

    vis_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]
    )
    Xv = vis_pipe.fit_transform(X_vis)
    Xv = np.asarray(Xv, dtype=np.float32)
    Xv = np.nan_to_num(Xv, nan=0.0)

    if PLOT_EDA_TSNE:
        n = min(1000, Xv.shape[0])
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(Xv.shape[0], n, replace=False) if Xv.shape[0] > n else np.arange(Xv.shape[0])
        tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE)
        X_emb = tsne.fit_transform(Xv[idx])
        fig = plt.figure(figsize=(9, 7))
        plt.scatter(X_emb[:, 0], X_emb[:, 1], c=y_vis[idx], alpha=0.6)
        plt.title("EDA – t-SNE (dataset completo)")
        fig.savefig(save_path("eda_tsne.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    if PLOT_EDA_PCA:
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        Xp = pca.fit_transform(Xv)
        fig = plt.figure(figsize=(9, 7))
        plt.scatter(Xp[:, 0], Xp[:, 1], c=y_vis, alpha=0.6)
        plt.title("EDA – PCA (dataset completo)")
        fig.savefig(save_path("eda_pca.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    df, y_col, g_col, d_col, r_col = load_data()

    # ===== 0) integrity checks =====
    if RUN_INTEGRITY_CHECKS:
        dataset_integrity_report(df, y_col, g_col, d_col=d_col, r_col=r_col)

    # ===== 1) dataset confounding test =====
    if RUN_DATASET_PREDICTABILITY_TEST and (d_col is not None):
        dataset_predictability_test(df, y_col, g_col, d_col)

    # ===== 2) EDA (exploratório) =====
    if any([PLOT_EDA_TSNE, PLOT_EDA_PCA, PLOT_EDA_LDA]):
        run_eda_plots(df, y_col, g_col)

    # ===== 3) main LOSO (principal) =====
    results = run_loso_parallel(df, y_col, g_col, light_models=False, do_perm_importance=True)

    # ===== 4) permutation sanity test =====
    if RUN_LABEL_PERMUTATION_TEST:
        label_permutation_test(df, y_col, g_col, d_col=d_col)

    print("\nArquivos gerados em:", os.path.abspath(OUT_DIR))
    print("Concluído.")
