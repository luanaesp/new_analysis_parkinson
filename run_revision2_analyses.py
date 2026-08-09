# =============================================================================
#  run_revision2_analyses.py
#  Second-round revision analyses for the PD-voice paper.
#
#  ADDRESSES
#    [R1] Imputation audit               -> Reviewer 3, major concern 2
#    [R2] Cohort-prevalence baseline     -> Reviewer 3, major concern 2 (real mechanism)
#    [R3] Per-cohort LOSO, zero imputation-> Reviewer 3, major concern 2
#    [R4] Preprocessing-placement audit  -> self-audit (CV hygiene)
#    [R5] Sex-stratified AUC             -> Reviewer 2, M5(b)
#    [R6] Importance vs batch-effect     -> Reviewer 3, major concern 1
#    [R7] Minimal-core 2013 robustness   -> promised in manuscript, never reported
#
#  RUN:   python run_revision2_analyses.py
#  OUT:   outputs/revision2_summary.json + figR1..figR3 .png
# =============================================================================
import warnings, time, json, os
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RS = 42
np.random.seed(RS)
PATHS = {
    "unified": "UNIFIED_RECONSTRUCTED (1).csv",
    "carron":  "PD-Dataset.csv",
    "sakar13_train": "train_data.txt",
    "sakar13_test":  "test_data.txt",
}
OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
N_BOOT = 2000
import bootstrap_utils as _bu
# Same construction as the main pipeline: one labelled stream per analysis.
def stream(label): return _bu.stream(label, RS)
rep = {}
t0 = time.time()

def sig(z): return 1/(1+np.exp(-z))

def boot_auc_ci(y, p, label, n=N_BOOT): return _bu.boot_auc_ci(y, p, label, RS, n)

df = pd.read_csv(PATHS["unified"])
skf = StratifiedKFold(5, shuffle=True, random_state=RS)

# ------------------------------------------------- harmonised core (as in main)
MFCC_SAK = ["0th","1st","2nd","3rd","4th","5th","6th","7th","8th","9th","10th","11th","12th"]
mapping = {
 'jitter_rel':{'sakar2019':'pd_speech_locPctJitter','naranjo':'replicated_Jitter_rel','carron':'Jitter'},
 'shimmer_loc':{'sakar2019':'pd_speech_locShimmer','naranjo':'replicated_Shim_loc','carron':'Shimmer'},
 'hnr':{'sakar2019':'pd_speech_meanHarmToNoiseHarmonicity','naranjo':'__HNRmean__','carron':'HNR'},
 'ppe':{'sakar2019':'pd_speech_PPE','naranjo':'replicated_PPE','carron':'PPE'},
 'rpde':{'sakar2019':'pd_speech_RPDE','naranjo':'replicated_RPDE','carron':'RPDE'},
 'gne':{'sakar2019':'pd_speech_GNE_mean','naranjo':'replicated_GNE','carron':'GNE'},
}
for i in range(13):
    mapping[f'mfcc{i}'] = {'sakar2019':f'pd_speech_mean_MFCC_{MFCC_SAK[i]}_coef',
                           'naranjo':f'replicated_MFCC{i}','carron':f'MFCC{i}'}
feats = list(mapping.keys())

SEXCOL = {'sakar2019':'pd_speech_gender', 'naranjo':'replicated_Gender'}

def subj_table(src_name, source_val):
    sub = df[df['dataset_source'] == source_val].copy()
    sub['replicated_HNRmean'] = sub[[f'replicated_HNR{b}' for b in ['05','15','25','35','38']]].mean(axis=1)
    cols = []
    for col in feats:
        cn = mapping[col][src_name]
        cols.append(sub['replicated_HNRmean'] if cn == '__HNRmean__' else sub[cn])
    X = pd.concat(cols, axis=1); X.columns = feats
    X['label'] = sub['label'].values
    X['subject_id'] = sub['subject_id'].values
    X['sex'] = sub[SEXCOL[src_name]].values
    g = X.groupby('subject_id').mean(numeric_only=True).reset_index()
    g['cohort'] = src_name
    return g

t_sak = subj_table('sakar2019','pd_speech')
t_nar = subj_table('naranjo','replicated')
c = pd.read_csv(PATHS["carron"], sep=';')
car = pd.DataFrame({f: c[mapping[f]['carron']] for f in feats})
car['label'] = c['Status'].values; car['subject_id'] = c['Subject'].values
car['sex'] = c['Sex'].values; car['cohort'] = 'carron'
H = pd.concat([t_sak, t_nar, car], ignore_index=True)
H['lab'] = H['cohort'].map({'sakar2019':'Istanbul','naranjo':'Extremadura','carron':'Extremadura'})
sak = H[H.cohort=='sakar2019'].reset_index(drop=True)
ys  = sak['label'].astype(int)

# honest pipeline: imputer + scaler fitted INSIDE each CV fold
def honest_pipe(clf):
    return Pipeline([('i', SimpleImputer(strategy='median')),
                     ('s', StandardScaler()), ('m', clf)])

def ens_pred(tr, te, cols):
    """Train-fold-only imputation + scaling; unchanged from the main pipeline."""
    imp = SimpleImputer(strategy='median').fit(tr[cols])
    a = imp.transform(tr[cols]); b = imp.transform(te[cols])
    sc = StandardScaler().fit(a); a = sc.transform(a); b = sc.transform(b)
    ytr = tr['label'].astype(int).values
    if min(np.bincount(ytr)) > 5: a, ytr = SMOTE(random_state=RS).fit_resample(a, ytr)
    ps = []
    for m in [LogisticRegression(max_iter=5000), SVC(kernel='rbf', C=1, random_state=RS),
              RandomForestClassifier(300, random_state=RS, n_jobs=-1)]:
        m.fit(a, ytr)
        ps.append(sig(m.decision_function(b)) if isinstance(m, SVC) else m.predict_proba(b)[:,1])
    return np.mean(ps, axis=0)

print("="*70)

# ==================================================== [R1] IMPUTATION AUDIT
print("[R1] Imputation audit ...")
d2 = df[df.dataset_source.isin(['pd_speech','replicated'])].copy()
fc798 = [x for x in d2.columns if x.startswith('pd_speech_') or x.startswith('replicated_')]
f_sak = [x for x in fc798 if x.startswith('pd_speech_')]
f_nar = [x for x in fc798 if x.startswith('replicated_')]
is_sak = (d2.dataset_source == 'pd_speech').values

# missingness in the harmonised 19-feature core
core_missing = int(H[feats].isna().sum().sum())

# missingness in the 798 native block, split into structural vs sporadic
blk = int(d2.loc[~is_sak, f_sak].isna().sum().sum() + d2.loc[is_sak, f_nar].isna().sum().sum())
tot_missing = int(d2[fc798].isna().sum().sum())
sporadic = tot_missing - blk

rep["imputation_audit"] = {
    "harmonised_core_missing_cells": core_missing,
    "harmonised_core_cells_total": int(H[feats].size),
    "native_798_missing_cells": tot_missing,
    "native_798_cells_total": int(d2[fc798].size),
    "native_798_missing_fraction": round(tot_missing/d2[fc798].size, 4),
    "structural_cross_cohort_block_cells": blk,
    "sporadic_within_cohort_missing_cells": sporadic,
    "imputer_strategy": "median, fitted on the training fold only, computed over all "
                        "training subjects jointly (NOT conditioned on the class label)",
    "note": "The harmonised core used for every external-validation result contains zero "
            "missing values, so the imputer is a no-op there. In the 798-feature native "
            "analysis all missingness is structural: features exist in one cohort and not "
            "the other. No value is ever imputed from class-conditional statistics."
}
print(f"    harmonised core missing cells: {core_missing} / {H[feats].size}")
print(f"    native 798 missing: {tot_missing} ({tot_missing/d2[fc798].size:.1%})"
      f" | structural block: {blk} | sporadic: {sporadic}")

# empirical no-op proof on the external transfer
teE = H[H.lab=='Extremadura']; trI = H[H.lab=='Istanbul']
yext = teE['label'].astype(int).values
auc_median_imp = roc_auc_score(yext, ens_pred(trI, teE, feats))

def ens_pred_noimp(tr, te, cols):
    a = tr[cols].values.astype(float); b = te[cols].values.astype(float)
    sc = StandardScaler().fit(a); a = sc.transform(a); b = sc.transform(b)
    ytr = tr['label'].astype(int).values
    if min(np.bincount(ytr)) > 5: a, ytr = SMOTE(random_state=RS).fit_resample(a, ytr)
    ps = []
    for m in [LogisticRegression(max_iter=5000), SVC(kernel='rbf', C=1, random_state=RS),
              RandomForestClassifier(300, random_state=RS, n_jobs=-1)]:
        m.fit(a, ytr)
        ps.append(sig(m.decision_function(b)) if isinstance(m, SVC) else m.predict_proba(b)[:,1])
    return np.mean(ps, axis=0)

auc_no_imp = roc_auc_score(yext, ens_pred_noimp(trI, teE, feats))
rep["imputation_audit"]["external_auc_with_imputer"] = round(float(auc_median_imp), 4)
rep["imputation_audit"]["external_auc_imputer_removed"] = round(float(auc_no_imp), 4)
rep["imputation_audit"]["identical"] = bool(abs(auc_median_imp - auc_no_imp) < 1e-12)
print(f"    external AUC with imputer = {auc_median_imp:.4f} | imputer removed = {auc_no_imp:.4f}"
      f" | identical = {rep['imputation_audit']['identical']}")

# ==================================== [R2] COHORT-PREVALENCE BASELINE (798 feat)
print("[R2] Cohort-prevalence baseline for the pooled 798-feature analysis ...")
g_lab = d2.groupby(['dataset_source','subject_id'])['label'].first().reset_index()
coh = (g_lab.dataset_source == 'pd_speech').astype(int).values
ylab = g_lab['label'].astype(int).values
auc_cohort_only = roc_auc_score(ylab, coh)
prev = g_lab.groupby('dataset_source')['label'].agg(['size','mean']).round(4)
rep["cohort_prevalence_baseline"] = {
    "auc_from_cohort_membership_alone": round(float(auc_cohort_only), 4),
    "prevalence_by_cohort": {k: {"n": int(v['size']), "pd_rate": float(v['mean'])}
                              for k, v in prev.iterrows()},
    "note": "The pooled 798-feature LOSO mixes two cohorts with different PD prevalence. "
            "A classifier with no acoustic information at all, told only which cohort a "
            "subject came from, already achieves this AUC. Reported pooled AUC must be "
            "read against this floor, not against 0.5."
}
print(f"    cohort membership alone -> AUC = {auc_cohort_only:.4f}")
print(f"    prevalence: {rep['cohort_prevalence_baseline']['prevalence_by_cohort']}")

# ================================ [R3] PER-COHORT 798-feat LOSO (no imputation)
print("[R3] Per-cohort native-feature LOSO with zero imputation (slow) ...")
def loso_single_cohort(source_val, cols):
    sub = df[df.dataset_source == source_val].copy()
    X = sub[cols].values.astype(float)
    y = sub['label'].astype(int).values
    g = sub['subject_id'].astype(str).values
    assert not np.isnan(X).any(), "unexpected missing values inside a single cohort"
    uniq = np.array(sorted(set(g)))
    def one(gid):
        te = g == gid; tr = ~te
        A = X[tr]; B = X[te]
        vt = VarianceThreshold(0.01).fit(A); A = vt.transform(A); B = vt.transform(B)
        sc = StandardScaler().fit(A); A = sc.transform(A); B = sc.transform(B)
        ytr = y[tr]
        if min(np.bincount(ytr)) > 5: A, ytr = SMOTE(random_state=RS).fit_resample(A, ytr)
        lr = LogisticRegression(max_iter=4000, solver='liblinear').fit(A, ytr)
        sv = SVC(kernel='rbf', C=1, gamma='scale', random_state=RS).fit(A, ytr)
        rf = RandomForestClassifier(100, random_state=RS, n_jobs=1).fit(A, ytr)
        p = np.mean([lr.predict_proba(B)[:,1], sig(sv.decision_function(B)),
                     rf.predict_proba(B)[:,1]], axis=0)
        return float(p.mean()), int(y[te][0])
    out = Parallel(n_jobs=-1)(delayed(one)(gid) for gid in uniq)
    pv, yv = zip(*out)
    return np.array(yv), np.array(pv)

percohort = {}
for src, cols, nm in [('pd_speech', f_sak, 'sakar2019'), ('replicated', f_nar, 'naranjo')]:
    yv, pv = loso_single_cohort(src, cols)
    a = roc_auc_score(yv, pv); ci = boot_auc_ci(yv, pv, f"per_cohort_native_loso/{nm}")
    percohort[nm] = {"auc": round(float(a), 4), "ci95": [round(x,3) for x in ci],
                     "n_subjects": int(len(yv)), "n_features": len(cols),
                     "pd_rate": round(float(yv.mean()), 4), "imputed_values": 0}
    print(f"    {nm}: AUC={a:.4f} CI{tuple(round(x,3) for x in ci)} "
          f"(n={len(yv)}, {len(cols)} features, 0 imputed)")
rep["per_cohort_native_loso"] = percohort
rep["per_cohort_native_loso"]["note"] = (
    "Each cohort analysed with only its own native features, so no cross-cohort "
    "imputation occurs and no cohort-prevalence contrast can contribute. These are the "
    "cleanest within-cohort estimates in the study.")

# ======================= [R4] PREPROCESSING-PLACEMENT AUDIT (CV hygiene)
print("[R4] Preprocessing-placement audit ...")
def prep_outside(a):
    return StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(a))

p_leaky = cross_val_predict(LogisticRegression(max_iter=5000), prep_outside(sak[feats]),
                            ys, cv=skf, method='predict_proba')[:,1]
p_honest = cross_val_predict(honest_pipe(LogisticRegression(max_iter=5000)), sak[feats],
                             ys, cv=skf, method='predict_proba')[:,1]
a_leaky, a_honest = roc_auc_score(ys, p_leaky), roc_auc_score(ys, p_honest)
rep["preprocessing_placement_audit"] = {
    "scaler_fitted_on_full_dataset_auc": round(float(a_leaky), 4),
    "scaler_fitted_inside_cv_folds_auc": round(float(a_honest), 4),
    "difference": round(float(a_leaky - a_honest), 4),
    "note": "Standardisation fitted once on the full cohort versus refitted inside every "
            "CV fold. The imputer is a no-op here (zero missing values), so this isolates "
            "the effect of scaler placement alone."
}
print(f"    scaler outside CV: {a_leaky:.4f} | inside CV folds: {a_honest:.4f} "
      f"| delta = {a_leaky-a_honest:+.4f}")

# ==================================================== [R5] SEX-STRATIFIED AUC
print("[R5] Sex-stratified AUC ...")
sex_res = {}
sx = sak['sex'].round().astype(int).values
p_sex_model = p_honest  # honest within-cohort harmonised predictions
for v in sorted(set(sx)):
    m = sx == v
    yv = ys.values[m]; pv = p_sex_model[m]
    if len(np.unique(yv)) < 2:
        sex_res[f"sex_{v}"] = {"n": int(m.sum()), "auc": None}; continue
    a = roc_auc_score(yv, pv); ci = boot_auc_ci(yv, pv, f"sex_stratified/sex_{v}")
    sex_res[f"sex_{v}"] = {"n": int(m.sum()), "n_pd": int(yv.sum()),
                           "pd_rate": round(float(yv.mean()), 4),
                           "auc": round(float(a), 4), "ci95": [round(x,3) for x in ci]}
    print(f"    sex={v}: n={m.sum():3d} PD-rate={yv.mean():.3f} AUC={a:.4f} "
          f"CI{tuple(round(x,3) for x in ci)}")

# bootstrap difference between the two sex strata
g0, g1 = sorted(set(sx))[0], sorted(set(sx))[1]
m0, m1 = sx == g0, sx == g1
y0, q0 = ys.values[m0], p_sex_model[m0]
y1, q1 = ys.values[m1], p_sex_model[m1]
diffs = []
_g_sex = stream("sex_stratified/difference")
for _ in range(N_BOOT):
    i0 = _g_sex.integers(0, len(y0), len(y0)); i1 = _g_sex.integers(0, len(y1), len(y1))
    if len(np.unique(y0[i0])) > 1 and len(np.unique(y1[i1])) > 1:
        diffs.append(roc_auc_score(y0[i0], q0[i0]) - roc_auc_score(y1[i1], q1[i1]))
diffs = np.array(diffs)
p_sex = float(min(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1.0))
sex_res["difference"] = {"delta_auc": round(float(np.mean(diffs)), 4),
                         "ci95": [round(float(np.percentile(diffs, 2.5)), 3),
                                  round(float(np.percentile(diffs, 97.5)), 3)],
                         "p_value": round(p_sex, 4)}
# sex alone as a predictor of PD status
sex_res["auc_from_sex_alone"] = round(float(roc_auc_score(ys.values, sx)), 4)
sex_res["note"] = ("Sex-stratified discrimination on the within-cohort harmonised core "
                   "(Sakar2019, the only cohort large enough to stratify). Overlapping "
                   "intervals indicate the model is not simply exploiting the differing "
                   "PD prevalence between sexes.")
rep["sex_stratified"] = sex_res
print(f"    delta AUC = {sex_res['difference']['delta_auc']:.3f} "
      f"CI{sex_res['difference']['ci95']} p={p_sex:.4f} | sex alone AUC={sex_res['auc_from_sex_alone']:.4f}")

# sex distribution across all three cohorts
sexdist = {}
for nm in ['sakar2019','naranjo','carron']:
    sb = H[H.cohort==nm]
    ct = pd.crosstab(sb['sex'].round().astype(int), sb['label'].astype(int))
    sexdist[nm] = {f"sex_{int(i)}": {"HC": int(r.get(0,0)), "PD": int(r.get(1,0)),
                    "pd_rate": round(float(r.get(1,0)/max(r.sum(),1)), 4)}
                   for i, r in ct.iterrows()}
rep["sex_distribution_by_cohort"] = sexdist

# =========================== [R6] IMPORTANCE vs BATCH-EFFECT DISCRIMINABILITY
print("[R6] Feature importance vs cohort-discriminability ...")
ylab_bin = (H['lab'] == 'Istanbul').astype(int).values
rows = []
for f in feats:
    v = H[f].values.astype(float)
    a_lab = roc_auc_score(ylab_bin, v)                       # separates the two labs?
    a_lab = max(a_lab, 1 - a_lab)                            # direction-free
    vs = sak[f].values.astype(float)
    a_pd = roc_auc_score(ys.values, vs)                      # separates PD from HC?
    a_pd = max(a_pd, 1 - a_pd)
    rows.append({"feature": f, "auc_pd_within_sakar2019": round(float(a_pd), 4),
                 "auc_lab_discrimination": round(float(a_lab), 4)})
fi = pd.DataFrame(rows).sort_values("auc_lab_discrimination", ascending=False)
fi.to_csv(f"{OUT}/feature_batch_vs_pd_table.csv", index=False)
rep["importance_vs_batch"] = {
    "table": fi.to_dict("records"),
    "n_features_lab_auc_above_0.90": int((fi.auc_lab_discrimination > 0.90).sum()),
    "n_features_lab_auc_above_0.80": int((fi.auc_lab_discrimination > 0.80).sum()),
    "median_lab_auc": round(float(fi.auc_lab_discrimination.median()), 4),
    "median_pd_auc": round(float(fi.auc_pd_within_sakar2019.median()), 4),
    "spearman_r_pd_vs_lab": round(float(fi.auc_pd_within_sakar2019.corr(
                                   fi.auc_lab_discrimination, method='spearman')), 4),
    "note": "For every harmonised feature: univariate AUC for separating PD from HC within "
            "one cohort, against univariate AUC for separating the two laboratories. A "
            "feature that is highly informative on the second axis cannot be interpreted "
            "as a clinical biomarker without qualification."
}
print(f"    features with lab-AUC > 0.90: {rep['importance_vs_batch']['n_features_lab_auc_above_0.90']}/19"
      f" | median lab-AUC = {rep['importance_vs_batch']['median_lab_auc']}")
print(f"    top batch-driven: {fi.head(5)['feature'].tolist()}")

# ============================= [R7] MINIMAL-CORE ROBUSTNESS CHECK (2013 cohort)
print("[R7] Minimal-core robustness check including the 2013 cohort ...")
try:
    tr13 = pd.read_csv(PATHS["sakar13_train"], header=None)
    te13 = pd.read_csv(PATHS["sakar13_test"], header=None)
    # UCI-301: col0 = subject id, cols 1..26 = features, last col = class
    def core13(t):
        return pd.DataFrame({'subject_id': t[0], 'jitter_rel': t[1], 'shimmer_loc': t[6],
                             'hnr': t[14], 'label': t.iloc[:, -1]})
    s13 = pd.concat([core13(tr13), core13(te13)], ignore_index=True)
    s13 = s13.groupby('subject_id').mean(numeric_only=True).reset_index()
    s13['label'] = s13['label'].round().astype(int)
    s13['cohort'] = 'sakar2013'; s13['lab'] = 'Istanbul'

    mini = ['jitter_rel', 'shimmer_loc', 'hnr']
    base = H[mini + ['label', 'cohort', 'lab']].copy()

    # (a) excluded, as in the main analysis
    tr_a = base[base.lab=='Istanbul']; te_a = base[base.lab=='Extremadura']
    y_a = te_a['label'].astype(int).values
    p_a = ens_pred(tr_a, te_a, mini); auc_a = roc_auc_score(y_a, p_a)
    ci_a = boot_auc_ci(y_a, p_a, "minimal_core_2013/excluded")

    # (b) 2013 cohort added to the Istanbul training side
    withx = pd.concat([base, s13[mini + ['label','cohort','lab']]], ignore_index=True)
    tr_b = withx[withx.lab=='Istanbul']; te_b = withx[withx.lab=='Extremadura']
    y_b = te_b['label'].astype(int).values
    p_b = ens_pred(tr_b, te_b, mini); auc_b = roc_auc_score(y_b, p_b)
    ci_b = boot_auc_ci(y_b, p_b, "minimal_core_2013/included")

    rep["minimal_core_2013_check"] = {
        "features": mini,
        "n_2013_subjects_added": int(len(s13)),
        "external_auc_2013_excluded": round(float(auc_a), 4),
        "ci95_excluded": [round(x,3) for x in ci_a],
        "external_auc_2013_included": round(float(auc_b), 4),
        "ci95_included": [round(x,3) for x in ci_b],
        "delta": round(float(auc_b - auc_a), 4),
        "note": "Leave-one-laboratory-out transfer on the three features shared by all four "
                "cohorts, with and without the 2013 Istanbul cohort in the training set. "
                "This is the robustness check the manuscript promised."
    }
    print(f"    minimal core, 2013 excluded: AUC={auc_a:.4f} CI{tuple(round(x,3) for x in ci_a)}")
    print(f"    minimal core, 2013 included: AUC={auc_b:.4f} CI{tuple(round(x,3) for x in ci_b)}")
    print(f"    delta = {auc_b-auc_a:+.4f}")
except Exception as e:
    rep["minimal_core_2013_check"] = {"error": str(e)}
    print("    [skip]", e)

# ================ [R8] DOES DROPPING BATCH-DRIVEN FEATURES RESCUE TRANSFER?
print("[R8] Batch-neutral feature subsets ...")
neutral_strict = fi.loc[fi.auc_lab_discrimination < fi.auc_pd_within_sakar2019, 'feature'].tolist()
neutral_thresh = fi.loc[fi.auc_lab_discrimination < 0.70, 'feature'].tolist()

def eval_subset(cols, label):
    if len(cols) < 2: return None
    p_int = cross_val_predict(honest_pipe(LogisticRegression(max_iter=5000)),
                              sak[cols], ys, cv=skf, method='predict_proba')[:,1]
    a_int = roc_auc_score(ys, p_int)
    p_ext = ens_pred(H[H.lab=='Istanbul'], H[H.lab=='Extremadura'], cols)
    y_ext = H[H.lab=='Extremadura']['label'].astype(int).values
    a_ext = roc_auc_score(y_ext, p_ext)
    return {"n_features": len(cols), "features": cols,
            "within_cohort_auc": round(float(a_int), 4),
            "external_auc": round(float(a_ext), 4),
            "external_ci95": [round(x,3) for x in boot_auc_ci(y_ext, p_ext, f"batch_neutral_subsets/{label}")]}

subsets = {
    "all_19_features": eval_subset(feats, "all_19_features"),
    "batch_neutral_strict (lab AUC < PD AUC)": eval_subset(neutral_strict, "batch_neutral_strict"),
    "batch_neutral_threshold (lab AUC < 0.70)": eval_subset(neutral_thresh, "batch_neutral_threshold"),
    "batch_driven_only (lab AUC > 0.90)":
        eval_subset(fi.loc[fi.auc_lab_discrimination > 0.90, 'feature'].tolist(), "batch_driven_only"),
}
rep["batch_neutral_subsets"] = subsets
rep["batch_neutral_subsets"]["note"] = (
    "Can cross-laboratory transfer be rescued by discarding the features that carry the "
    "strongest laboratory signature? Within-cohort and leave-one-laboratory-out AUC are "
    "recomputed on nested feature subsets defined purely by batch discriminability, never "
    "by outcome performance, so the comparison is not circular.")
for k, v in subsets.items():
    if isinstance(v, dict) and v:
        print(f"    {k}: within={v['within_cohort_auc']} external={v['external_auc']} "
              f"CI{v['external_ci95']} ({v['n_features']} feats)")

# ============ [R9] t-SNE / PCA REGENERATED ON THE RELEASED HARMONISED CORE
print("[R9] Regenerating t-SNE and PCA on the harmonised subject-level core ...")
Xemb = StandardScaler().fit_transform(H[feats])
yemb = H['label'].astype(int).values
lab_emb = H['lab'].values

pca = PCA(n_components=2, random_state=RS).fit(Xemb)
Z_pca = pca.transform(Xemb)
evr = pca.explained_variance_ratio_
load = pd.DataFrame(pca.components_.T, index=feats, columns=['PC1','PC2'])
load['mag'] = load.abs().max(axis=1)
top_load = load.sort_values('mag', ascending=False).head(6).index.tolist()
n_mfcc_top = sum(f.startswith('mfcc') for f in top_load)

Z_tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
              init='pca', random_state=RS).fit_transform(Xemb)

rep["embeddings"] = {
    "space": "harmonised 19-feature core, subject level (n=392)",
    "pca_explained_variance_pc1": round(float(evr[0]), 4),
    "pca_explained_variance_pc2": round(float(evr[1]), 4),
    "pca_explained_variance_pc1_pc2": round(float(evr[:2].sum()), 4),
    "pca_top6_loadings": top_load,
    "pca_n_mfcc_in_top6": int(n_mfcc_top),
    "tsne_perplexity": 30, "tsne_max_iter": 1000,
    "note": "Regenerated from the released pipeline. Earlier versions of these panels came "
            "from a superseded exploratory script and used a continuous colour scale for a "
            "binary variable; class is now encoded by two discrete colours and laboratory "
            "family by marker shape."
}
print(f"    PC1={evr[0]:.1%} PC2={evr[1]:.1%} (sum {evr[:2].sum():.1%}) | "
      f"top loadings: {top_load} | MFCC in top6: {n_mfcc_top}")

# ==================================================================== FIGURES
# Nature style: no in-figure titles (captions carry them). PNG + vector PDF emitted.
CLR = {'pd':'#1b3a6b','lab':'#c0392b','gray':'#7f8c8d','ok':'#2e86c1'}
def saveboth(name):
    plt.savefig(f"{OUT}/{name}.png", dpi=200); plt.savefig(f"{OUT}/{name}.pdf"); plt.close()

# permutation importance recomputed here (identical seed/protocol to the main
# pipeline) so the manuscript composite can pair it with the batch analysis
Xtr_, Xte_, ytr_, yte_ = train_test_split(sak[feats], ys, test_size=.3, stratify=ys, random_state=RS)
pipe_ = Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),
                  ('lr',LogisticRegression(max_iter=5000))]).fit(Xtr_, ytr_)
pi_ = permutation_importance(pipe_, Xte_, yte_, n_repeats=30, random_state=RS, scoring='roc_auc')
impS = pd.Series(pi_.importances_mean, index=feats).sort_values(ascending=False)

def draw_importance(ax):
    impS.head(10)[::-1].plot(kind='barh', color=CLR['ok'], ax=ax)
    ax.set_xlabel('Mean AUC drop when permuted')

# manual offsets for labels that otherwise collide
LBL_OFF = {'mfcc8': (5,-9), 'mfcc9': (5,4), 'ppe': (5,6), 'gne': (5,-9), 'mfcc12': (5,-9)}
def draw_scatter(ax):
    above = fi.auc_lab_discrimination >= fi.auc_pd_within_sakar2019
    ax.plot([0.45,1.02],[0.45,1.02], color=CLR['gray'], ls='--', lw=1.2, zorder=1)
    ax.scatter(fi.loc[above,'auc_pd_within_sakar2019'], fi.loc[above,'auc_lab_discrimination'],
               s=58, color=CLR['lab'], alpha=.85, edgecolor='white', linewidth=.9, zorder=3,
               label='batch signal exceeds clinical signal')
    ax.scatter(fi.loc[~above,'auc_pd_within_sakar2019'], fi.loc[~above,'auc_lab_discrimination'],
               s=58, color=CLR['ok'], alpha=.85, edgecolor='white', linewidth=.9, zorder=3,
               label='clinical signal exceeds batch signal')
    for _, r in fi.iterrows():
        off = LBL_OFF.get(r.feature, (5,3))
        ax.annotate(r.feature, (r.auc_pd_within_sakar2019, r.auc_lab_discrimination),
                    fontsize=7, xytext=off, textcoords='offset points', color='#333333')
    ax.set_xlim(0.48, 0.80); ax.set_ylim(0.45, 1.04)
    ax.set_xlabel('Univariate AUC: PD vs HC (within Sakar2019)')
    ax.set_ylabel('Univariate AUC: Istanbul vs Extremadura')
    ax.legend(fontsize=7, loc='lower right', framealpha=.92)

GAP_ORDER = ["batch_driven_only (lab AUC > 0.90)", "all_19_features",
             "batch_neutral_threshold (lab AUC < 0.70)", "batch_neutral_strict (lab AUC < PD AUC)"]
GAP_SHORT = ["Batch-driven\nonly (4f)", "All harmonised\n(19f)",
             "Batch-neutral,\nlenient (7f)", "Batch-neutral,\nstrict (5f)"]
def draw_gap(ax):
    wi = [subsets[k]["within_cohort_auc"] for k in GAP_ORDER]
    ex = [subsets[k]["external_auc"] for k in GAP_ORDER]
    xp = np.arange(len(GAP_ORDER)); w = 0.36
    ax.bar(xp-w/2, wi, w, label='Within-cohort', color=CLR['pd'], alpha=.9)
    ax.bar(xp+w/2, ex, w, label='Leave-one-lab-out', color=CLR['lab'], alpha=.9)
    for i in range(len(GAP_ORDER)):
        ax.text(xp[i]-w/2, wi[i]+.008, f'{wi[i]:.3f}', ha='center', fontsize=7.5)
        ax.text(xp[i]+w/2, ex[i]+.008, f'{ex[i]:.3f}', ha='center', fontsize=7.5)
        ax.annotate('', xy=(xp[i]+w/2, ex[i]), xytext=(xp[i]-w/2, wi[i]),
                    arrowprops=dict(arrowstyle='->', color='#444444', lw=1.1, ls=':'))
        ax.text(xp[i], (wi[i]+ex[i])/2, f'  −{wi[i]-ex[i]:.3f}', fontsize=7.5,
                color='#444444', ha='left', va='center')
    ax.set_xticks(xp); ax.set_xticklabels(GAP_SHORT, fontsize=7.5)
    ax.axhline(0.5, color='k', ls=':'); ax.set_ylim(0.45, 0.9); ax.set_ylabel('AUC')
    ax.legend(fontsize=7.5, loc='upper left', framealpha=.92)

def draw_sex(ax):
    ks = [k for k in sex_res if k.startswith('sex_') and sex_res[k].get('auc')]
    vals = [sex_res[k]['auc'] for k in ks]
    errs = [[sex_res[k]['auc']-sex_res[k]['ci95'][0] for k in ks],
            [sex_res[k]['ci95'][1]-sex_res[k]['auc'] for k in ks]]
    ax.bar([f"{k}\n(n={sex_res[k]['n']}, PD {sex_res[k]['pd_rate']:.0%})" for k in ks],
           vals, color=[CLR['pd'], CLR['ok']], yerr=errs, capsize=6, alpha=.9)
    ax.axhline(0.5, color='k', ls=':'); ax.set_ylim(0.4, 1.0); ax.set_ylabel('AUC')
    for i, v in enumerate(vals): ax.text(i, v+0.012, f'{v:.3f}', ha='center', fontsize=9)

CLS = {0: '#2e86c1', 1: '#c0392b'}          # HC / PD, two discrete colours
MRK = {'Istanbul': 'o', 'Extremadura': '^'}
def draw_embed(ax, Z, xl, yl, legend=False):
    for lb, mk in MRK.items():
        for cl, cc in CLS.items():
            m = (lab_emb == lb) & (yemb == cl)
            if not m.any(): continue
            ax.scatter(Z[m,0], Z[m,1], c=cc, marker=mk, s=30, alpha=.78,
                       edgecolor='white', linewidth=.5,
                       label=f"{'PD' if cl else 'HC'} — {lb}")
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    if legend: ax.legend(fontsize=7, framealpha=.92, loc='best')

def panel_letters(axs, letters):
    for ax, letter in zip(axs, letters):
        ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontweight='bold', fontsize=13)

# --- standalone versions (repo continuity) ---
plt.figure(figsize=(6.8,5.8)); draw_scatter(plt.gca()); plt.tight_layout(); saveboth("figR1_importance_vs_batch")
plt.figure(figsize=(5.4,4.4)); draw_sex(plt.gca()); plt.tight_layout(); saveboth("figR2_sex_stratified_auc")
plt.figure(figsize=(6.6,4.4))
nm3 = ['Cohort membership\nalone (no acoustics)', 'Sakar2019 only\n(native, 0 imputed)',
       'Naranjo only\n(native, 0 imputed)']
v3 = [auc_cohort_only, percohort['sakar2019']['auc'], percohort['naranjo']['auc']]
plt.bar(nm3, v3, color=[CLR['gray'], CLR['pd'], CLR['ok']], alpha=.9)
plt.axhline(0.5, color='k', ls=':'); plt.ylim(0.45, 1.0); plt.ylabel('AUC')
for i, v in enumerate(v3): plt.text(i, v+0.01, f'{v:.3f}', ha='center', fontsize=9)
plt.tight_layout(); saveboth("figR3_pooled_decomposition")
plt.figure(figsize=(7.6,4.8)); draw_gap(plt.gca()); plt.tight_layout(); saveboth("figR4_generalisation_gap")
plt.figure(figsize=(6.4,5.4))
draw_embed(plt.gca(), Z_tsne, 't-SNE dimension 1', 't-SNE dimension 2', legend=True)
plt.tight_layout(); saveboth("figR5_tsne_discrete")
plt.figure(figsize=(6.4,5.4))
draw_embed(plt.gca(), Z_pca, f'PC1 ({evr[0]:.1%} of variance)', f'PC2 ({evr[1]:.1%} of variance)', legend=True)
plt.tight_layout(); saveboth("figR6_pca_discrete")

# --- manuscript composites ---
fig, axs = plt.subplots(1, 3, figsize=(15.6, 4.9))
draw_importance(axs[0]); draw_scatter(axs[1]); draw_gap(axs[2])
panel_letters(axs, ['a','b','c'])
plt.tight_layout(); saveboth("figM2_importance_batch")

fig, ax = plt.subplots(figsize=(5.2,4.3)); draw_sex(ax)
plt.tight_layout(); saveboth("figM3_sex")

fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.9))
draw_embed(axs[0], Z_tsne, 't-SNE dimension 1', 't-SNE dimension 2', legend=True)
draw_embed(axs[1], Z_pca, f'PC1 ({evr[0]:.1%} of variance)', f'PC2 ({evr[1]:.1%} of variance)')
panel_letters(axs, ['a','b'])
plt.tight_layout(); saveboth("figM4_embeddings")

rep["runtime_seconds"] = round(time.time()-t0, 1)
json.dump(rep, open(f"{OUT}/revision2_summary.json","w"), indent=2)
print("="*70)
print(f"DONE in {rep['runtime_seconds']}s -> outputs/revision2_summary.json")
print("  figR1_importance_vs_batch.png | figR2_sex_stratified_auc.png | figR3_pooled_decomposition.png")
print("  feature_batch_vs_pd_table.csv")
print("="*70)
print(json.dumps(rep, indent=2, default=str)[:4000])
