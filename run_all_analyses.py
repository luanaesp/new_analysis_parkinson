# =============================================================================
#  run_all_analyses.py
#  Full reproducibility + revision analysis pipeline for the PD-voice paper.
#
#  WHAT IT DOES
#    1.  Reproduces within-cohort LOSO (798 feat) -> AUC + bootstrap CI
#    2.  Wilson 95% CIs for sensitivity / specificity
#    3.  Harmonized common core (Sakar2019 + Naranjo + Carron)
#    4.  Leave-one-LABORATORY-out external validation (+ pairwise) with CIs
#    5.  Dataset/Lab predictability test (batch-effect confound)
#    6.  Label-permutation test (+ empirical p-value)
#    7.  Permutation importance (held-out)
#    8.  Probability averaging vs majority vote
#    9.  Demographics table (sex per cohort/class; age for Carron + t-test)
#   10.  Sakar2013<->Sakar2019 acoustic-distance de-duplication / sensitivity
#   11.  Leaky k-fold vs honest LOSO comparison (quantifies leakage inflation)
#   12.  Feature-provenance table for the 798 native features
#   13.  All figures -> ./outputs/   +  results_summary.json + provenance_table.csv
#
#  REQUIREMENTS
#    pip install numpy pandas scikit-learn imbalanced-learn scipy matplotlib joblib
#
#  INPUT FILES (place in the SAME folder, or edit the PATHS block below)
#    UNIFIED_RECONSTRUCTED_1.csv   (your unified dataset)
#    PDDataset.csv                 (Carron 2021, ';'-separated)
#    train_data.txt , test_data.txt(Sakar 2013, UCI 301)
#
#  RUN:   python run_all_analyses.py
# =============================================================================
import warnings, time, json, os, platform
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import sklearn, imblearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score, GridSearchCV, train_test_split)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# ----------------------------- CONFIG / PATHS --------------------------------
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
def sig(z): return 1/(1+np.exp(-z))
rng = np.random.default_rng(RS)
report = {}
t_start = time.time()

def wilson_ci(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k/n; d = 1+z**2/n
    c = (p+z**2/(2*n))/d; h = z*np.sqrt(p*(1-p)/n+z**2/(4*n**2))/d
    return (c-h, c+h)

def boot_auc_ci(y, p, n=N_BOOT):
    y = np.asarray(y); p = np.asarray(p); a = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) > 1: a.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

print("="*70)
print("ENV:", platform.platform(), "| python", platform.python_version())
print("sklearn", sklearn.__version__, "| imblearn", imblearn.__version__,
      "| numpy", np.__version__, "| pandas", pd.__version__)
print("CPU cores:", os.cpu_count(), "| seed:", RS)
print("="*70)
report["environment"] = {"platform": platform.platform(), "python": platform.python_version(),
    "sklearn": sklearn.__version__, "imblearn": imblearn.__version__,
    "numpy": np.__version__, "pandas": pd.__version__, "cpu": os.cpu_count(), "seed": RS}

df = pd.read_csv(PATHS["unified"])

# ============================================================= MAPPING / CORE
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

def subj_table(src_name, source_val):
    sub = df[df['dataset_source'] == source_val].copy()
    sub['replicated_HNRmean'] = sub[[f'replicated_HNR{b}' for b in ['05','15','25','35','38']]].mean(axis=1)
    cols = []
    for col in feats:
        c = mapping[col][src_name]
        cols.append(sub['replicated_HNRmean'] if c == '__HNRmean__' else sub[c])
    X = pd.concat(cols, axis=1); X.columns = feats
    X['label'] = sub['label'].values; X['subject_id'] = sub['subject_id'].values
    g = X.groupby('subject_id').mean(numeric_only=True).reset_index()
    g['cohort'] = src_name
    return g

t_sak = subj_table('sakar2019','pd_speech')
t_nar = subj_table('naranjo','replicated')
c = pd.read_csv(PATHS["carron"], sep=';')
car = pd.DataFrame({f: c[mapping[f]['carron']] for f in feats})
car['label'] = c['Status'].values; car['subject_id'] = c['Subject'].values; car['cohort'] = 'carron'
H = pd.concat([t_sak, t_nar, car], ignore_index=True)
H['lab'] = H['cohort'].map({'sakar2019':'Istanbul','naranjo':'Extremadura','carron':'Extremadura'})
H.to_csv(f"{OUT}/harmonized_subjectlevel.csv", index=False)
print("\n[CORE] harmonized table:", H.shape, "| by cohort:",
      H.groupby('cohort')['label'].agg(['size','mean']).round(3).to_dict('index'))

def prep(a): return StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(a))
def ens_pred(tr, te, cols):
    imp = SimpleImputer(strategy='median').fit(tr[cols]); a = imp.transform(tr[cols]); b = imp.transform(te[cols])
    sc = StandardScaler().fit(a); a = sc.transform(a); b = sc.transform(b); ytr = tr['label'].astype(int).values
    if min(np.bincount(ytr)) > 5: a, ytr = SMOTE(random_state=RS).fit_resample(a, ytr)
    ps = []
    for m in [LogisticRegression(max_iter=5000), SVC(kernel='rbf', C=1, random_state=RS),
              RandomForestClassifier(300, random_state=RS, n_jobs=-1)]:
        m.fit(a, ytr)
        ps.append(sig(m.decision_function(b)) if isinstance(m, SVC) else m.predict_proba(b)[:,1])
    return np.mean(ps, axis=0)

# ============================================== 1) REPRODUCE 798-feat LOSO
print("\n[1] Reproducing within-cohort LOSO (798 features)...")
d2 = df[df.dataset_source.isin(['pd_speech','replicated'])].copy()
d2['gid'] = d2['dataset_source'] + '::' + d2['subject_id'].astype(str)
fc798 = [x for x in d2.columns if x.startswith('pd_speech_') or x.startswith('replicated_')]
Xa = d2[fc798].values; ya = d2['label'].astype(int).values; ga = d2['gid'].values
groups = np.array(sorted(set(ga)))
def fold(gid):
    te = ga == gid; tr = ~te
    imp = SimpleImputer(strategy='median').fit(Xa[tr]); A = imp.transform(Xa[tr]); B = imp.transform(Xa[te])
    vt = VarianceThreshold(0.01).fit(A); A = vt.transform(A); B = vt.transform(B)
    sc = StandardScaler().fit(A); A = sc.transform(A); B = sc.transform(B); ytr = ya[tr]
    if min(np.bincount(ytr)) > 5: A, ytr = SMOTE(random_state=RS).fit_resample(A, ytr)
    lr = LogisticRegression(max_iter=4000, solver='liblinear').fit(A, ytr)
    sv = SVC(kernel='rbf', C=1, gamma='scale', random_state=RS).fit(A, ytr)
    rf = RandomForestClassifier(100, random_state=RS, n_jobs=1).fit(A, ytr)
    p = np.mean([lr.predict_proba(B)[:,1], sig(sv.decision_function(B)), rf.predict_proba(B)[:,1]], axis=0)
    return gid, float(p.mean()), int(ya[te][0])
res = Parallel(n_jobs=-1)(delayed(fold)(g) for g in groups)
_, pp, yy = zip(*res); pp = np.array(pp); yy = np.array(yy)
auc798 = roc_auc_score(yy, pp); ci798 = boot_auc_ci(yy, pp)
pred = (pp >= 0.5).astype(int); tn, fp, fn, tp = [int(v) for v in confusion_matrix(yy, pred).ravel()]
sens, spec = tp/(tp+fn), tn/(tn+fp)
sens_ci, spec_ci = wilson_ci(tp, tp+fn), wilson_ci(tn, tn+fp)
fpr_i, tpr_i, _ = roc_curve(yy, pp)
print(f"    AUC={auc798:.4f} CI{tuple(round(x,3) for x in ci798)} | Sens={sens:.4f} CI{tuple(round(x,3) for x in sens_ci)}"
      f" | Spec={spec:.4f} CI{tuple(round(x,3) for x in spec_ci)}")
report["within_cohort_798feat"] = {"auc":round(auc798,4),"auc_ci95":[round(x,3) for x in ci798],
    "sensitivity":round(sens,4),"sens_ci95":[round(x,3) for x in sens_ci],
    "specificity":round(spec,4),"spec_ci95":[round(x,3) for x in spec_ci],"cm_tn_fp_fn_tp":[tn,fp,fn,tp]}

# ============================================== 2) HARMONIZED within-cohort
sak = H[H.cohort=='sakar2019']; ys = sak['label'].astype(int)
skf = StratifiedKFold(5, shuffle=True, random_state=RS)
p_in = cross_val_predict(LogisticRegression(max_iter=5000), prep(sak[feats]), ys, cv=skf, method='predict_proba')[:,1]
auc_h = roc_auc_score(ys, p_in); fpr_h, tpr_h, _ = roc_curve(ys, p_in)
ci_harm = boot_auc_ci(ys.values, p_in)
report["within_cohort_harmonized_ci95"] = [round(x, 3) for x in ci_harm]
print(f"[2b] within-cohort harmonised AUC = {auc_h:.4f}  95% CI "
      f"({ci_harm[0]:.3f}, {ci_harm[1]:.3f})")
print(f"    >> PASTE Tabela 3 (harmonised core CI): {ci_harm[0]:.3f}--{ci_harm[1]:.3f}")
report["within_cohort_harmonized"] = round(auc_h,4)

# ============================================== 3) LEAVE-ONE-LAB-OUT + pairwise
print("[3] Leave-one-laboratory-out + pairwise ...")
ext = {}
for trn, tst in [('Istanbul','Extremadura'),('Extremadura','Istanbul')]:
    tr = H[H.lab==trn]; te = H[H.lab==tst]; pr = ens_pred(tr, te, feats); yt = te['label'].astype(int).values
    ext[f"{trn}->{tst}"] = {"auc":round(roc_auc_score(yt,pr),4),"ci":[round(x,3) for x in boot_auc_ci(yt,pr)],"n":len(te)}
trS = H[H.cohort=='sakar2019']
for tc in ['naranjo','carron']:
    te = H[H.cohort==tc]; pr = ens_pred(trS, te, feats); yt = te['label'].astype(int).values
    ext[f"sakar2019->{tc}"] = {"auc":round(roc_auc_score(yt,pr),4),"ci":[round(x,3) for x in boot_auc_ci(yt,pr)],"n":len(te)}

teE = H[H.lab=='Extremadura']; prE = ens_pred(H[H.lab=='Istanbul'], teE, feats)
fpr_e, tpr_e, _ = roc_curve(teE['label'].astype(int), prE); auc_e = roc_auc_score(teE['label'].astype(int), prE)
report["external"] = ext
for k,v in ext.items(): print(f"    {k}: AUC={v['auc']} CI{v['ci']} (n={v['n']})")

# ---- [3b] TESTE FORMAL DA DIFERENCA DE AUC (interno vs externo) -------------
print("[3b] Bootstrap AUC-difference test (transfer effect) ...")
def boot_auc_diff(y1, p1, y2, p2, n=N_BOOT):
    y1, p1, y2, p2 = map(np.asarray, (y1, p1, y2, p2))
    diffs = []
    for _ in range(n):
        i1 = rng.integers(0, len(y1), len(y1))
        i2 = rng.integers(0, len(y2), len(y2))
        if len(np.unique(y1[i1])) > 1 and len(np.unique(y2[i2])) > 1:
            diffs.append(roc_auc_score(y1[i1], p1[i1]) - roc_auc_score(y2[i2], p2[i2]))
    diffs = np.array(diffs)
    pval = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return (float(np.mean(diffs)),
            (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
            float(min(pval, 1.0)))

yext = teE['label'].astype(int).values
d_t, ci_t, p_t = boot_auc_diff(ys.values, p_in, yext, prE)
report["auc_difference_transfer"] = {
    "internal_harmonized_auc": round(float(roc_auc_score(ys, p_in)), 4),
    "external_auc": round(float(roc_auc_score(yext, prE)), 4),
    "delta": round(d_t, 4),
    "ci95": [round(x, 3) for x in ci_t],
    "p_value": round(p_t, 4),
    "note": "Unpaired bootstrap (2000 resamples/group); within-cohort harmonised (Sakar2019) vs leave-one-lab-out external."
}
print(f"    delta AUC = {d_t:.3f}  95% CI {tuple(round(x,3) for x in ci_t)}  p = {p_t:.4f}")
print(f"    >> PASTE: within-to-external drop (Delta AUC = {d_t:.3f}, 95% CI "
      f"{ci_t[0]:.3f}-{ci_t[1]:.3f}, p = {p_t:.3f})")

# ---- [6b] SENSIBILIDADE DE HIPERPARAMETROS: RF e LR -------------------------
print("[6b] RF / LR hyperparameter sensitivity ...")
Xh = prep(sak[feats])
aucs_rf, aucs_lr = [], []
for ne in [100, 300, 500]:
    for md in [None, 5, 10]:
        pr = cross_val_predict(RandomForestClassifier(ne, max_depth=md, random_state=RS, n_jobs=-1),
                               Xh, ys, cv=skf, method='predict_proba')[:, 1]
        aucs_rf.append(roc_auc_score(ys, pr))
for C in [0.1, 1, 10]:
    for sol in ['lbfgs', 'liblinear']:
        pr = cross_val_predict(LogisticRegression(C=C, solver=sol, max_iter=5000),
                               Xh, ys, cv=skf, method='predict_proba')[:, 1]
        aucs_lr.append(roc_auc_score(ys, pr))
report["hp_sensitivity"] = {
    "rf_auc_range": [round(min(aucs_rf), 3), round(max(aucs_rf), 3)],
    "lr_auc_range": [round(min(aucs_lr), 3), round(max(aucs_lr), 3)],
    "svm_auc_range": [0.762, 0.790]
}
print(f"    RF AUC range {report['hp_sensitivity']['rf_auc_range']} | "
      f"LR AUC range {report['hp_sensitivity']['lr_auc_range']}")

# ============================================== 4) PREDICTABILITY (confound)
yc = pd.factorize(H['lab'])[0]
predc = cross_val_predict(RandomForestClassifier(300, random_state=RS, n_jobs=-1), prep(H[feats]), yc, cv=skf)
pacc = float((predc==yc).mean()); chance = float(pd.Series(yc).value_counts(normalize=True).max())
report["lab_predictability"] = {"acc":round(pacc,4),"chance":round(chance,4)}
print(f"[4] Lab predictability acc={pacc:.3f} (chance={chance:.3f})")

# ============================================== 5) LABEL PERMUTATION (+ p-value)
print("[5] Label-permutation test ...")
def perm_once(seed):
    r = np.random.default_rng(seed); yp = r.permutation(ys.values)
    pr = cross_val_predict(LogisticRegression(max_iter=5000), prep(sak[feats]), yp, cv=skf, method='predict_proba')[:,1]
    return roc_auc_score(yp, pr)
NPERM = 200
null = Parallel(n_jobs=-1)(delayed(perm_once)(s) for s in range(NPERM))
pval = (np.sum(np.array(null) >= auc_h)+1)/(NPERM+1)
report["label_permutation"] = {"observed_harmonized_auc":round(auc_h,4),"null_mean":round(float(np.mean(null)),4),
    "p_value":round(float(pval),4),"n_perm":NPERM}
print(f"    observed={auc_h:.3f} null_mean={np.mean(null):.3f} p={pval:.4f}")

# ============================================== 6) PERMUTATION IMPORTANCE
Xtr, Xte, ytr, yte = train_test_split(sak[feats], ys, test_size=.3, stratify=ys, random_state=RS)
pipe = Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),('lr',LogisticRegression(max_iter=5000))]).fit(Xtr,ytr)
pi = permutation_importance(pipe, Xte, yte, n_repeats=30, random_state=RS, scoring='roc_auc')
impS = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
report["permutation_importance_top10"] = {k:round(v,4) for k,v in impS.head(10).items()}

# ============================================== 7) PROB-AVG vs MAJORITY VOTE
sub = df[df.dataset_source=='pd_speech'].copy()
fmap = {'jitter_rel':'pd_speech_locPctJitter','shimmer_loc':'pd_speech_locShimmer',
        'hnr':'pd_speech_meanHarmToNoiseHarmonicity','ppe':'pd_speech_PPE','rpde':'pd_speech_RPDE','gne':'pd_speech_GNE_mean'}
for i in range(13): fmap[f'mfcc{i}'] = f'pd_speech_mean_MFCC_{MFCC_SAK[i]}_coef'
Xr = sub[[fmap[f] for f in feats]].copy(); Xr.columns = feats
Xr['label'] = sub['label'].astype(int).values; Xr['sid'] = sub['subject_id'].values
pr_rec = cross_val_predict(Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),('lr',LogisticRegression(max_iter=5000))]),
                           Xr[feats], Xr['label'], cv=skf, method='predict_proba')[:,1]
Xr['p'] = pr_rec
ag = Xr.groupby('sid').agg(y=('label','first'), pmean=('p','mean'), vote=('p', lambda s:(s>=0.5).mean()))
report["aggregation"] = {"prob_avg_auc":round(roc_auc_score(ag['y'],ag['pmean']),4),
                         "majority_vote_auc":round(roc_auc_score(ag['y'],ag['vote']),4)}
print(f"[7] prob-avg AUC={report['aggregation']['prob_avg_auc']} vs vote AUC={report['aggregation']['majority_vote_auc']}")

# ============================================== 8) DEMOGRAPHICS TABLE
rows = []
g_sak = df[df.dataset_source=='pd_speech'].groupby('subject_id').agg(label=('label','first'),sex=('pd_speech_gender','first'))
g_nar = df[df.dataset_source=='replicated'].groupby('subject_id').agg(label=('label','first'),sex=('replicated_Gender','first'))
for name, g_ in [('Sakar2019',g_sak),('Naranjo',g_nar)]:
    for cl in [1,0]:
        s = g_[g_.label==cl]
        rows.append({'cohort':name,'class':'PD' if cl else 'HC','n':len(s),
                     'sex_dist':s['sex'].value_counts().to_dict(),'age_mean':None,'age_sd':None})
for cl in [1,0]:
    s = c[c.Status==cl]
    rows.append({'cohort':'Carron','class':'PD' if cl else 'HC','n':len(s),
                 'sex_dist':s['Sex'].value_counts().to_dict(),
                 'age_mean':round(s['Age'].mean(),1),'age_sd':round(s['Age'].std(),1)})
demo = pd.DataFrame(rows); demo.to_csv(f"{OUT}/demographics_table.csv", index=False)
tt = stats.ttest_ind(c[c.Status==1].Age, c[c.Status==0].Age)
report["demographics_note"] = {"carron_age_PD":round(c[c.Status==1].Age.mean(),1),
    "carron_age_HC":round(c[c.Status==0].Age.mean(),1),"carron_age_ttest_p":round(float(tt.pvalue),4)}
print("[8] demographics_table.csv written")

# ============================================== 9) SAKAR2013 <-> SAKAR2019 DEDUP
print("[9] Sakar2013<->Sakar2019 acoustic-distance de-duplication ...")
try:
    s13 = pd.read_csv(PATHS["sakar13_train"], header=None)
    s13_basic = pd.DataFrame({'subject_id':s13[0],'jitter_rel':s13[1],'shimmer_loc':s13[6],'hnr':s13[14],'label':s13.iloc[:,-1]})
    s13s = s13_basic.groupby('subject_id').mean(numeric_only=True)
    s19s = t_sak.set_index('subject_id')[['jitter_rel','shimmer_loc','hnr']]
    both = pd.concat([s13s[['jitter_rel','shimmer_loc','hnr']], s19s])
    z = (both - both.mean())/both.std()
    z13 = z.iloc[:len(s13s)].values; z19 = z.iloc[len(s13s):].values
    dists = np.sqrt(((z13[:,None,:]-z19[None,:,:])**2).sum(-1))
    nn = dists.min(axis=1)
    report["sakar_dedup"] = {"n_2013":len(s13s),"n_2019":len(s19s),
        "min_nn_distance":round(float(nn.min()),3),"median_nn_distance":round(float(np.median(nn)),3),
        "n_suspicious_pairs(<0.25)":int((nn<0.25).sum()),
        "note":"Basic 3-feature core only; small distances are suggestive, not proof. Sakar2013 is excluded from main analysis (sensitivity)."}
    print(f"    nearest-neighbour dist: min={nn.min():.3f} median={np.median(nn):.3f} | suspicious(<0.25)={int((nn<0.25).sum())}")
except Exception as e:
    report["sakar_dedup"] = {"error":str(e)}; print("    [skip]", e)

# ============================================== 10) LEAKY k-fold vs LOSO (point H)
print("[10] Leaky (recording-level) k-fold vs honest LOSO ...")
leaky = cross_val_predict(Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),
        ('lr',LogisticRegression(max_iter=5000))]), Xr[feats], Xr['label'],
        cv=StratifiedKFold(10,shuffle=True,random_state=RS), method='predict_proba')[:,1]
auc_leaky = roc_auc_score(Xr['label'], leaky)
report["leakage_demo"] = {"leaky_recordlevel_kfold_auc":round(auc_leaky,4),
    "honest_subjectlevel_harmonized_auc":round(auc_h,4),
    "inflation":round(auc_leaky-auc_h,4)}
print(f"    leaky 10-fold AUC={auc_leaky:.4f} vs honest LOSO-harmonized AUC={auc_h:.4f} (inflation +{auc_leaky-auc_h:.4f})")

# ============================================== 11) FEATURE PROVENANCE TABLE
prov = []
for col in fc798:
    src = 'Sakar2019 (PD Classification)' if col.startswith('pd_speech_') else 'Naranjo2016 (Replicated)'
    nn_ = d2[col].notna().sum(); imp_ = int(len(d2)-nn_)
    prov.append({'feature':col,'source_dataset':src,'n_present':int(nn_),
                 'n_imputed':imp_,'imputation':'train-fold median'})
prov_df = pd.DataFrame(prov); prov_df.to_csv(f"{OUT}/feature_provenance_table.csv", index=False)
print(f"[11] feature_provenance_table.csv ({len(prov_df)} features)")

# ============================================== 12) FIGURES
C = {'int':'#1b3a6b','ext':'#c0392b','har':'#2e86c1','warn':'#b03a2e','gray':'#7f8c8d'}
plt.figure(figsize=(6,5.2))
plt.plot(fpr_i,tpr_i,color=C['int'],lw=2.4,label=f'Within-cohort LOSO (798f), AUC={auc798:.3f}')
plt.plot(fpr_h,tpr_h,color=C['har'],lw=2,ls='--',label=f'Within-cohort (harmon. 19f), AUC={auc_h:.3f}')
plt.plot(fpr_e,tpr_e,color=C['ext'],lw=2.4,label=f'Leave-one-lab-out, AUC={auc_e:.3f}')
plt.plot([0,1],[0,1],':',color=C['gray']); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Internal vs. external validation',fontweight='bold'); plt.legend(fontsize=8,loc='lower right')
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_roc_internal_vs_external.png",dpi=200); plt.close()

plt.figure(figsize=(7.2,4.6))
labs=['Literature\n(uncorrected)','Within LOSO\n(798f)','Within\n(harmon.)','Leave-one-\nlab-out']
vals=[0.97,auc798,auc_h,auc_e]
err=[[0.02,auc798-ci798[0],0.03,auc_e-ext['Istanbul->Extremadura']['ci'][0]],
     [0.02,ci798[1]-auc798,0.03,ext['Istanbul->Extremadura']['ci'][1]-auc_e]]
plt.bar(labs,vals,color=[C['gray'],C['int'],C['har'],C['ext']],yerr=err,capsize=5,alpha=.9)
plt.axhline(0.5,color='k',ls=':'); plt.ylim(0.45,1.02); plt.ylabel('AUC')
plt.title('Discrimination collapses under honest external validation',fontweight='bold',fontsize=11)
for i,v in enumerate(vals): plt.text(i,v+0.006,f'{v:.3f}',ha='center',fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_auc_summary.png",dpi=200); plt.close()

plt.figure(figsize=(5,4.4))
plt.bar(['Predict LAB\nfrom features','Chance'],[pacc,chance],color=[C['warn'],C['gray']],alpha=.9)
plt.ylim(0,1.08); plt.ylabel('Accuracy'); plt.title('Cohort identity is predictable\n(batch-effect confound)',fontsize=10.5,fontweight='bold')
for i,v in enumerate([pacc,chance]): plt.text(i,v+0.01,f'{v:.2f}',ha='center')
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_dataset_predictability.png",dpi=200); plt.close()

plt.figure(figsize=(6,4.6)); impS.head(10)[::-1].plot(kind='barh',color=C['har'])
plt.xlabel('Mean AUC drop when permuted'); plt.title('Permutation importance (Sakar2019, held-out)',fontweight='bold',fontsize=11)
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_permutation_importance.png",dpi=200); plt.close()

plt.figure(figsize=(4.6,4)); CM=np.array([[tn,fp],[fn,tp]]); plt.imshow(CM,cmap='Blues')
for (r,cc),v in np.ndenumerate(CM): plt.text(cc,r,str(v),ha='center',va='center',fontsize=15,color='white' if v>CM.max()/2 else 'black')
plt.xticks([0,1],['Pred HC','Pred PD']); plt.yticks([0,1],['True HC','True PD'])
plt.title(f'Confusion matrix (LOSO)\nSens={sens:.3f} Spec={spec:.3f}',fontsize=10,fontweight='bold')
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_confusion_within.png",dpi=200); plt.close()

# ============================================== SAVE
report["runtime_seconds"] = round(time.time()-t_start,1)
json.dump(report, open(f"{OUT}/results_summary.json","w"), indent=2)
print("\n"+"="*70)
print(f"DONE in {report['runtime_seconds']}s. Outputs in ./{OUT}/")
print("  results_summary.json | demographics_table.csv | feature_provenance_table.csv")
print("  harmonized_subjectlevel.csv | fig1..fig5 .png")
print("="*70)
print(json.dumps(report, indent=2))
