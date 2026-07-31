import subprocess
subprocess.run(['pip', 'install', 'xgboost', 'shap', 'scikit-learn',
                'pandas', 'numpy', 'scipy', 'openpyxl', 'matplotlib',
                'optuna', '--quiet'], check=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json, glob, warnings, optuna
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, average_precision_score)
from scipy.stats import wilcoxon
import shap

# ── This version uses the ORIGINAL imbalanced data without SMOTE.
# ── Focal loss handles the imbalance directly — that is its purpose.
# ── 5 seeds x 5 folds = 25 measurements for Wilcoxon significance.

SEEDS   = [42, 123, 456, 789, 1024]
METRICS = ['acc', 'f1', 'auc_roc', 'auc_pr']

files_found = glob.glob("*.xlsx") + glob.glob("*.xls")
FILE = files_found[0]
print("File found:", FILE)

df = pd.read_excel(FILE, sheet_name=0)
df.columns = [
    'Timestamp','Level','Department','Devices',
    'PwdReuse','WeakPwd','TwoFA','PwdChange',
    'ClickLinks','LoginEmail','PhishEmail','PhishResponse',
    'UnknownApps','PublicWiFi','OSUpdates','Antivirus','Comment'
]
df = df.drop(columns=['Timestamp','Comment'], errors='ignore')
df = df.dropna(subset=['Level','PwdReuse','WeakPwd'])

scale_cols = ['PwdReuse','WeakPwd','ClickLinks','LoginEmail','UnknownApps']
df[scale_cols] = df[scale_cols].apply(pd.to_numeric, errors='coerce').fillna(3)
df['RiskScore'] = df[scale_cols].sum(axis=1)
df['RiskLevel'] = df['RiskScore'].apply(
    lambda s: 0 if s <= 12 else (1 if s <= 17 else 2))

cat_cols = ['Level','Department','Devices','TwoFA','PwdChange',
            'PhishEmail','PhishResponse','PublicWiFi','OSUpdates','Antivirus']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str).str.strip())

feature_cols = ['Level','Department','Devices','PwdReuse','WeakPwd',
                'TwoFA','PwdChange','ClickLinks','LoginEmail','PhishEmail',
                'PhishResponse','UnknownApps','PublicWiFi','OSUpdates','Antivirus']

X = df[feature_cols]
y = df['RiskLevel'].astype(int)
print(f"Shape: {X.shape} | Classes: {dict(y.value_counts().sort_index())}")

# ── TUNE FOCAL LOSS ON ORIGINAL IMBALANCED DATA ───────────────
print("\nTuning focal loss parameters via Bayesian optimisation...")

def tune_objective(trial):
    gamma = trial.suggest_float('gamma', 0.5, 5.0)
    alpha = trial.suggest_float('alpha', 0.25, 0.95)

    def focal(y_true, y_pred):
        p    = 1.0 / (1.0 + np.exp(-y_pred))
        grad = alpha * (1 - p) ** gamma * (p - y_true)
        hess = alpha * (1 - p) ** gamma * p * (1 - p)
        return grad, hess

    skf    = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for tr, te in skf.split(X, y):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        proba_all = np.zeros((len(Xte), 3))
        for cls in range(3):
            m = XGBClassifier(n_estimators=300, max_depth=6,
                              learning_rate=0.05, random_state=42,
                              use_label_encoder=False)
            m.set_params(objective=focal)
            m.fit(Xtr, (ytr == cls).astype(int), verbose=False)
            proba_all[:, cls] = m.predict_proba(Xte)[:, 1]
        proba_all = proba_all / proba_all.sum(axis=1, keepdims=True).clip(1e-9)
        scores.append(f1_score(yte, np.argmax(proba_all, axis=1), average='macro'))
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(tune_objective, n_trials=30)
BEST_GAMMA = study.best_params['gamma']
BEST_ALPHA = study.best_params['alpha']
print(f"Best gamma: {BEST_GAMMA:.4f} | Best alpha: {BEST_ALPHA:.4f}")

def focal_loss(y_true, y_pred):
    p    = 1.0 / (1.0 + np.exp(-y_pred))
    grad = BEST_ALPHA * (1 - p) ** BEST_GAMMA * (p - y_true)
    hess = BEST_ALPHA * (1 - p) ** BEST_GAMMA * p * (1 - p)
    return grad, hess

# ── 5 SEEDS x 5 FOLDS ON ORIGINAL IMBALANCED DATA ────────────
print("\nRunning 5 seeds x 5 folds on original imbalanced data...")

BASE_PARAMS = dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8,
                   eval_metric='mlogloss', use_label_encoder=False)
GK_PARAMS   = dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8,
                   use_label_encoder=False)

res_b  = {m: [] for m in METRICS}
res_gk = {m: [] for m in METRICS}

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        bm = XGBClassifier(**BASE_PARAMS, random_state=seed)
        bm.fit(Xtr, ytr, verbose=False)
        preds = bm.predict(Xte)
        proba = bm.predict_proba(Xte)
        res_b['acc'].append(accuracy_score(yte, preds))
        res_b['f1'].append(f1_score(yte, preds, average='macro'))
        res_b['auc_roc'].append(roc_auc_score(yte, proba,
            multi_class='ovr', average='macro'))
        res_b['auc_pr'].append(average_precision_score(
            pd.get_dummies(yte), proba, average='macro'))

        proba_all = np.zeros((len(Xte), 3))
        for cls in range(3):
            gk = XGBClassifier(**GK_PARAMS, random_state=seed)
            gk.set_params(objective=focal_loss)
            gk.fit(Xtr, (ytr == cls).astype(int), verbose=False)
            proba_all[:, cls] = gk.predict_proba(Xte)[:, 1]
        proba_all = proba_all / proba_all.sum(axis=1, keepdims=True).clip(1e-9)
        preds_gk  = np.argmax(proba_all, axis=1)
        res_gk['acc'].append(accuracy_score(yte, preds_gk))
        res_gk['f1'].append(f1_score(yte, preds_gk, average='macro'))
        res_gk['auc_roc'].append(roc_auc_score(yte, proba_all,
            multi_class='ovr', average='macro'))
        res_gk['auc_pr'].append(average_precision_score(
            pd.get_dummies(yte), proba_all, average='macro'))

# ── COMPARISON TABLE ──────────────────────────────────────────
metric_labels = {'acc':'Accuracy','f1':'Macro F1-Score',
                 'auc_roc':'AUC-ROC','auc_pr':'AUC-PR'}
rows = []
for m, label in metric_labels.items():
    b = np.array(res_b[m])
    g = np.array(res_gk[m])
    try:    _, p = wilcoxon(b, g)
    except: p = 1.0
    delta = g.mean() - b.mean()
    rows.append({
        'Metric':           label,
        'XGBoost Baseline': f'{b.mean():.4f} +- {b.std():.4f}',
        'GK-XGBoost':       f'{g.mean():.4f} +- {g.std():.4f}',
        'Delta':            f'{delta:+.4f}',
        'p-value':          f'{p:.4f}',
        'Sig. (p<0.05)':    'Yes' if p < 0.05 else 'No',
        'Better?':          'YES' if delta > 0 else 'NO'
    })

table = pd.DataFrame(rows)
print("\nTable 1 — KNUST Field Data Reengineering Results:")
print(table.to_string(index=False))
table.to_csv('results_knust_reengineering.csv', index=False)

# ── SHAP ──────────────────────────────────────────────────────
bm_final = XGBClassifier(**BASE_PARAMS, random_state=42)
bm_final.fit(X, y, verbose=False)
gk_final = XGBClassifier(**GK_PARAMS, random_state=42)
gk_final.set_params(objective=focal_loss)
gk_final.fit(X, (y == 2).astype(int), verbose=False)

def flatten_shap(vals):
    arr = np.array(vals)
    if arr.ndim == 3:   return np.abs(arr).mean(axis=(0, 2))
    elif arr.ndim == 2: return np.abs(arr).mean(axis=0)
    else:               return np.abs(arr).mean(axis=0)

mean_b  = flatten_shap(shap.TreeExplainer(bm_final).shap_values(X))
mean_gk = flatten_shap(shap.TreeExplainer(gk_final).shap_values(X))

shap_df = pd.DataFrame({
    'Feature':         feature_cols,
    'SHAP_Baseline':   mean_b[:len(feature_cols)],
    'SHAP_GK_XGBoost': mean_gk[:len(feature_cols)]
}).sort_values('SHAP_GK_XGBoost', ascending=False)
shap_df.to_csv('shap_knust.csv', index=False)

json.dump({'gamma': BEST_GAMMA, 'alpha': BEST_ALPHA,
           'feature_cols': feature_cols},
          open('gk_params.json', 'w'))

# ── PLOT ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(4); w = 0.35
bv = [np.mean(res_b[m])  for m in METRICS]
gv = [np.mean(res_gk[m]) for m in METRICS]
axes[0].bar(x-w/2, bv, w, label='XGBoost Baseline', color='steelblue')
axes[0].bar(x+w/2, gv, w, label='GK-XGBoost',       color='darkorange')
axes[0].set_xticks(x)
axes[0].set_xticklabels(list(metric_labels.values()), rotation=15, ha='right')
axes[0].set_ylim(0.5, 1.05)
axes[0].set_title('KNUST Field Data: Baseline vs GK-XGBoost')
axes[0].legend(); axes[0].set_ylabel('Score')

top10 = shap_df.head(10); yp = np.arange(len(top10))
axes[1].barh(yp-0.2, top10['SHAP_Baseline'],   0.4, label='Baseline',    color='steelblue')
axes[1].barh(yp+0.2, top10['SHAP_GK_XGBoost'], 0.4, label='GK-XGBoost', color='darkorange')
axes[1].set_yticks(yp); axes[1].set_yticklabels(top10['Feature'])
axes[1].set_title('SHAP Feature Importance Top 10')
axes[1].legend(); axes[1].set_xlabel('Mean |SHAP value|')

plt.tight_layout()
plt.savefig('reengineering_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nReengineering complete. Files saved:")
print("  results_knust_reengineering.csv")
print("  shap_knust.csv")
print("  gk_params.json")
print("  reengineering_results.png")
