import subprocess
subprocess.run(['pip', 'install', 'xgboost', 'shap', 'scikit-learn',
                'pandas', 'numpy', 'scipy', 'matplotlib', 'openpyxl',
                '--quiet'], check=True)

import json, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             average_precision_score, precision_score,
                             recall_score, confusion_matrix,
                             classification_report)
from scipy.stats import wilcoxon
from scipy import stats

# ============================================================
# LOAD KNUST DATA
# ============================================================
knust_file = (glob.glob("*.xlsx") + glob.glob("*.xls"))[0]
print(f"File: {knust_file}")

raw = pd.read_excel(knust_file, sheet_name=0)
raw.columns = [
    'Timestamp','Level','Department','Devices',
    'PwdReuse','WeakPwd','TwoFA','PwdChange',
    'ClickLinks','LoginEmail','PhishEmail','PhishResponse',
    'UnknownApps','PublicWiFi','OSUpdates','Antivirus','Comment']
raw = raw.drop(columns=['Timestamp','Comment'], errors='ignore')
raw = raw.dropna(subset=['Level','PwdReuse','WeakPwd'])

scale_cols = ['PwdReuse','WeakPwd','ClickLinks','LoginEmail','UnknownApps']
raw[scale_cols] = raw[scale_cols].apply(pd.to_numeric, errors='coerce').fillna(3)
raw['RiskScore'] = raw[scale_cols].sum(axis=1)
raw['RiskLevel'] = raw['RiskScore'].apply(
    lambda s: 0 if s<=12 else (1 if s<=17 else 2))

cat_cols = ['Level','Department','Devices','TwoFA','PwdChange',
            'PhishEmail','PhishResponse','PublicWiFi','OSUpdates','Antivirus']
le = LabelEncoder()
for col in cat_cols:
    raw[col] = le.fit_transform(raw[col].astype(str).str.strip())

feature_cols = ['Level','Department','Devices','PwdReuse','WeakPwd',
                'TwoFA','PwdChange','ClickLinks','LoginEmail','PhishEmail',
                'PhishResponse','UnknownApps','PublicWiFi','OSUpdates','Antivirus']
X = raw[feature_cols]
y = raw['RiskLevel'].astype(int)

scaler = RobustScaler()
X_sc   = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

SEEDS   = [42, 123, 456, 789, 1024]
GAMMA   = 2.2151
ALPHA   = 0.7183
MCW     = 1
N_CLS   = 3
CLASSES = ['Low Risk', 'Medium Risk', 'High Risk']

def make_mc_focal(g, a, n):
    def focal(yt, yp):
        ns = len(yt)
        yp = yp.reshape(ns, n)
        yp = yp - yp.max(axis=1, keepdims=True)
        sm = np.exp(yp)/np.exp(yp).sum(axis=1, keepdims=True)
        oh = np.zeros_like(sm)
        oh[np.arange(ns), yt.astype(int)] = 1
        pt = (sm*oh).sum(axis=1, keepdims=True)
        fw = a*(1-pt)**g
        return (fw*(sm-oh)).flatten(), (fw*sm*(1-sm)).flatten()
    return focal

focal_fn = make_mc_focal(GAMMA, ALPHA, N_CLS)

BASE_PARAMS = dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8,
                   objective='multi:softmax', num_class=N_CLS,
                   eval_metric='mlogloss', use_label_encoder=False)
GK_PARAMS   = dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8,
                   objective=focal_fn, num_class=N_CLS,
                   eval_metric='mlogloss', use_label_encoder=False,
                   min_child_weight=MCW)

# ============================================================
# RUN 5 SEEDS x 5 FOLDS — collect ALL fold-level scores
# ============================================================
print("\nRunning 5 seeds x 5 folds to collect fold-level scores...")

fold_b  = {'acc':[],'f1':[],'auc_roc':[],'auc_pr':[]}
fold_gk = {'acc':[],'f1':[],'auc_roc':[],'auc_pr':[]}

# Also collect predictions for confusion matrix and per-class metrics
all_y_true_b  = []; all_y_pred_b  = []
all_y_true_gk = []; all_y_pred_gk = []
train_times_b  = []; train_times_gk = []

import time
y_arr = np.array(y)

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X_sc, y_arr):
        Xtr,Xte = X_sc.iloc[tr], X_sc.iloc[te]
        ytr,yte = y_arr[tr], y_arr[te]

        # Baseline
        t0 = time.time()
        bm = XGBClassifier(**BASE_PARAMS, random_state=seed)
        bm.fit(Xtr, ytr, verbose=False)
        train_times_b.append(time.time()-t0)
        preds_b = bm.predict(Xte)
        proba_b = bm.predict_proba(Xte)
        fold_b['acc'].append(accuracy_score(yte, preds_b))
        fold_b['f1'].append(f1_score(yte, preds_b, average='macro'))
        fold_b['auc_roc'].append(roc_auc_score(yte, proba_b, multi_class='ovr', average='macro'))
        fold_b['auc_pr'].append(average_precision_score(pd.get_dummies(yte), proba_b, average='macro'))
        all_y_true_b.extend(yte); all_y_pred_b.extend(preds_b)

        # GK-XGBoost
        t0 = time.time()
        gk = XGBClassifier(**GK_PARAMS, random_state=seed)
        gk.fit(Xtr, ytr, verbose=False)
        train_times_gk.append(time.time()-t0)
        preds_gk = gk.predict(Xte)
        proba_gk = gk.predict_proba(Xte)
        fold_gk['acc'].append(accuracy_score(yte, preds_gk))
        fold_gk['f1'].append(f1_score(yte, preds_gk, average='macro'))
        fold_gk['auc_roc'].append(roc_auc_score(yte, proba_gk, multi_class='ovr', average='macro'))
        fold_gk['auc_pr'].append(average_precision_score(pd.get_dummies(yte), proba_gk, average='macro'))
        all_y_true_gk.extend(yte); all_y_pred_gk.extend(preds_gk)

print("Done.")

# ============================================================
# FIGURE R1: BOX PLOT — Cross-Validation Stability (25 folds)
# ============================================================
print("\nGenerating Figure R1: Box plot (CV stability)...")

fig, axes = plt.subplots(1, 4, figsize=(14, 6))
metrics  = ['acc','f1','auc_roc','auc_pr']
mlabels  = ['Accuracy','Macro F1-Score','AUC-ROC','AUC-PR']
positions = [1, 2]

for ax, m, ml in zip(axes, metrics, mlabels):
    b_vals  = np.array(fold_b[m])
    gk_vals = np.array(fold_gk[m])

    bp = ax.boxplot([b_vals, gk_vals],
                    positions=positions,
                    widths=0.5,
                    patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(color='black', linewidth=1),
                    capprops=dict(color='black', linewidth=1.5),
                    flierprops=dict(marker='o', color='black', markersize=4),
                    boxprops=dict(linewidth=1))

    # Black/white fills
    bp['boxes'][0].set_facecolor('white')
    bp['boxes'][0].set_edgecolor('black')
    bp['boxes'][1].set_facecolor('#555555')
    bp['boxes'][1].set_edgecolor('black')

    # Diamond mean markers
    ax.plot(1, b_vals.mean(),  marker='D', color='black', markersize=6, zorder=5)
    ax.plot(2, gk_vals.mean(), marker='D', color='white',
            markeredgecolor='black', markersize=6, zorder=5)

    # Jitter points
    np.random.seed(42)
    ax.scatter(np.random.normal(1, 0.05, len(b_vals)),  b_vals,
               color='black', alpha=0.3, s=12, zorder=4)
    ax.scatter(np.random.normal(2, 0.05, len(gk_vals)), gk_vals,
               color='black', alpha=0.3, s=12, zorder=4)

    ax.set_title(ml, fontsize=12, fontweight='bold')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Baseline\nXGBoost', 'GK-XGBoost'], fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    _, pv = wilcoxon(b_vals, gk_vals)
    ax.text(1.5, gk_vals.max() + 0.002, f'p={pv:.4f}',
            ha='center', fontsize=9, fontstyle='italic')

    ax.text(1, b_vals.min()-0.004,
            f'Mean={b_vals.mean():.4f}\nSD={b_vals.std():.4f}',
            ha='center', fontsize=8)
    ax.text(2, gk_vals.min()-0.004,
            f'Mean={gk_vals.mean():.4f}\nSD={gk_vals.std():.4f}',
            ha='center', fontsize=8)

white_p = mpatches.Patch(facecolor='white', edgecolor='black', label='Baseline XGBoost')
gray_p  = mpatches.Patch(facecolor='#555555', edgecolor='black', label='GK-XGBoost')
fig.legend(handles=[white_p, gray_p], loc='upper center',
           ncol=2, fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.02))

plt.suptitle('Figure R1. Cross-validation stability: distribution of 25 fold scores\n(5 seeds x 5 folds). Diamond markers denote fold means.',
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig('Figure_R1_BoxPlot_CV_Stability.png', dpi=200, bbox_inches='tight',
            facecolor='white')
plt.show()
print("Saved: Figure_R1_BoxPlot_CV_Stability.png")

# ============================================================
# FIGURE R2: NORMALIZED CONFUSION MATRICES
# ============================================================
print("\nGenerating Figure R2: Confusion matrices...")

cm_b  = confusion_matrix(all_y_true_b,  all_y_pred_b,  normalize='true')
cm_gk = confusion_matrix(all_y_true_gk, all_y_pred_gk, normalize='true')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, cm, title in zip(axes,
    [cm_b, cm_gk],
    ['(a) Baseline XGBoost', '(b) GK-XGBoost (proposed)']):

    im = ax.imshow(cm, interpolation='nearest', cmap='Greys',
                   vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(N_CLS))
    ax.set_yticks(range(N_CLS))
    ax.set_xticklabels(['Low', 'Medium', 'High'], fontsize=11)
    ax.set_yticklabels(['Low', 'Medium', 'High'], fontsize=11)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')

    for i in range(N_CLS):
        for j in range(N_CLS):
            color = 'white' if cm[i,j] > 0.5 else 'black'
            ax.text(j, i, f'{cm[i,j]:.3f}', ha='center', va='center',
                    fontsize=12, fontweight='bold', color=color)

plt.suptitle('Figure R2. Normalised confusion matrices for (a) Baseline XGBoost and\n'
             '(b) GK-XGBoost on the KNUST field dataset (row-normalised; values represent recall per class).',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('Figure_R2_Confusion_Matrices.png', dpi=200, bbox_inches='tight',
            facecolor='white')
plt.show()
print("Saved: Figure_R2_Confusion_Matrices.png")

# ============================================================
# TABLE: Per-class Precision, Recall, F1
# ============================================================
print("\nGenerating per-class performance table...")

report_b  = classification_report(all_y_true_b,  all_y_pred_b,
                                   target_names=CLASSES, output_dict=True)
report_gk = classification_report(all_y_true_gk, all_y_pred_gk,
                                   target_names=CLASSES, output_dict=True)

rows = []
for cls in CLASSES:
    rows.append({
        'Model':     'Baseline XGBoost',
        'Class':     cls,
        'Precision': f"{report_b[cls]['precision']:.4f}",
        'Recall':    f"{report_b[cls]['recall']:.4f}",
        'F1-Score':  f"{report_b[cls]['f1-score']:.4f}",
        'Support':   int(report_b[cls]['support'])
    })
for cls in CLASSES:
    rows.append({
        'Model':     'GK-XGBoost',
        'Class':     cls,
        'Precision': f"{report_gk[cls]['precision']:.4f}",
        'Recall':    f"{report_gk[cls]['recall']:.4f}",
        'F1-Score':  f"{report_gk[cls]['f1-score']:.4f}",
        'Support':   int(report_gk[cls]['support'])
    })

per_class_df = pd.DataFrame(rows)
per_class_df.to_csv('Table_PerClass_Performance.csv', index=False)
print("\nPer-class performance table:")
print(per_class_df.to_string(index=False))

# ============================================================
# COHEN'S D EFFECT SIZE
# ============================================================
print("\nComputing Cohen's d effect sizes...")

def cohens_d(a, b):
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(b) - np.mean(a)) / pooled_std

print("\nCohen's d effect sizes (GK-XGBoost vs Baseline):")
for m, ml in zip(['acc','f1','auc_roc','auc_pr'], ['Accuracy','Macro F1','AUC-ROC','AUC-PR']):
    d = cohens_d(np.array(fold_b[m]), np.array(fold_gk[m]))
    _, p = wilcoxon(np.array(fold_b[m]), np.array(fold_gk[m]))
    interp = 'small' if abs(d)<0.5 else ('medium' if abs(d)<0.8 else 'large')
    print(f"  {ml:<15}: d = {d:.4f} ({interp}) | p = {p:.4f}")

cohen_df = pd.DataFrame([{
    'Metric': ml,
    'Cohens_d': f"{cohens_d(np.array(fold_b[m]), np.array(fold_gk[m])):.4f}",
    'Interpretation': ('small' if abs(cohens_d(np.array(fold_b[m]),np.array(fold_gk[m])))<0.5
                       else ('medium' if abs(cohens_d(np.array(fold_b[m]),np.array(fold_gk[m])))<0.8
                             else 'large')),
    'p_value': f"{wilcoxon(np.array(fold_b[m]),np.array(fold_gk[m]))[1]:.4f}"
} for m, ml in zip(['acc','f1','auc_roc','auc_pr'],
                   ['Accuracy','Macro F1','AUC-ROC','AUC-PR'])])
cohen_df.to_csv('Table_CohenD_EffectSize.csv', index=False)

# ============================================================
# TRAINING TIME
# ============================================================
print(f"\nTraining time per fold:")
print(f"  Baseline XGBoost:  {np.mean(train_times_b):.2f}s +/- {np.std(train_times_b):.2f}s")
print(f"  GK-XGBoost:        {np.mean(train_times_gk):.2f}s +/- {np.std(train_times_gk):.2f}s")
overhead = ((np.mean(train_times_gk) - np.mean(train_times_b)) / np.mean(train_times_b)) * 100
print(f"  Overhead:          {overhead:.1f}%")

time_df = pd.DataFrame([{
    'Model': 'Baseline XGBoost',
    'Mean_time_s': f"{np.mean(train_times_b):.2f}",
    'SD_s':        f"{np.std(train_times_b):.2f}"
},{
    'Model': 'GK-XGBoost',
    'Mean_time_s': f"{np.mean(train_times_gk):.2f}",
    'SD_s':        f"{np.std(train_times_gk):.2f}"
}])
time_df.to_csv('Table_Training_Time.csv', index=False)

print("\nAll results files saved:")
print("  Figure_R1_BoxPlot_CV_Stability.png")
print("  Figure_R2_Confusion_Matrices.png")
print("  Table_PerClass_Performance.csv")
print("  Table_CohenD_EffectSize.csv")
print("  Table_Training_Time.csv")
