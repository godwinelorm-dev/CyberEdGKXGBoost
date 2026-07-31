import subprocess
subprocess.run(['pip', 'install', 'xgboost', 'shap', 'scikit-learn',
                'pandas', 'numpy', 'matplotlib', 'openpyxl', '--quiet'], check=True)

import glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

import shap
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler

# ============================================================
# ROLE-DIFFERENTIATED SHAP EXPLANATIONS
# Researcher: Koku Godwin Komla | KNUST
#
# This script generates three separate SHAP explanation views
# from the same GK-XGBoost model — one per stakeholder role:
#   Role 1: Students     — individual risk behaviour profile
#   Role 2: Educators    — population-level risk patterns
#   Role 3: IT Admins    — feature-level prediction drivers
#
# HOW TO USE:
# Upload your KNUST Excel file and run this script after
# running GK_XGBoost_Reengineering.py
# ============================================================

GAMMA = 2.2151
ALPHA = 0.7183
MCW   = 1

RISK_LABELS = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
RISK_COLORS = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}

feature_cols = ['Level','Department','Devices','PwdReuse','WeakPwd',
                'TwoFA','PwdChange','ClickLinks','LoginEmail','PhishEmail',
                'PhishResponse','UnknownApps','PublicWiFi','OSUpdates','Antivirus']

feature_friendly = {
    'PwdReuse':      'Password Reuse',
    'WeakPwd':       'Weak Password',
    'TwoFA':         '2FA Usage',
    'PwdChange':     'Password Change Frequency',
    'ClickLinks':    'Clicking Email Links',
    'LoginEmail':    'Logging In via Email Links',
    'PhishEmail':    'Phishing Email Awareness',
    'PhishResponse': 'Phishing Response Behaviour',
    'UnknownApps':   'Unknown App Installation',
    'PublicWiFi':    'Public WiFi Usage',
    'OSUpdates':     'OS Update Frequency',
    'Antivirus':     'Antivirus Protection',
    'Level':         'Level of Study',
    'Department':    'Department',
    'Devices':       'Number of Devices'
}

# ── LOAD AND PREPROCESS KNUST DATA ───────────────────────────
print("Loading KNUST field data...")
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

# Keep original string values for role-based analysis
raw_original = raw.copy()

scale_cols = ['PwdReuse','WeakPwd','ClickLinks','LoginEmail','UnknownApps']
raw[scale_cols] = raw[scale_cols].apply(pd.to_numeric, errors='coerce').fillna(3)
raw['RiskScore'] = raw[scale_cols].sum(axis=1)
raw['RiskLevel'] = raw['RiskScore'].apply(
    lambda s: 0 if s<=12 else (1 if s<=17 else 2))

cat_cols = ['Level','Department','Devices','TwoFA','PwdChange',
            'PhishEmail','PhishResponse','PublicWiFi','OSUpdates','Antivirus']
le = LabelEncoder()
encoders = {}
for col in cat_cols:
    raw[col] = raw[col].astype(str).str.strip()
    le_col = LabelEncoder()
    raw[col] = le_col.fit_transform(raw[col])
    encoders[col] = le_col

X = raw[feature_cols]
y = raw['RiskLevel'].astype(int)

scaler = RobustScaler()
X_sc   = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

print(f"Dataset: {X_sc.shape} | Classes: {dict(y.value_counts().sort_index())}")

# ── TRAIN GK-XGBoost ─────────────────────────────────────────
print("\nTraining GK-XGBoost (focal loss + min_child_weight)...")
n_cls = 3

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

focal_fn = make_mc_focal(GAMMA, ALPHA, n_cls)

# Train baseline for comparison
bm = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8,
                   objective='multi:softmax', num_class=n_cls,
                   eval_metric='mlogloss', use_label_encoder=False,
                   random_state=42)
bm.fit(X_sc, y, verbose=False)

# Train GK-XGBoost
gk = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8,
                   objective=focal_fn, num_class=n_cls,
                   eval_metric='mlogloss', use_label_encoder=False,
                   min_child_weight=MCW, random_state=42)
gk.fit(X_sc, y, verbose=False)
print("Models trained.")

# ── COMPUTE SHAP VALUES ───────────────────────────────────────
print("\nComputing SHAP values...")
explainer_gk = shap.TreeExplainer(gk)
shap_vals_gk = explainer_gk.shap_values(X_sc)  # shape: (n, features) or list

explainer_bm = shap.TreeExplainer(bm)
shap_vals_bm = explainer_bm.shap_values(X_sc)

# Get predictions
y_pred = gk.predict(X_sc)
y_prob = gk.predict_proba(X_sc)

raw['Predicted_Risk'] = y_pred
raw['Risk_Prob_Low']  = y_prob[:, 0] if y_prob.shape[1] == 3 else y_prob[:, 0]
raw['Risk_Prob_Med']  = y_prob[:, 1] if y_prob.shape[1] >= 2 else 0
raw['Risk_Prob_High'] = y_prob[:, 2] if y_prob.shape[1] == 3 else y_prob[:, 1]

# Mean absolute SHAP per feature — handles all output shapes
def get_mean_shap(vals, n_features):
    arr = np.array(vals)
    if arr.ndim == 3:   # (n_classes, n_samples, n_features)
        return np.abs(arr[2]).mean(axis=0)[:n_features]
    elif arr.ndim == 2: # (n_samples, n_features)
        return np.abs(arr).mean(axis=0)[:n_features]
    else:
        return np.abs(arr).flatten()[:n_features]

n_feat       = len(feature_cols)
mean_shap_gk = get_mean_shap(shap_vals_gk, n_feat)
mean_shap_bm = get_mean_shap(shap_vals_bm, n_feat)

print(f"  SHAP GK shape: {np.array(shap_vals_gk).shape}")
print(f"  mean_shap_gk length: {len(mean_shap_gk)}")
print(f"  feature_cols length: {len(feature_cols)}")

# Ensure lengths match
min_len      = min(len(mean_shap_gk), len(mean_shap_bm), len(feature_cols))
mean_shap_gk = mean_shap_gk[:min_len]
mean_shap_bm = mean_shap_bm[:min_len]
feat_cols    = feature_cols[:min_len]

shap_df = pd.DataFrame({
    'Feature':         feat_cols,
    'Friendly_Name':   [feature_friendly.get(f, f) for f in feat_cols],
    'SHAP_GK':         mean_shap_gk,
    'SHAP_Baseline':   mean_shap_bm
}).sort_values('SHAP_GK', ascending=False)

# ── ROLE 1: STUDENT VIEW ─────────────────────────────────────
print("\nGenerating Role 1 — Student View...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: Top risky behaviours for students
top8 = shap_df.head(8)
colors = ['#e74c3c' if v > shap_df['SHAP_GK'].median()
          else '#f39c12' for v in top8['SHAP_GK']]
axes[0].barh(range(len(top8)), top8['SHAP_GK'], color=colors)
axes[0].set_yticks(range(len(top8)))
axes[0].set_yticklabels(top8['Friendly_Name'], fontsize=11)
axes[0].set_xlabel('Risk Contribution Score', fontsize=11)
axes[0].set_title('Your Top Risk Behaviours\n(Higher = More Risky)', fontsize=12, fontweight='bold')
axes[0].invert_yaxis()

high_patch  = mpatches.Patch(color='#e74c3c', label='High Risk Behaviour')
med_patch   = mpatches.Patch(color='#f39c12', label='Medium Risk Behaviour')
axes[0].legend(handles=[high_patch, med_patch], fontsize=9)

# Right: Risk level distribution with friendly labels
risk_counts = y.value_counts().sort_index()
bar_colors  = [RISK_COLORS[i] for i in risk_counts.index]
bars = axes[1].bar([RISK_LABELS[i] for i in risk_counts.index],
                   risk_counts.values, color=bar_colors, edgecolor='white', linewidth=1.5)
for bar, count in zip(bars, risk_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f'{count}\n({count/len(y)*100:.1f}%)',
                 ha='center', fontsize=10, fontweight='bold')
axes[1].set_ylabel('Number of Students', fontsize=11)
axes[1].set_title('KNUST Student Risk Distribution\n(Your Peers)', fontsize=12, fontweight='bold')
axes[1].set_ylim(0, max(risk_counts.values) * 1.2)

plt.suptitle('ROLE 1 — Student Cybersecurity Risk Profile\nGenerated by GK-XGBoost + SHAP',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('shap_role1_students.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: shap_role1_students.png")

# ── ROLE 2: EDUCATOR VIEW ─────────────────────────────────────
print("\nGenerating Role 2 — Educator View...")

raw_original['RiskLevel']     = y.values
raw_original['Predicted_Risk'] = y_pred

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: Risk by department
dept_risk = raw_original.groupby('Department')['RiskLevel'].apply(
    lambda x: (x >= 2).sum() / len(x) * 100).sort_values(ascending=False)
top_depts = dept_risk.head(7)
axes[0].barh(range(len(top_depts)), top_depts.values,
             color='#e74c3c', alpha=0.8)
axes[0].set_yticks(range(len(top_depts)))
axes[0].set_yticklabels(top_depts.index, fontsize=10)
axes[0].set_xlabel('% of High-Risk Students', fontsize=11)
axes[0].set_title('High-Risk Students by Department\n(Educator View)', fontsize=12, fontweight='bold')
axes[0].invert_yaxis()
for i, v in enumerate(top_depts.values):
    axes[0].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)

# Right: Risk by level of study
level_risk = raw_original.groupby('Level')['RiskLevel'].value_counts(normalize=True).unstack(fill_value=0)
level_risk.columns = [RISK_LABELS.get(c, c) for c in level_risk.columns]
level_colors = ['#2ecc71','#f39c12','#e74c3c']
level_risk.plot(kind='bar', ax=axes[1], color=level_colors[:len(level_risk.columns)],
                edgecolor='white', linewidth=0.5)
axes[1].set_xlabel('Level of Study', fontsize=11)
axes[1].set_ylabel('Proportion of Students', fontsize=11)
axes[1].set_title('Risk Distribution by Level of Study\n(Educator View)', fontsize=12, fontweight='bold')
axes[1].legend(title='Risk Level', fontsize=9)
axes[1].tick_params(axis='x', rotation=30)

plt.suptitle('ROLE 2 — Educator Cybersecurity Risk Dashboard\nGenerated by GK-XGBoost + SHAP',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('shap_role2_educators.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: shap_role2_educators.png")

# ── ROLE 3: IT ADMINISTRATOR VIEW ─────────────────────────────
print("\nGenerating Role 3 — IT Administrator View...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: GK-XGBoost vs Baseline SHAP comparison
top10 = shap_df.head(10)
yp    = np.arange(len(top10))
axes[0].barh(yp - 0.2, top10['SHAP_Baseline'], 0.4,
             label='Baseline XGBoost', color='steelblue', alpha=0.85)
axes[0].barh(yp + 0.2, top10['SHAP_GK'],       0.4,
             label='GK-XGBoost (Engineered)', color='darkorange', alpha=0.85)
axes[0].set_yticks(yp)
axes[0].set_yticklabels(top10['Friendly_Name'], fontsize=10)
axes[0].set_xlabel('Mean |SHAP Value| (High-Risk Class)', fontsize=11)
axes[0].set_title('Feature Importance: Baseline vs GK-XGBoost\n(IT Admin View)', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].invert_yaxis()

# Right: High-risk student feature heatmap by predicted class
feature_means = pd.DataFrame({
    'Low Risk':    X_sc[y_pred == 0][shap_df.head(8)['Feature'].tolist()].mean(),
    'Medium Risk': X_sc[y_pred == 1][shap_df.head(8)['Feature'].tolist()].mean(),
    'High Risk':   X_sc[y_pred == 2][shap_df.head(8)['Feature'].tolist()].mean(),
}).T
feature_means.columns = [feature_friendly.get(c, c) for c in feature_means.columns]

im = axes[1].imshow(feature_means.values, cmap='RdYlGn_r', aspect='auto')
axes[1].set_xticks(range(len(feature_means.columns)))
axes[1].set_xticklabels(feature_means.columns, rotation=45, ha='right', fontsize=8)
axes[1].set_yticks(range(len(feature_means.index)))
axes[1].set_yticklabels(feature_means.index, fontsize=10)
axes[1].set_title('Mean Feature Values by Risk Class\n(Red = Riskier Behaviour)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=axes[1], label='Scaled Feature Value')

plt.suptitle('ROLE 3 — IT Administrator Risk Intelligence Dashboard\nGenerated by GK-XGBoost + SHAP',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('shap_role3_it_admins.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: shap_role3_it_admins.png")

# ── SAVE ROLE-DIFFERENTIATED SHAP TABLE ──────────────────────
shap_df['Friendly_Name'] = shap_df['Feature'].map(feature_friendly)
shap_df['Rank_GK']       = range(1, len(shap_df)+1)
shap_df['Delta_SHAP']    = shap_df['SHAP_GK'] - shap_df['SHAP_Baseline']
shap_df['Direction']     = shap_df['Delta_SHAP'].apply(
    lambda x: 'GK-XGBoost focuses MORE' if x > 0 else 'GK-XGBoost focuses LESS')
shap_df.to_csv('shap_role_differentiated.csv', index=False)

# ── STUDENT RISK REPORT (individual-level) ───────────────────
student_report = pd.DataFrame({
    'Student_ID':    range(1, len(y)+1),
    'Actual_Risk':   [RISK_LABELS[r] for r in y],
    'Predicted_Risk':[RISK_LABELS[r] for r in y_pred],
    'High_Risk_Prob':raw['Risk_Prob_High'].round(4),
    'Top_Risk_Factor': shap_df.iloc[0]['Friendly_Name'],
})
student_report.to_csv('student_risk_report.csv', index=False)

print("\n" + "="*55)
print("ROLE-DIFFERENTIATED SHAP COMPLETE")
print("="*55)
print("Files saved:")
print("  shap_role1_students.png    — Student view")
print("  shap_role2_educators.png   — Educator view")
print("  shap_role3_it_admins.png   — IT Admin view")
print("  shap_role_differentiated.csv")
print("  student_risk_report.csv")
