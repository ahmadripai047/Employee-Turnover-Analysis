import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ─── Style ───────────────────────────────────────────────────────────────────
PALETTE = {"main": "#2C3E50", "accent": "#E74C3C", "green": "#27AE60",
           "blue": "#2980B9", "orange": "#E67E22", "light": "#ECF0F1"}
sns.set_theme(style="whitegrid", font="Arial")
plt.rcParams.update({"figure.dpi": 150, "font.family": "Arial"})

# ─── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_excel('/mnt/user-data/uploads/pdm_dataset.xlsx', sheet_name='turnover')
df.columns = ['turnover', 'gender', 'age', 'self_control', 'anxiety', 'experience']
df['turnover_label'] = df['turnover'].map({1: 'Turnover (Yes)', 0: 'Stay (No)'})
df['gender_label']   = df['gender'].map({1: 'Female', 0: 'Male'})

# ═══════════════════════════════════════════════════════════════════
# FIGURE 1 — Overview Dashboard (4 panels)
# ═══════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle("Employee Turnover Dataset – Overview Dashboard",
              fontsize=16, fontweight='bold', color=PALETTE['main'], y=1.01)

# 1a Turnover distribution
ax = axes[0, 0]
counts = df['turnover_label'].value_counts()
bars = ax.bar(counts.index, counts.values,
              color=[PALETTE['accent'], PALETTE['green']], edgecolor='white', linewidth=1.5)
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 6,
            f"{b.get_height()}\n({b.get_height()/len(df)*100:.1f}%)",
            ha='center', fontsize=11, fontweight='bold', color=PALETTE['main'])
ax.set_title("Distribution of Employee Turnover", fontweight='bold', fontsize=12)
ax.set_ylabel("Number of Employees", fontsize=10)
ax.set_ylim(0, 680)
ax.tick_params(axis='x', labelsize=11)

# 1b Gender distribution
ax = axes[0, 1]
gen_counts = df['gender_label'].value_counts()
colors_g = [PALETTE['blue'], PALETTE['orange']]
wedges, texts, autotexts = ax.pie(gen_counts.values, labels=gen_counts.index,
                                   autopct='%1.1f%%', colors=colors_g,
                                   startangle=90, pctdistance=0.75,
                                   wedgeprops=dict(edgecolor='white', linewidth=2))
for t in autotexts: t.set_fontsize(12); t.set_fontweight('bold')
ax.set_title("Gender Distribution", fontweight='bold', fontsize=12)

# 1c Age distribution by turnover
ax = axes[1, 0]
for label, color in [('Turnover (Yes)', PALETTE['accent']), ('Stay (No)', PALETTE['green'])]:
    subset = df[df['turnover_label'] == label]['age']
    ax.hist(subset, bins=20, alpha=0.65, color=color, label=label, edgecolor='white')
ax.axvline(df[df['turnover']==1]['age'].mean(), color=PALETTE['accent'], linestyle='--', linewidth=1.5)
ax.axvline(df[df['turnover']==0]['age'].mean(), color=PALETTE['green'], linestyle='--', linewidth=1.5)
ax.set_title("Age Distribution by Turnover Status", fontweight='bold', fontsize=12)
ax.set_xlabel("Age (Years)", fontsize=10); ax.set_ylabel("Frequency", fontsize=10)
ax.legend(fontsize=10)

# 1d Experience distribution by turnover
ax = axes[1, 1]
for label, color in [('Turnover (Yes)', PALETTE['accent']), ('Stay (No)', PALETTE['green'])]:
    subset = df[df['turnover_label'] == label]['experience']
    ax.hist(subset, bins=25, alpha=0.65, color=color, label=label, edgecolor='white')
ax.set_title("Work Experience Distribution by Turnover", fontweight='bold', fontsize=12)
ax.set_xlabel("Experience (Years)", fontsize=10); ax.set_ylabel("Frequency", fontsize=10)
ax.legend(fontsize=10)

plt.tight_layout()
fig1.savefig('/home/claude/fig1_overview.png', bbox_inches='tight', facecolor='white')
plt.close()
print("fig1 done")

# ═══════════════════════════════════════════════════════════════════
# FIGURE 2 — Psychological Variables Analysis
# ═══════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle("Psychological Variables Analysis (Self-Control & Anxiety)",
              fontsize=16, fontweight='bold', color=PALETTE['main'], y=1.01)

# 2a Boxplot self-control vs turnover
ax = axes[0, 0]
data_sc = [df[df['turnover']==0]['self_control'], df[df['turnover']==1]['self_control']]
bp = ax.boxplot(data_sc, patch_artist=True, widths=0.5,
                medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], [PALETTE['green'], PALETTE['accent']]):
    patch.set_facecolor(color); patch.set_alpha(0.8)
ax.set_xticks([1,2]); ax.set_xticklabels(['Stay (No)', 'Turnover (Yes)'], fontsize=11)
ax.set_title("Self-Control Score by Turnover Status", fontweight='bold', fontsize=12)
ax.set_ylabel("Self-Control Score (0–10)", fontsize=10)

# 2b Boxplot anxiety vs turnover
ax = axes[0, 1]
data_ax = [df[df['turnover']==0]['anxiety'], df[df['turnover']==1]['anxiety']]
bp2 = ax.boxplot(data_ax, patch_artist=True, widths=0.5,
                 medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp2['boxes'], [PALETTE['green'], PALETTE['accent']]):
    patch.set_facecolor(color); patch.set_alpha(0.8)
ax.set_xticks([1,2]); ax.set_xticklabels(['Stay (No)', 'Turnover (Yes)'], fontsize=11)
ax.set_title("Anxiety Score by Turnover Status", fontweight='bold', fontsize=12)
ax.set_ylabel("Anxiety Score (0–10)", fontsize=10)

# 2c Scatter self-control vs anxiety
ax = axes[1, 0]
for label, color, marker in [('Stay (No)', PALETTE['green'], 'o'), ('Turnover (Yes)', PALETTE['accent'], 'X')]:
    sub = df[df['turnover_label']==label]
    ax.scatter(sub['self_control'], sub['anxiety'], c=color, alpha=0.3,
               marker=marker, s=25, label=label)
ax.set_xlabel("Self-Control Score", fontsize=10)
ax.set_ylabel("Anxiety Score", fontsize=10)
ax.set_title("Self-Control vs Anxiety", fontweight='bold', fontsize=12)
ax.legend(fontsize=10)

# 2d Mean comparison bar chart
ax = axes[1, 1]
means_stay    = df[df['turnover']==0][['self_control','anxiety']].mean()
means_turnover= df[df['turnover']==1][['self_control','anxiety']].mean()
x = np.arange(2)
w = 0.35
b1 = ax.bar(x - w/2, [means_stay['self_control'], means_stay['anxiety']],
            w, label='Stay (No)', color=PALETTE['green'], edgecolor='white')
b2 = ax.bar(x + w/2, [means_turnover['self_control'], means_turnover['anxiety']],
            w, label='Turnover (Yes)', color=PALETTE['accent'], edgecolor='white')
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
            f"{b.get_height():.2f}", ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(['Self-Control', 'Anxiety'], fontsize=11)
ax.set_title("Mean Scores: Stay vs Turnover", fontweight='bold', fontsize=12)
ax.set_ylabel("Mean Score", fontsize=10); ax.set_ylim(0, 8)
ax.legend(fontsize=10)

plt.tight_layout()
fig2.savefig('/home/claude/fig2_psychology.png', bbox_inches='tight', facecolor='white')
plt.close()
print("fig2 done")

# ═══════════════════════════════════════════════════════════════════
# FIGURE 3 — Correlation & Feature Importance
# ═══════════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle("Correlation Analysis & Logistic Regression",
              fontsize=16, fontweight='bold', color=PALETTE['main'], y=1.01)

# 3a Heatmap
ax = axes[0]
corr = df[['turnover','gender','age','self_control','anxiety','experience']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, annot_kws={"size": 10},
            xticklabels=['Turnover','Gender','Age','Self\nControl','Anxiety','Experience'],
            yticklabels=['Turnover','Gender','Age','Self\nControl','Anxiety','Experience'])
ax.set_title("Correlation Matrix (Lower Triangle)", fontweight='bold', fontsize=12)

# 3b Logistic Regression coefficients
features = ['gender','age','self_control','anxiety','experience']
X = df[features]; y = df['turnover']
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_sc, y, test_size=0.2, random_state=42)
lr = LogisticRegression(random_state=42, max_iter=500)
lr.fit(X_train, y_train)

coef_df = pd.DataFrame({'Feature': features, 'Coefficient': lr.coef_[0]})
coef_df = coef_df.sort_values('Coefficient')
colors_c = [PALETTE['accent'] if c > 0 else PALETTE['green'] for c in coef_df['Coefficient']]
ax2 = axes[1]
bars = ax2.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_c, edgecolor='white')
ax2.axvline(0, color=PALETTE['main'], linewidth=1.5, linestyle='--')
for b, v in zip(bars, coef_df['Coefficient']):
    ax2.text(v + (0.005 if v >= 0 else -0.005), b.get_y()+b.get_height()/2,
             f"{v:.3f}", va='center', ha='left' if v>=0 else 'right', fontsize=10)
ax2.set_title("Logistic Regression Coefficients\n(Standardized Features)", fontweight='bold', fontsize=12)
ax2.set_xlabel("Coefficient Value", fontsize=10)
feat_labels = {'gender':'Gender','age':'Age','self_control':'Self-Control',
               'anxiety':'Anxiety','experience':'Experience'}
ax2.set_yticklabels([feat_labels[f] for f in coef_df['Feature']], fontsize=10)
pos_p = mpatches.Patch(color=PALETTE['accent'], label='→ Increases Turnover Risk')
neg_p = mpatches.Patch(color=PALETTE['green'], label='→ Reduces Turnover Risk')
ax2.legend(handles=[pos_p, neg_p], fontsize=9)

plt.tight_layout()
fig3.savefig('/home/claude/fig3_correlation_lr.png', bbox_inches='tight', facecolor='white')
plt.close()
print("fig3 done")

# ═══════════════════════════════════════════════════════════════════
# FIGURE 4 — Model Evaluation (ROC + Confusion Matrix)
# ═══════════════════════════════════════════════════════════════════
fig4, axes = plt.subplots(1, 2, figsize=(13, 5))
fig4.suptitle("Logistic Regression Model Evaluation",
              fontsize=16, fontweight='bold', color=PALETTE['main'], y=1.01)

y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)[:,1]

# 4a Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Stay (0)', 'Turnover (1)'],
            yticklabels=['Stay (0)', 'Turnover (1)'],
            annot_kws={"size": 14, "weight": "bold"})
axes[0].set_title("Confusion Matrix", fontweight='bold', fontsize=13)
axes[0].set_ylabel("Actual", fontsize=11); axes[0].set_xlabel("Predicted", fontsize=11)

# 4b ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
axes[1].plot(fpr, tpr, color=PALETTE['accent'], lw=2.5, label=f'ROC Curve (AUC = {auc:.3f})')
axes[1].plot([0,1],[0,1], '--', color='grey', lw=1.5, label='Random Classifier')
axes[1].fill_between(fpr, tpr, alpha=0.1, color=PALETTE['accent'])
axes[1].set_xlabel("False Positive Rate", fontsize=11)
axes[1].set_ylabel("True Positive Rate", fontsize=11)
axes[1].set_title("ROC Curve", fontweight='bold', fontsize=13)
axes[1].legend(fontsize=11)

plt.tight_layout()
fig4.savefig('/home/claude/fig4_model_eval.png', bbox_inches='tight', facecolor='white')
plt.close()
print("fig4 done")

# ═══════════════════════════════════════════════════════════════════
# FIGURE 5 — Gender & Age Group Deep-Dive
# ═══════════════════════════════════════════════════════════════════
fig5, axes = plt.subplots(1, 2, figsize=(13, 5))
fig5.suptitle("Turnover Rate by Demographics",
              fontsize=16, fontweight='bold', color=PALETTE['main'], y=1.01)

# 5a Turnover rate by gender
ax = axes[0]
gender_to = df.groupby('gender_label')['turnover'].mean() * 100
bars = ax.bar(gender_to.index, gender_to.values,
              color=[PALETTE['blue'], PALETTE['orange']], edgecolor='white', linewidth=1.5, width=0.5)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
            f"{b.get_height():.1f}%", ha='center', fontsize=12, fontweight='bold')
ax.set_title("Turnover Rate by Gender", fontweight='bold', fontsize=12)
ax.set_ylabel("Turnover Rate (%)", fontsize=10); ax.set_ylim(0, 70)

# 5b Turnover rate by age group
ax2 = axes[1]
bins = [18, 25, 30, 35, 40, 60]
labels_age = ['18–24','25–29','30–34','35–39','40+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels_age, right=False)
ag_to = df.groupby('age_group', observed=True)['turnover'].mean() * 100
bars2 = ax2.bar(ag_to.index, ag_to.values, color=PALETTE['blue'],
                edgecolor='white', linewidth=1.5, width=0.6)
for b in bars2:
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
             f"{b.get_height():.1f}%", ha='center', fontsize=11, fontweight='bold')
ax2.set_title("Turnover Rate by Age Group", fontweight='bold', fontsize=12)
ax2.set_ylabel("Turnover Rate (%)", fontsize=10); ax2.set_ylim(0, 70)
ax2.set_xlabel("Age Group", fontsize=10)

plt.tight_layout()
fig5.savefig('/home/claude/fig5_demographics.png', bbox_inches='tight', facecolor='white')
plt.close()
print("fig5 done")

# ─── Print stats for README ──────────────────────────────────────
print("\n===== STATS =====")
print(f"n = {len(df)}")
print(f"Turnover rate = {df['turnover'].mean()*100:.1f}%")
print(f"Female % = {df['gender'].mean()*100:.1f}%")
print(f"Age mean = {df['age'].mean():.1f} ± {df['age'].std():.1f}")
print(f"Experience mean = {df['experience'].mean():.1f} ± {df['experience'].std():.1f}")
print(f"Self-control mean = {df['self_control'].mean():.2f} ± {df['self_control'].std():.2f}")
print(f"Anxiety mean = {df['anxiety'].mean():.2f} ± {df['anxiety'].std():.2f}")

report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)
print(f"AUC-ROC: {auc:.4f}")

# Logit model (statsmodels for p-values)
X_sm = sm.add_constant(df[features])
logit_model = sm.Logit(df['turnover'], X_sm).fit(disp=0)
print("\nLogit Summary:")
print(logit_model.summary2().tables[1].to_string())
