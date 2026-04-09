import streamlit as st
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

# ─── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Turnover Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .metric-number { font-size: 2.2rem; font-weight: 700; }
    .metric-label  { font-size: 0.85rem; color: #6c757d; margin-top: 4px; }
    .section-header {
        font-size: 1.3rem; font-weight: 700;
        color: #2C3E50; margin: 1.5rem 0 0.8rem;
        border-left: 4px solid #E74C3C; padding-left: 10px;
    }
    .insight-box {
        background: #EBF5FB; border-left: 4px solid #2980B9;
        border-radius: 6px; padding: 14px 18px; margin: 10px 0;
        font-size: 0.92rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load Data ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("employee-turnover-dataset.xlsx", sheet_name="turnover")
    df.columns = ['turnover', 'gender', 'age', 'self_control', 'anxiety', 'experience']
    df['turnover_label'] = df['turnover'].map({1: 'Turnover', 0: 'Bertahan'})
    df['gender_label']   = df['gender'].map({1: 'Perempuan', 0: 'Laki-laki'})
    bins = [18, 25, 30, 35, 40, 100]
    labels_age = ['18–24', '25–29', '30–34', '35–39', '40+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels_age, right=False)
    return df

@st.cache_data
def train_model(df):
    features = ['gender', 'age', 'self_control', 'anxiety', 'experience']
    X = df[features]; y = df['turnover']
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_sc, y, test_size=0.2, random_state=42)
    lr = LogisticRegression(random_state=42, max_iter=500)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]
    # Statsmodels for p-values
    X_sm = sm.add_constant(df[features])
    logit = sm.Logit(y, X_sm).fit(disp=0)
    return lr, scaler, X_test, y_test, y_pred, y_prob, logit, features

df = load_data()
lr, scaler, X_test, y_test, y_pred, y_prob, logit, features = train_model(df)

PALETTE = {"red": "#E74C3C", "green": "#27AE60", "blue": "#2980B9",
           "orange": "#E67E22", "dark": "#2C3E50", "light": "#ECF0F1"}

# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/analytics.png", width=70)
    st.title("📊 Employee Turnover")
    st.caption("Predictive Data Mining — Analisis Ujian Kuliah")
    st.divider()

    st.subheader("🔎 Filter Data")
    gender_filter = st.multiselect(
        "Gender", options=['Laki-laki', 'Perempuan'],
        default=['Laki-laki', 'Perempuan']
    )
    age_range = st.slider("Rentang Usia", int(df.age.min()), int(df.age.max()), (18, 58))
    turnover_filter = st.multiselect(
        "Status Turnover", options=['Turnover', 'Bertahan'],
        default=['Turnover', 'Bertahan']
    )
    st.divider()
    st.subheader("📑 Navigasi")
    page = st.radio("Halaman", [
        "Overview",
        "Variabel Psikologis",
        "Korelasi & Regresi",
        "Evaluasi Model",
        "Prediksi Individu"
    ])
    st.divider()
    st.caption("Dataset: 1.129 observasi karyawan\nMata kuliah: Pengantar Data Mining")

# ─── Apply filter ─────────────────────────────────────────
dff = df[
    (df['gender_label'].isin(gender_filter)) &
    (df['age'].between(age_range[0], age_range[1])) &
    (df['turnover_label'].isin(turnover_filter))
]

# ═══════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Employee Turnover — Overview Dashboard")
    st.caption(f"Data setelah filter: **{len(dff):,}** dari {len(df):,} karyawan")
    st.divider()

    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    kpis = [
        ("Total Karyawan", f"{len(dff):,}", "#2C3E50"),
        ("Turnover", f"{dff['turnover'].mean()*100:.1f}%", "#E74C3C"),
        ("Rata-rata Usia", f"{dff['age'].mean():.1f} thn", "#2980B9"),
        ("Rata-rata Pengalaman", f"{dff['experience'].mean():.1f} thn", "#27AE60"),
        ("♀ Proporsi Perempuan", f"{dff['gender'].mean()*100:.1f}%", "#E67E22"),
    ]
    for col, (label, val, color) in zip([col1,col2,col3,col4,col5], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    col_a, col_b = st.columns(2)

    # Turnover Distribution
    with col_a:
        st.markdown('<div class="section-header">Distribusi Turnover</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        counts = dff['turnover_label'].value_counts()
        colors = [PALETTE['red'] if x == 'Turnover' else PALETTE['green'] for x in counts.index]
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=1.5, width=0.5)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+4,
                    f"{b.get_height()}\n({b.get_height()/len(dff)*100:.1f}%)",
                    ha='center', fontsize=11, fontweight='bold')
        ax.set_ylim(0, counts.max()*1.25)
        ax.set_ylabel("Jumlah Karyawan"); ax.grid(axis='y', alpha=0.3)
        sns.despine(); st.pyplot(fig, use_container_width=True); plt.close()

    # Gender Distribution
    with col_b:
        st.markdown('<div class="section-header">Distribusi Gender</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        gen_c = dff['gender_label'].value_counts()
        wedges, texts, auts = ax.pie(
            gen_c.values, labels=gen_c.index, autopct='%1.1f%%',
            colors=[PALETTE['blue'], PALETTE['orange']],
            startangle=90, pctdistance=0.78,
            wedgeprops=dict(edgecolor='white', linewidth=2)
        )
        for a in auts: a.set_fontsize(12); a.set_fontweight('bold')
        st.pyplot(fig, use_container_width=True); plt.close()

    col_c, col_d = st.columns(2)

    # Age distribution
    with col_c:
        st.markdown('<div class="section-header">Distribusi Usia per Status</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for label, color in [('Turnover', PALETTE['red']), ('Bertahan', PALETTE['green'])]:
            sub = dff[dff['turnover_label'] == label]['age']
            ax.hist(sub, bins=20, alpha=0.6, color=color, label=label, edgecolor='white')
        ax.set_xlabel("Usia (Tahun)"); ax.set_ylabel("Frekuensi")
        ax.legend(); ax.grid(axis='y', alpha=0.3); sns.despine()
        st.pyplot(fig, use_container_width=True); plt.close()

    # Experience distribution
    with col_d:
        st.markdown('<div class="section-header">Distribusi Pengalaman Kerja</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for label, color in [('Turnover', PALETTE['red']), ('Bertahan', PALETTE['green'])]:
            sub = dff[dff['turnover_label'] == label]['experience']
            ax.hist(sub, bins=25, alpha=0.6, color=color, label=label, edgecolor='white')
        ax.set_xlabel("Pengalaman (Tahun)"); ax.set_ylabel("Frekuensi")
        ax.legend(); ax.grid(axis='y', alpha=0.3); sns.despine()
        st.pyplot(fig, use_container_width=True); plt.close()

    # Insight box
    st.markdown("""
    <div class="insight-box">
    💡 <b>Insight:</b> Dataset ini sangat seimbang dengan turnover rate 50.6% vs 49.4% yang bertahan.
    Mayoritas responden adalah perempuan (75.6%), dengan rata-rata usia 31 tahun dan pengalaman kerja ~28 tahun.
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE: PSYCHOLOGICAL VARIABLES
# ═══════════════════════════════════════════════════════════
elif page == "Variabel Psikologis":
    st.title("Analisis Variabel Psikologis")
    st.caption("Self-Control & Anxiety sebagai prediktor turnover")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Self-Control per Status Turnover</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        data_sc = [dff[dff['turnover']==0]['self_control'], dff[dff['turnover']==1]['self_control']]
        bp = ax.boxplot(data_sc, patch_artist=True, widths=0.5,
                        medianprops=dict(color='white', linewidth=2.5))
        for patch, color in zip(bp['boxes'], [PALETTE['green'], PALETTE['red']]):
            patch.set_facecolor(color); patch.set_alpha(0.8)
        ax.set_xticks([1,2]); ax.set_xticklabels(['Bertahan', 'Turnover'], fontsize=11)
        ax.set_ylabel("Skor Self-Control (0–10)"); ax.grid(axis='y', alpha=0.3); sns.despine()
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown('<div class="section-header">Anxiety per Status Turnover</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        data_ax = [dff[dff['turnover']==0]['anxiety'], dff[dff['turnover']==1]['anxiety']]
        bp2 = ax.boxplot(data_ax, patch_artist=True, widths=0.5,
                         medianprops=dict(color='white', linewidth=2.5))
        for patch, color in zip(bp2['boxes'], [PALETTE['green'], PALETTE['red']]):
            patch.set_facecolor(color); patch.set_alpha(0.8)
        ax.set_xticks([1,2]); ax.set_xticklabels(['Bertahan', 'Turnover'], fontsize=11)
        ax.set_ylabel("Skor Anxiety (0–10)"); ax.grid(axis='y', alpha=0.3); sns.despine()
        st.pyplot(fig, use_container_width=True); plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Scatter: Self-Control vs Anxiety</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, color, marker in [('Bertahan', PALETTE['green'], 'o'), ('Turnover', PALETTE['red'], 'X')]:
            sub = dff[dff['turnover_label']==label]
            ax.scatter(sub['self_control'], sub['anxiety'], c=color, alpha=0.25,
                       marker=marker, s=20, label=label)
        ax.set_xlabel("Self-Control Score"); ax.set_ylabel("Anxiety Score")
        ax.legend(); ax.grid(alpha=0.3); sns.despine()
        st.pyplot(fig, use_container_width=True); plt.close()

    with col4:
        st.markdown('<div class="section-header">Perbandingan Rata-rata Skor</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        means_s = dff[dff['turnover']==0][['self_control','anxiety']].mean()
        means_t = dff[dff['turnover']==1][['self_control','anxiety']].mean()
        x = np.arange(2); w = 0.35
        b1 = ax.bar(x-w/2, [means_s['self_control'], means_s['anxiety']], w,
                    label='Bertahan', color=PALETTE['green'], edgecolor='white')
        b2 = ax.bar(x+w/2, [means_t['self_control'], means_t['anxiety']], w,
                    label='Turnover', color=PALETTE['red'], edgecolor='white')
        for b in list(b1)+list(b2):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
                    f"{b.get_height():.2f}", ha='center', fontsize=10, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(['Self-Control', 'Anxiety'], fontsize=11)
        ax.set_ylim(0, 8); ax.legend(); ax.grid(axis='y', alpha=0.3); sns.despine()
        st.pyplot(fig, use_container_width=True); plt.close()

    # Stats table
    st.markdown('<div class="section-header">Statistik Deskriptif per Variabel</div>', unsafe_allow_html=True)
    stats_df = dff.groupby('turnover_label')[['self_control','anxiety','age','experience']].agg(['mean','std','median']).round(2)
    stats_df.columns = [' — '.join(c) for c in stats_df.columns]
    st.dataframe(stats_df, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE: CORRELATION & REGRESSION
# ═══════════════════════════════════════════════════════════
elif page == "Korelasi & Regresi":
    st.title("Korelasi & Logistic Regression")
    st.divider()

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        corr = dff[['turnover','gender','age','self_control','anxiety','experience']].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        labels = ['Turnover','Gender','Age','Self\nCtrl','Anxiety','Experience']
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, vmin=-1, vmax=1, ax=ax,
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, annot_kws={"size": 10})
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown('<div class="section-header">Koefisien Regresi Logistik</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        coef_df = pd.DataFrame({'Feature': features, 'Coefficient': lr.coef_[0]}).sort_values('Coefficient')
        colors_c = [PALETTE['red'] if c > 0 else PALETTE['green'] for c in coef_df['Coefficient']]
        bars = ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_c, edgecolor='white')
        ax.axvline(0, color=PALETTE['dark'], linewidth=1.5, linestyle='--')
        for b, v in zip(bars, coef_df['Coefficient']):
            ax.text(v+(0.005 if v>=0 else -0.005), b.get_y()+b.get_height()/2,
                    f"{v:.3f}", va='center', ha='left' if v>=0 else 'right', fontsize=10)
        feat_lbl = {'gender':'Gender','age':'Usia','self_control':'Self-Control',
                    'anxiety':'Anxiety','experience':'Pengalaman'}
        ax.set_yticklabels([feat_lbl[f] for f in coef_df['Feature']])
        ax.set_xlabel("Nilai Koefisien (Standardized)")
        pos_p = mpatches.Patch(color=PALETTE['red'], label='↑ Naikkan risiko turnover')
        neg_p = mpatches.Patch(color=PALETTE['green'], label='↓ Turunkan risiko turnover')
        ax.legend(handles=[pos_p, neg_p], fontsize=8); ax.grid(axis='x', alpha=0.3); sns.despine()
        st.pyplot(fig, use_container_width=True); plt.close()

    # Logit p-value table
    st.markdown('<div class="section-header">Hasil Uji Statistik (Statsmodels Logit)</div>', unsafe_allow_html=True)
    summary = logit.summary2().tables[1].reset_index()
    summary.columns = ['Variabel', 'Koefisien', 'Std Error', 'z', 'p-value', 'CI Lower', 'CI Upper']
    summary = summary[summary['Variabel'] != 'const']
    summary['Signifikan'] = summary['p-value'].apply(lambda p: '✅ Ya (p<0.05)' if p < 0.05 else ('⚠️ Marginal' if p < 0.10 else '❌ Tidak'))
    summary = summary.round(4)
    st.dataframe(summary[['Variabel','Koefisien','Std Error','z','p-value','Signifikan']], use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Temuan:</b> <b>Anxiety</b> adalah satu-satunya variabel yang signifikan secara statistik (p = 0.037).
    Usia bersifat marginal (p = 0.090). Variabel lain tidak signifikan pada α = 5%, mengindikasikan
    kemungkinan perlunya variabel tambahan (kepuasan kerja, kompensasi, dll).
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE: MODEL EVALUATION
# ═══════════════════════════════════════════════════════════
elif page == "Evaluasi Model":
    st.title("Evaluasi Model — Logistic Regression")
    st.divider()

    # Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:,1])

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, label, val, color in zip(
        [c1,c2,c3,c4,c5],
        ["Accuracy","Precision","Recall","F1-Score","AUC-ROC"],
        [acc, prec, rec, f1, auc],
        [PALETTE['blue'], PALETTE['green'], PALETTE['orange'], PALETTE['dark'], PALETTE['red']]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number" style="color:{color}">{val:.3f}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Bertahan (0)', 'Turnover (1)'],
                    yticklabels=['Bertahan (0)', 'Turnover (1)'],
                    annot_kws={"size": 16, "weight": "bold"})
        ax.set_ylabel("Aktual", fontsize=11); ax.set_xlabel("Prediksi", fontsize=11)
        st.pyplot(fig, use_container_width=True); plt.close()

    with col_b:
        st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fpr, tpr, _ = roc_curve(y_test, lr.predict_proba(X_test)[:,1])
        ax.plot(fpr, tpr, color=PALETTE['red'], lw=2.5, label=f'ROC (AUC = {auc:.3f})')
        ax.plot([0,1],[0,1],'--', color='grey', lw=1.5, label='Random')
        ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE['red'])
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=11); ax.grid(alpha=0.3); sns.despine()
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
    report = classification_report(y_test, y_pred, target_names=['Bertahan (0)', 'Turnover (1)'], output_dict=True)
    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(report_df, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Interpretasi:</b> AUC = 0.528 mendekati 0.5 (random classifier), mengindikasikan model
    dengan variabel yang tersedia memiliki kemampuan diskriminasi yang terbatas. Ini adalah temuan
    yang valid secara akademis — menunjukkan bahwa turnover tidak bisa diprediksi hanya dari
    faktor psikologis & demografis dasar ini. Diperlukan variabel tambahan seperti kepuasan kerja,
    gaji, atau budaya organisasi.
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE: PREDICTION
# ═══════════════════════════════════════════════════════════
elif page == "Prediksi Individu":
    st.title("Prediksi Risiko Turnover Individu")
    st.caption("Masukkan data karyawan untuk memprediksi kemungkinan turnover")
    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Input Data Karyawan")
        gender_in  = st.selectbox("Gender", ["Laki-laki", "Perempuan"])
        age_in     = st.slider("Usia", 18, 60, 30)
        sc_in      = st.slider("Skor Self-Control (0–10)", 0.0, 10.0, 5.0, step=0.1)
        anx_in     = st.slider("Skor Anxiety (0–10)", 0.0, 10.0, 5.0, step=0.1)
        exp_in     = st.slider("Pengalaman Kerja (tahun)", 0.0, 50.0, 10.0, step=0.5)

        predict_btn = st.button("Prediksi Sekarang", type="primary", use_container_width=True)

    with col2:
        st.subheader("📊 Hasil Prediksi")
        if predict_btn:
            gender_val = 1 if gender_in == "Perempuan" else 0
            input_data = np.array([[gender_val, age_in, sc_in, anx_in, exp_in]])
            input_scaled = scaler.transform(input_data)
            prob = lr.predict_proba(input_scaled)[0]
            pred = lr.predict(input_scaled)[0]

            prob_turnover = prob[1] * 100
            prob_stay     = prob[0] * 100

            if pred == 1:
                st.error(f"⚠️ **RISIKO TURNOVER TERDETEKSI**")
                risk_color = "#E74C3C"
                risk_label = "Tinggi"
            else:
                st.success(f"✅ **KEMUNGKINAN BERTAHAN**")
                risk_color = "#27AE60"
                risk_label = "Rendah"

            # Gauge-like display
            st.markdown(f"""
            <div style="background:white; border-radius:12px; padding:20px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.1);">
                <div style="font-size:3rem; font-weight:800; color:{risk_color}">{prob_turnover:.1f}%</div>
                <div style="font-size:1rem; color:#6c757d">Probabilitas Turnover</div>
                <div style="margin-top:8px; font-size:0.9rem; background:{risk_color}22; color:{risk_color};
                            padding:6px 14px; border-radius:20px; display:inline-block; font-weight:600">
                    Risiko: {risk_label}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            fig, ax = plt.subplots(figsize=(5, 2.5))
            bars = ax.barh(['Bertahan', 'Turnover'], [prob_stay, prob_turnover],
                           color=[PALETTE['green'], PALETTE['red']], edgecolor='white')
            for b, v in zip(bars, [prob_stay, prob_turnover]):
                ax.text(v+0.5, b.get_y()+b.get_height()/2, f"{v:.1f}%",
                        va='center', fontsize=12, fontweight='bold')
            ax.set_xlim(0, 120); ax.set_xlabel("Probabilitas (%)"); ax.grid(axis='x', alpha=0.3)
            sns.despine(); st.pyplot(fig, use_container_width=True); plt.close()
        else:
            st.info("👈 Isi form di kiri lalu klik **Prediksi Sekarang**")
            st.markdown("""
            <div class="insight-box">
            ℹ️ Model menggunakan <b>Logistic Regression</b> yang dilatih pada 80% data (903 sampel)
            dan divalidasi pada 20% data (226 sampel). Model ini bersifat indikatif dan tidak
            dimaksudkan untuk keputusan manajerial.
            </div>""", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────
st.divider()
st.caption("📚 Portofolio Akademik | Pengantar Data Mining | Dataset UAS")
