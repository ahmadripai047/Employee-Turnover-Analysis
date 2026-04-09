# 📊 Employee Turnover Analysis — Pengantar Data Mining (PDM)

## 📋 Deskripsi Dataset

Dataset ini merupakan data ujian akhir semester (UAS) mata kuliah **Pengatar Data Mining (PDM)** yang menganalisis faktor-faktor psikologis dan demografis yang mempengaruhi **keputusan karyawan untuk keluar (turnover)** dari perusahaan.

Dataset terdiri dari **1.129 observasi karyawan** dengan 6 variabel utama:

| Variabel | Kode | Deskripsi | Tipe |
|---|---|---|---|
| Employee Turnover | Y | Status turnover: `1` = Keluar, `0` = Bertahan | Binary |
| Gender | X1 | Jenis kelamin: `0` = Laki-laki, `1` = Perempuan | Kategorikal |
| Age | X2 | Usia karyawan (tahun) | Numerik |
| Self-Control | X3 | Skor self-control (skala 0–10) | Numerik |
| Anxiety | X4 | Skor kecemasan (skala 0–10) | Numerik |
| Experience Time | X5 | Lama pengalaman kerja (tahun) | Numerik |

> **Konteks:** Penelitian ini menggunakan pendekatan **Predictive Data Mining** untuk mengidentifikasi prediktor turnover karyawan dari aspek psikologis (self-control & anxiety) serta demografis (gender, age, experience).

---

## 📁 Struktur Repository

```
├── pdm_dataset.xlsx          # Dataset original
├── analysis.py               # Script analisis (Python)
├── README.md                 
├── figures/
│   ├── fig1_overview.png         # Overview & distribusi data
│   ├── fig2_psychology.png       # Analisis variabel psikologis
│   ├── fig3_correlation_lr.png   # Korelasi & koefisien regresi logistik
│   ├── fig4_model_eval.png       # Evaluasi model (ROC, Confusion Matrix)
│   └── fig5_demographics.png     # Analisis demografis
```

---

## 📊 Statistik Deskriptif

| Variabel | Mean | Std Dev | Min | Max |
|---|---|---|---|---|
| Turnover Rate | 50.6% | — | — | — |
| Age | 31.1 | 7.0 | 18 | 58 |
| Self-Control | 5.44 | 2.09 | 0 | 9.5 |
| Anxiety | 5.59 | 1.74 | 0 | 9.4 |
| Experience | 28.0 | 25.9 | 0 | 99.8 |

- **Gender:** 75.6% Perempuan, 24.4% Laki-laki
- **Turnover:** 571 karyawan keluar (50.6%), 558 bertahan (49.4%)

---

## 🔍 Analisis yang Dilakukan

### 1. Exploratory Data Analysis (EDA)
- Distribusi variabel target (turnover)
- Distribusi demografis (gender, usia, pengalaman kerja)
- Visualisasi missing values dan outlier

### 2. Analisis Variabel Psikologis
- Perbandingan skor self-control dan anxiety antara kelompok turnover vs bertahan
- Boxplot dan mean comparison
- Scatter plot hubungan self-control vs anxiety

### 3. Analisis Korelasi
- Correlation matrix (Pearson) antar seluruh variabel
- Identifikasi multikolinearitas

### 4. Regresi Logistik (Logistic Regression)
- Model prediksi binary (turnover: ya/tidak)
- Koefisien standardized untuk interpretasi feature importance
- Uji signifikansi (p-value) menggunakan statsmodels

### 5. Evaluasi Model
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- ROC Curve & AUC Score

### 6. Analisis Demografis
- Turnover rate berdasarkan gender
- Turnover rate berdasarkan kelompok usia

---

## 📈 Temuan Utama

### Hasil Regresi Logistik (Statsmodels)

| Variabel | Koefisien | p-value | Signifikan |
|---|---|---|---|
| Self-Control | -0.044 | 0.126 | ❌ Tidak |
| **Anxiety** | **-0.076** | **0.037** | **✅ Ya (p < 0.05)** |
| Age | -0.015 | 0.090 | ⚠️ Marginal |
| Experience | -0.002 | 0.288 | ❌ Tidak |
| Gender | -0.027 | 0.854 | ❌ Tidak |

**Interpretasi:**
- **Anxiety** adalah satu-satunya prediktor yang signifikan secara statistik (p = 0.037). Menariknya, koefisien negatif menunjukkan karyawan dengan anxiety lebih tinggi justru cenderung *bertahan* — temuan ini counterintuitive dan layak dieksplorasi lebih lanjut dalam konteks spesifik dataset ini.
- Self-control dan faktor demografis tidak menunjukkan pengaruh signifikan pada model ini.
- Model keseluruhan memiliki AUC = **0.528**, mengindikasikan bahwa variabel yang tersedia memiliki kemampuan prediksi terbatas dan mungkin diperlukan variabel tambahan (seperti kepuasan kerja, kompensasi, dll).

### Performa Model

| Metrik | Nilai |
|---|---|
| Accuracy | 50% |
| AUC-ROC | 0.528 |
| Precision (macro avg) | 0.50 |
| Recall (macro avg) | 0.50 |

---

## 🛠️ Tools & Library

```python
pandas==2.x          # Data manipulation
numpy==1.x           # Numerical computation
matplotlib==3.x      # Visualization
seaborn==0.x         # Statistical visualization
scikit-learn==1.x    # Machine learning (Logistic Regression, metrics)
statsmodels==0.x     # Statistical modeling (p-values, logit)
```

---

## 🚀 Cara Menjalankan

```bash
# Clone repository
git clone https://github.com/username/employee-turnover-pdm.git
cd employee-turnover-pdm

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels openpyxl

# Jalankan analisis
python analysis.py
```

---

## 🎓 Konteks Akademik

Proyek ini merupakan bagian dari tugas ujian mata kuliah **Predictive Data Mining** pada jenjang perguruan tinggi. Analisis berfokus pada penerapan teknik regresi logistik untuk klasifikasi biner dalam konteks manajemen sumber daya manusia.

---

## 📌 Catatan

Dataset ini bersifat akademik dan digunakan semata untuk keperluan pembelajaran. Temuan tidak dapat digeneralisasi tanpa validasi lebih lanjut pada populasi yang lebih beragam.
