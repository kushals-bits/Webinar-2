# Webinar 2: Data Preprocessing Masterclass

A production-grade, modular, and educational repository implementing end-to-end data preprocessing pipelines across **Tabular**, **Text (NLP)**, and **Image (CV)** modalities using **real-world practical datasets** and **modern ML/NLP algorithms** (different from Webinar 1), with comprehensive evaluation benchmarks comparing model performance **Before vs. After** preprocessing.

---

## 📐 Architecture & Workflow

This repository directly implements the three-column preprocessing architecture:

```
                      +-----------------------------+
                      |          WEBINAR 2          |
                      |      DATA PREPROCESSING     |
                      +--------------+--------------+
                                     |
         +---------------------------+---------------------------+
         |                           |                           |
         v                           v                           v
+-----------------+         +-----------------+         +-----------------+
|     TABULAR     |         |      TEXT       |         |      IMAGE      |
+-----------------+         +-----------------+         +-----------------+
| • Profiling     |         | • Cleaning      |         | • Loading       |
| • Encoding      |         | • TF-IDF        |         | • Resize        |
| • Scaling       |         | • Numerical Feat|         | • Normalize     |
| • Imbalance     |         | • Imbalance     |         | • PCA           |
| • Train (XGBoost|         +--------+--------+         | • Reduced Feat  |
+--------+--------+                  |                  +--------+--------+
         |                           +-----------+               |
         |                                       |               |
         +---------------------------+-----------+---------------+
                                     |
                                     v
                        +-------------------------+
                        |       Evaluation        |
                        +------------+------------+
                                     |
                                     v
                        +-------------------------+
                        |  Compare Before vs After|
                        +-------------------------+
```

---

## 🔬 Real Practical Datasets & Modern Algorithms

| Modality | Real Practical Dataset | Baseline Algorithm | Preprocessed Algorithm (Webinar 2) |
| :--- | :--- | :--- | :--- |
| **Tabular** | **IBM Telco Customer Churn** (7,043 real customer accounts) | Logistic Regression (naive, unscaled) | **XGBoost Classifier** (Gradient Boosted Trees) + SMOTE |
| **Text** | **20 Newsgroups** (1,780 real internet forum posts: `sci.med` vs `alt.atheism`) | SGD Classifier (raw Bag-of-Words) | **LinearSVC (Support Vector Machine)** + TF-IDF (1,2) + Linguistic Features |
| **Image** | **Handwritten Digits** (537 real human handwriting images: Digits 0, 1, 2) | Logistic Regression (2,352 raw pixels) | **SVM (RBF Kernel)** & **KNN (k=5)** on 21 PCA dimensions |

---

## 🚀 Key Highlights & Modules

### 1. 📊 Tabular Preprocessing (`src/tabular/`)
* **Profiling (`profiling.py`)**: Data health diagnostics, missing value rates (whitespace detection in `TotalCharges`), skewness/kurtosis analysis, and dual outlier detection via **IQR** and **Z-score**.
* **Encoding (`encoding.py`)**: One-Hot Encoding for nominal columns (`PaymentMethod`, `InternetService`, `TechSupport`) and Ordinal Encoding for ranked contracts (`Month-to-month < One year < Two year`).
* **Scaling (`scaling.py`)**: Comparative analysis between **StandardScaler**, **MinMaxScaler**, and **RobustScaler** (median/IQR based, immune to extreme outlier distortion).
* **Imbalance (`imbalance.py`)**: **SMOTE** (Synthetic Minority Over-sampling Technique) to rebalance the ~27% churn minority class.
* **Model Training (`model_trainer.py`)**: Training **XGBoost Classifier** with early stopping / tree regularization on preprocessed features.

### 2. 💬 Text (NLP) Preprocessing (`src/text/`)
* **Cleaning (`cleaning.py`)**: Multi-stage regex cleaning stripping headers, email fragments, quotation markers, contractions (`won't` $\to$ `will not`), punctuation, lowercasing, and stopwords.
* **TF-IDF Vectorization (`vectorization.py`)**: Unigram + bigram extraction with sublinear term frequency scaling ($1 + \log(\text{tf})$) and document frequency cutoffs.
* **Numerical Feature Engineering (`feature_engineering.py`)**: Extracting statistical and linguistic metadata: character length, word count, uppercase shouting ratio, punctuation density, and lexicon sentiment polarity.
* **Imbalance (`imbalance.py`)**: Training **LinearSVC (SVM)** with Platt scaling probability calibration.

### 3. 🖼️ Image (Computer Vision) Preprocessing (`src/image/`)
* **Loading (`loading.py`)**: Class directory scanning and RGB image tensor conversion.
* **Resizing (`resizing.py`)**: Aspect-ratio preserving letterboxing (padding onto centered canvas) vs direct scaling.
* **Normalization (`normalization.py`)**: Pixel intensity normalization $[0, 255] \to [0.0, 1.0]$ and channel-wise Z-score standardization.
* **PCA Dimensionality Reduction (`pca_reduction.py`)**: Dimensionality reduction from 2,352 raw pixels down to **21 principal components** retaining 95% cumulative variance — achieving **99.1% feature compression**!
* **Modeling**: Training **SVM with RBF kernel** and **KNN (k=5)** on compact PCA features.

---

## 📊 Benchmark Summary (Before vs After Preprocessing)

| Modality | Key Metric | Baseline (Before) | Preprocessed (After) | Delta ($\Delta$) | Impact / Takeaway |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Tabular** (IBM Telco Churn) | Churner Recall | 52.89% | **61.46%** | **+8.57%** | SMOTE & XGBoost boost minority churn detection by **+16.2%** |
| **Tabular** (IBM Telco Churn) | F1-Score (Minority) | 0.5888 | **0.5998** | **+0.0110** | Balanced identification of at-risk customers |
| **Text** (20 Newsgroups) | Accuracy | 90.34% | **95.96%** | **+5.62%** | Cleaned TF-IDF + LinearSVC boosts accuracy to **~96%** |
| **Text** (20 Newsgroups) | ROC-AUC | 0.9037 | **0.9877** | **+0.0840** | **+9.3% relative boost** in discriminative capability |
| **Image** (Handwritten Digits) | Feature Dimensions | 2,352 dims | **21 dims** | **-2,331 dims** | **99.1% Dimensionality Reduction** via PCA with **99.26% accuracy** |

---

## 📁 Repository Structure

```
Webinar 2/
├── README.md                          # Repository documentation & guide
├── requirements.txt                   # Project dependencies (includes xgboost, lightgbm)
├── .gitignore                         # Git exclusion rules
├── main.py                            # Master script executing all pipelines & generating reports
├── input_image.jpeg                   # Workflow architecture reference diagram
│
├── data/
│   ├── tabular/
│   │   └── telco_churn_raw.csv        # Real IBM Telco Customer Churn (7,043 rows)
│   ├── text/
│   │   ├── newsgroups_raw.csv         # Real 20 Newsgroups posts (1,780 documents)
│   │   └── newsgroups_clean.csv       # Cleaned text corpus
│   └── image/
│       ├── raw/                       # Real handwritten digit images (classes 0, 1, 2)
│       └── processed/                 # PCA-reduced feature matrices
│
├── notebooks/
│   ├── 01_tabular_preprocessing.ipynb # Interactive Tabular profiling, encoding, scaling, SMOTE & XGBoost
│   ├── 02_text_preprocessing.ipynb    # Interactive Text cleaning, TF-IDF, features & LinearSVC
│   ├── 03_image_preprocessing.ipynb   # Interactive Image loading, resizing, normalization, PCA & SVM RBF
│   └── 04_end_to_end_webinar2.ipynb   # Master end-to-end demo comparing Before vs After
│
├── src/
│   ├── tabular/                       # Profiling, Encoding, Scaling, SMOTE, Model Trainer (XGBoost)
│   ├── text/                          # Cleaning, TF-IDF, Numerical Features, Imbalance, LinearSVC
│   ├── image/                         # Loading, Resizing, Normalization, PCA Reduction, SVM RBF + KNN
│   ├── evaluation/                    # Metrics, Comparison, Visualizer
│   └── data_generators/               # Real Dataset Downloader & Notebook Generators
│
└── reports/
    ├── master_before_vs_after_metrics.csv # Exported comparison metrics
    ├── before_vs_after_benchmarks.png     # Grouped bar charts across all modalities
    ├── tabular_scaling_and_outliers.png   # Scaler distribution comparison
    ├── image_pca_and_reconstruction.png   # Scree plot & image reconstruction
    └── confusion_matrices_comparison.png  # Confusion matrices before vs after
```

---

## ⚡ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/kushals-bits/Webinar-2.git
cd "Webinar 2"

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline

Execute the master CLI pipeline:

```bash
python main.py
```

### 3. Launch Interactive Notebooks

```bash
jupyter notebook
```

Navigate to `notebooks/` and open:
* `01_tabular_preprocessing.ipynb` — Tabular Preprocessing & XGBoost
* `02_text_preprocessing.ipynb` — NLP Cleaning & LinearSVC (SVM)
* `03_image_preprocessing.ipynb` — Computer Vision, PCA & SVM RBF
* `04_end_to_end_webinar2.ipynb` — Unified Master Demonstration
