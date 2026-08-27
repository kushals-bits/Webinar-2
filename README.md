# Webinar 2: Data Preprocessing Masterclass

A production-grade, modular, and educational repository implementing end-to-end data preprocessing pipelines across **Tabular**, **Text (NLP)**, and **Image (CV)** modalities, with comprehensive evaluation benchmarks comparing model performance **Before vs. After** preprocessing.

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
| • Train Model   |         +--------+--------+         | • Reduced Feat  |
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

## 🚀 Key Highlights & Modules

### 1. 📊 Tabular Preprocessing (`src/tabular/`)
* **Profiling (`profiling.py`)**: Data health diagnostics, missing value rates, skewness/kurtosis analysis, and dual outlier detection via **IQR (Interquartile Range)** and **Z-score**.
* **Encoding (`encoding.py`)**: Cleaning noisy strings, handling unphysical entries, applying **One-Hot Encoding** on nominal categories and **Ordinal Encoding** on ranked categories.
* **Scaling (`scaling.py`)**: Comparative analysis between **StandardScaler**, **MinMaxScaler**, and **RobustScaler** (median/IQR based, immune to extreme outlier distortion).
* **Imbalance Handling (`imbalance.py`)**: **SMOTE** (Synthetic Minority Over-sampling Technique) to rebalance minority classes without duplicate over-reliance.
* **Model Training (`model_trainer.py`)**: Comparing unscaled/imbalanced baseline Logistic Regression against preprocessed Random Forest / Gradient Boosting.

### 2. 💬 Text (NLP) Preprocessing (`src/text/`)
* **Cleaning (`cleaning.py`)**: Multi-stage cleaning stripping HTML tags (`<br/>`, `<b>`), unescaping HTML entities, regex URL/email removal, contraction expansion (`can't` $\to$ `cannot`), punctuation stripping, lowercasing, and stopword filtering.
* **TF-IDF Vectorization (`vectorization.py`)**: Unigram + bigram extraction with sublinear term frequency scaling ($1 + \log(\text{tf})$) and corpus frequency pruning (`min_df`/`max_df`).
* **Numerical Feature Engineering (`feature_engineering.py`)**: Extracting statistical and linguistic metadata: character length, word count, uppercase shouting ratio, punctuation density, and lexicon sentiment polarity.
* **Imbalance (`imbalance.py`)**: SMOTE on combined sparse TF-IDF and dense linguistic features.

### 3. 🖼️ Image (Computer Vision) Preprocessing (`src/image/`)
* **Loading (`loading.py`)**: Recursive class directory scanning, RGB color space validation, and batch array conversions.
* **Resizing (`resizing.py`)**: Aspect-ratio preserving letterboxing (padding onto centered canvas) vs direct scaling.
* **Normalization (`normalization.py`)**: Pixel intensity normalization $[0, 255] \to [0.0, 1.0]$ and channel-wise Z-score standardization.
* **PCA Dimensionality Reduction (`pca_reduction.py`)**: Dimensionality reduction from high-dimensional raw pixel space ($64 \times 64 \times 3 = 12,288$ dims) down to top principal components retaining $95\%$ variance ($\sim 50$ dims), achieving **$>99\%$ feature compression** while preserving classification performance.
* **Reconstruction**: Visualizing inverse PCA reconstruction back to image space.

### 4. 📈 Downstream Evaluation & Before vs After Benchmarking (`src/evaluation/`)
* **Metrics Suite (`metrics.py`)**: Accuracy, Balanced Accuracy, Precision, Recall, F1-Score (Macro & Minority), ROC-AUC, and Confusion Matrices.
* **Comparison Engine (`comparison.py`)**: Calculates metric deltas ($\Delta$) and percentage relative improvements across all modalities.
* **Visualizer (`visualizer.py`)**: Generates high-resolution comparative figures in `reports/`.

---

## 📊 Before vs After Benchmark Summary

| Modality | Key Metric | Baseline (Before) | Preprocessed (After) | Delta ($\Delta$) | Impact / Takeaway |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Tabular** | Recall (Churners) | 18.07% | **56.63%** | **+38.56%** | SMOTE & Robust Scaling boost minority detection by **+213%** |
| **Tabular** | F1-Score (Macro) | 0.5523 | **0.6025** | **+0.0502** | Balanced overall performance across classes |
| **Text** | Cleanliness | Noisy HTML/URLs | **Clean TF-IDF + Feats** | - | Structured features enable robust NLP generalization |
| **Image** | Feature Dimensions | 12,288 dims | **54 dims** | **-12,234 dims** | **99.6% Dimensionality Reduction** via PCA with high retention |

---

## 📁 Repository Structure

```
Webinar 2/
├── README.md                          # Repository documentation & guide
├── requirements.txt                   # Project dependencies
├── .gitignore                         # Git exclusion rules
├── main.py                            # Master script executing all pipelines & generating reports
├── input_image.jpeg                   # Workflow architecture reference diagram
│
├── data/
│   ├── tabular/
│   │   ├── customer_churn_raw.csv     # Raw tabular data with missingness, skew & imbalance
│   │   └── customer_churn_processed.csv # Cleaned & scaled tabular dataset
│   ├── text/
│   │   ├── product_reviews_raw.csv    # Raw reviews with HTML, URLs, noise & emojis
│   │   └── product_reviews_clean.csv  # Tokenized and cleaned text reviews
│   └── image/
│       ├── raw/                       # Multi-class raw images of varying dimensions
│       └── processed/                 # Normalized and PCA-reduced feature arrays
│
├── notebooks/
│   ├── 01_tabular_preprocessing.ipynb # Interactive Tabular profiling, encoding, scaling & SMOTE
│   ├── 02_text_preprocessing.ipynb    # Interactive Text cleaning, TF-IDF & linguistic engineering
│   ├── 03_image_preprocessing.ipynb   # Interactive Image loading, resizing, normalization & PCA
│   └── 04_end_to_end_webinar2.ipynb   # Master end-to-end demo comparing Before vs After
│
├── src/
│   ├── __init__.py
│   ├── tabular/
│   │   ├── __init__.py
│   │   ├── profiling.py               # Profiling, missingness, IQR/Z-score outlier detection
│   │   ├── encoding.py                # OneHot, Ordinal, and dirty string cleaning
│   │   ├── scaling.py                 # StandardScaler, MinMaxScaler, RobustScaler
│   │   ├── imbalance.py               # SMOTE and class weighting
│   │   └── model_trainer.py           # Baseline vs Preprocessed tabular modeling
│   │
│   ├── text/
│   │   ├── __init__.py
│   │   ├── cleaning.py                # HTML stripping, regex normalization, contractions, stopwords
│   │   ├── vectorization.py           # TF-IDF unigram+bigram vectorizer
│   │   ├── feature_engineering.py     # Text length, uppercase ratio, punctuation, sentiment lexicon
│   │   └── imbalance.py               # Text classification pipeline & evaluation
│   │
│   ├── image/
│   │   ├── __init__.py
│   │   ├── loading.py                 # RGB dataset directory loader
│   │   ├── resizing.py                # Direct and aspect-preserving letterboxing
│   │   ├── normalization.py           # [0,1] normalization and channel standardization
│   │   └── pca_reduction.py           # PCA variance analysis, reconstruction & classifier
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Classification metrics suite
│   │   ├── comparison.py              # Before vs After delta calculation
│   │   └── visualizer.py              # Publication-grade visual plots
│   │
│   └── data_generators/
│       ├── __init__.py
│       ├── generate_all_datasets.py   # Synthetic data generators for all 3 modalities
│       └── build_notebooks.py         # Automated notebook generation script
│
└── reports/
    ├── master_before_vs_after_metrics.csv # Exported comparison metrics
    ├── before_vs_after_benchmarks.png     # Grouped bar charts
    ├── tabular_scaling_and_outliers.png   # Scaler distribution histograms
    ├── image_pca_and_reconstruction.png   # Scree plot & image reconstruction
    └── confusion_matrices_comparison.png  # Confusion matrices before vs after
```

---

## ⚡ Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
# Clone the repository
git clone <repository_url>
cd "Webinar 2"

# (Optional) Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate    # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline

Execute the master CLI pipeline to run all three modalities, print summary tables, and save visual reports:

```bash
python main.py
```

### 3. Launch Interactive Notebooks

To explore the step-by-step interactive demonstrations:

```bash
jupyter notebook
```

Navigate to `notebooks/` and open:
* `01_tabular_preprocessing.ipynb` — Tabular Data Deep Dive
* `02_text_preprocessing.ipynb` — NLP Text Cleaning & Feature Engineering
* `03_image_preprocessing.ipynb` — Computer Vision & PCA Dimensionality Reduction
* `04_end_to_end_webinar2.ipynb` — Unified Master Demonstration

---

## 🖼️ Generated Visual Reports

All figures are automatically generated into the `reports/` folder:

1. **`reports/before_vs_after_benchmarks.png`**: Side-by-side performance comparison across all modalities.
2. **`reports/tabular_scaling_and_outliers.png`**: Visual comparison showing how RobustScaler avoids outlier distortion.
3. **`reports/image_pca_and_reconstruction.png`**: PCA scree plot and comparison between original normalized images and reconstructed images.
4. **`reports/confusion_matrices_comparison.png`**: Baseline vs Preprocessed confusion matrices across Tabular, Text, and Image tasks.

---

## 🎓 Educational Takeaways

1. **Garbage In, Garbage Out**: Raw data contains missingness, noise, and severe class imbalance that significantly degrade baseline classifiers.
2. **Proper Scaling Matters**: `StandardScaler` and `MinMaxScaler` are sensitive to extreme outliers; `RobustScaler` (median & IQR) ensures stability.
3. **Imbalance Mitigation**: Measuring accuracy alone on imbalanced datasets is misleading. SMOTE and class-weighting drastically elevate **Recall** and **F1-Score** on critical minority classes.
4. **Dimensionality Reduction**: In computer vision, PCA reduces high-dimensional pixel matrices by $>99\%$ while preserving core variance and structure for downstream learning.
