"""
Script to build all 4 interactive Jupyter notebooks for Webinar 2: Data Preprocessing.
Updated with REAL Practical Datasets and Advanced ML/NLP Algorithms (XGBoost, LinearSVC, SVM RBF + KNN).
"""

import os
import nbformat as nbf


def create_tabular_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "# Webinar 2: Data Preprocessing — Track 1: Tabular Pipeline\n"
        "### Dataset: Real IBM Telco Customer Churn (7,043 customer records)\n"
        "### Algorithms: XGBoost & LightGBM vs Baseline Logistic Regression\n\n"
        "This notebook covers the complete tabular preprocessing lifecycle on real-world business data:\n"
        "1. **Profiling**: Missing value audit in `TotalCharges`, extreme outliers in `MonthlyCharges`, skewness, and cardinality.\n"
        "2. **Encoding**: One-Hot Encoding for nominal columns (`PaymentMethod`, `InternetService`), Ordinal Encoding for `Contract`.\n"
        "3. **Scaling**: Comparing `StandardScaler`, `MinMaxScaler`, and `RobustScaler` on skewed billing features.\n"
        "4. **Imbalance**: Handling the ~27% churn minority class using **SMOTE**.\n"
        "5. **Train Model & Evaluation**: Comparing naive baseline Logistic Regression vs preprocessed **XGBoost Classifier**."
    ))

    # Cell 1: Imports
    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "import os\n"
        "sys.path.append(os.path.abspath('..'))\n\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "from src.tabular import (\n"
        "    profile_dataframe,\n"
        "    print_profiling_report,\n"
        "    TabularEncoder,\n"
        "    TabularScaler,\n"
        "    compare_scalers,\n"
        "    balance_dataset,\n"
        "    run_tabular_pipeline\n"
        ")\n\n"
        "pd.set_option('display.max_columns', None)\n"
        "print('Imports loaded successfully!')"
    ))

    # Step 1: Profiling
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 1: Data Profiling & Health Audit (IBM Telco Churn)\n"
        "Inspecting missing values, string anomalies, distributions, and class imbalance."
    ))
    cells.append(nbf.v4.new_code_cell(
        "df_raw = pd.read_csv('../data/tabular/telco_churn_raw.csv')\n"
        "print(f'IBM Telco Dataset Shape: {df_raw.shape}')\n"
        "df_raw.head()"
    ))
    cells.append(nbf.v4.new_code_cell(
        "# Convert TotalCharges to numeric for profiling\n"
        "df_profile = df_raw.copy()\n"
        "df_profile['TotalCharges'] = pd.to_numeric(df_profile['TotalCharges'].astype(str).str.strip(), errors='coerce')\n"
        "profile = profile_dataframe(df_profile, target_col='Churn')\n"
        "print_profiling_report(profile)"
    ))

    # Step 2: Encoding
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 2: Categorical Encoding (Nominal vs Ordinal)\n"
        "- Nominal features (`PaymentMethod`, `InternetService`, etc.) $\\to$ **One-Hot Encoding**\n"
        "- Ordinal feature (`Contract`: Month-to-month < One year < Two year) $\\to$ **Ordinal Encoding**"
    ))
    cells.append(nbf.v4.new_code_cell(
        "# Prepare binary and categorical features\n"
        "X_clean = df_raw.drop(columns=['Churn', 'customerID'])\n"
        "X_clean['TotalCharges'] = pd.to_numeric(X_clean['TotalCharges'].astype(str).str.strip(), errors='coerce')\n"
        "y = df_raw['Churn'].map({'Yes': 1, 'No': 0})\n\n"
        "nominal_cols = ['PaymentMethod', 'InternetService', 'OnlineSecurity', 'DeviceProtection']\n"
        "ordinal_cols = ['Contract']\n"
        "ordinal_order = {'Contract': ['Month-to-month', 'One year', 'Two year']}\n\n"
        "encoder = TabularEncoder(\n"
        "    nominal_cols=nominal_cols,\n"
        "    ordinal_cols=ordinal_cols,\n"
        "    ordinal_categories=ordinal_order\n"
        ")\n"
        "X_encoded = encoder.fit_transform(X_clean)\n"
        "print(f'Encoded Feature Matrix Shape: {X_encoded.shape}')\n"
        "X_encoded.head()"
    ))

    # Step 3: Scaling
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 3: Feature Scaling Comparison (Standard vs MinMax vs Robust)\n"
        "Comparing scaler resistance to extreme charges."
    ))
    cells.append(nbf.v4.new_code_cell(
        "scaler_comparison = compare_scalers(X_encoded, num_cols=['MonthlyCharges', 'tenure'])\n"
        "scaler_comparison"
    ))
    cells.append(nbf.v4.new_code_cell(
        "from src.evaluation.visualizer import plot_tabular_scaling_and_outliers\n"
        "plot_tabular_scaling_and_outliers(df_profile, num_col='MonthlyCharges', output_path='../reports/tabular_scaling_and_outliers.png')\n\n"
        "scaler = TabularScaler(method='robust')\n"
        "X_scaled = scaler.fit_transform(X_encoded)\n"
        "X_scaled.head()"
    ))

    # Step 4: Imbalance
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 4: Class Imbalance Handling with SMOTE"
    ))
    cells.append(nbf.v4.new_code_cell(
        "print('Original Churn Distribution:')\n"
        "print(y.value_counts(normalize=True) * 100)\n\n"
        "X_balanced, y_balanced = balance_dataset(X_scaled, y, method='smote')\n"
        "print('\\nBalanced Distribution After SMOTE:')\n"
        "print(y_balanced.value_counts())"
    ))

    # Step 5: XGBoost Training & Evaluation
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 5: Model Training (XGBoost vs Baseline) & Evaluation"
    ))
    cells.append(nbf.v4.new_code_cell(
        "tabular_results = run_tabular_pipeline('../data/tabular/telco_churn_raw.csv')\n\n"
        "from src.evaluation.comparison import generate_modality_comparison\n"
        "comp_df = generate_modality_comparison(tabular_results['baseline_metrics'], tabular_results['preprocessed_metrics'], 'Tabular (Telco Churn)')\n"
        "comp_df"
    ))

    nb.cells = cells
    os.makedirs("notebooks", exist_ok=True)
    with open("notebooks/01_tabular_preprocessing.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("-> Generated notebooks/01_tabular_preprocessing.ipynb")


def create_text_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "# Webinar 2: Data Preprocessing — Track 2: Text Pipeline\n"
        "### Dataset: 20 Newsgroups (Real Internet Forum Discussions — sci.med vs alt.atheism)\n"
        "### Algorithms: LinearSVC (SVM) vs Baseline SGD Classifier\n\n"
        "This notebook covers the complete NLP text preprocessing workflow on real discussion posts:\n"
        "1. **Text Cleaning**: Removing email artifacts, quotation headers, special characters, and stopwords.\n"
        "2. **TF-IDF Vectorization**: Extracting unigram + bigram representations with sublinear TF scaling.\n"
        "3. **Numerical Feature Engineering**: Extracting text length, uppercase shouting ratio, punctuation density.\n"
        "4. **Imbalance Handling**: Addressing class imbalance using SMOTE on text embeddings.\n"
        "5. **Model Evaluation**: Comparing raw Bag-of-Words SGD Classifier vs Cleaned TF-IDF **LinearSVC**."
    ))

    # Cell 1: Imports
    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "import os\n"
        "sys.path.append(os.path.abspath('..'))\n\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n\n"
        "from src.text import (\n"
        "    clean_text,\n"
        "    batch_clean_texts,\n"
        "    TFIDFProcessor,\n"
        "    extract_numerical_text_features,\n"
        "    run_text_pipeline\n"
        ")\n\n"
        "print('Text NLP modules imported successfully!')"
    ))

    # Step 1: Inspect Raw Text
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 1: Inspect Real Internet Forum Posts\n"
        "Observing real-world noise: quotation tags, email fragments, contractions, and typos."
    ))
    cells.append(nbf.v4.new_code_cell(
        "df_text = pd.read_csv('../data/text/newsgroups_raw.csv')\n"
        "print(f'Total documents: {len(df_text)}')\n"
        "df_text.head()"
    ))

    # Step 2: Cleaning Pipeline
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 2: Cleaning & Text Normalization"
    ))
    cells.append(nbf.v4.new_code_cell(
        "sample_raw = df_text['raw_text'].iloc[0]\n"
        "sample_cleaned = clean_text(sample_raw)\n"
        "print(f'BEFORE CLEANING (first 250 chars):\\n{sample_raw[:250]}...\\n')\n"
        "print(f'AFTER CLEANING (first 250 chars):\\n{sample_cleaned[:250]}...')"
    ))

    # Step 3: TF-IDF
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 3: TF-IDF Vectorization with N-Grams"
    ))
    cells.append(nbf.v4.new_code_cell(
        "cleaned_texts = batch_clean_texts(df_text['raw_text'])\n"
        "tfidf = TFIDFProcessor(max_features=500, ngram_range=(1, 2), sublinear_tf=True)\n"
        "tfidf_matrix = tfidf.fit_transform(cleaned_texts)\n"
        "print(f'TF-IDF Matrix Shape: {tfidf_matrix.shape}')\n"
        "top_kw = tfidf.get_top_keywords(cleaned_texts, top_n=10)\n"
        "top_kw"
    ))

    # Step 4: Numerical Features
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 4: Linguistic Numerical Feature Engineering"
    ))
    cells.append(nbf.v4.new_code_cell(
        "num_feats = extract_numerical_text_features(df_text['raw_text'])\n"
        "num_feats.head()"
    ))

    # Step 5: Full NLP Pipeline Execution
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 5: Full NLP Pipeline Execution (LinearSVC vs SGD)"
    ))
    cells.append(nbf.v4.new_code_cell(
        "text_results = run_text_pipeline('../data/text/newsgroups_raw.csv')\n\n"
        "from src.evaluation.comparison import generate_modality_comparison\n"
        "comp_df = generate_modality_comparison(text_results['baseline_metrics'], text_results['preprocessed_metrics'], 'Text (20 Newsgroups)')\n"
        "comp_df"
    ))

    nb.cells = cells
    with open("notebooks/02_text_preprocessing.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("-> Generated notebooks/02_text_preprocessing.ipynb")


def create_image_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "# Webinar 2: Data Preprocessing — Track 3: Image Pipeline\n"
        "### Dataset: Real Handwritten Digits (MNIST / Digits — Classes 0, 1, 2)\n"
        "### Algorithms: SVM (RBF Kernel) & KNN vs Baseline Logistic Regression\n\n"
        "This notebook covers the computer vision preprocessing and PCA compression workflow on real handwritten digits:\n"
        "1. **Loading**: Reading multi-class handwritten digit images.\n"
        "2. **Resize**: Aspect-ratio preserving letterboxing to standardized resolution.\n"
        "3. **Normalize**: Pixel intensity scaling $[0, 255] \\to [0.0, 1.0]$.\n"
        "4. **PCA**: Principal Component Analysis for dimensionality reduction (>95% variance retention).\n"
        "5. **Modeling & Reconstruction**: Training **SVM with RBF Kernel** & **KNN (k=5)** on compact PCA features."
    ))

    # Cell 1: Imports
    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "import os\n"
        "sys.path.append(os.path.abspath('..'))\n\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n\n"
        "from src.image import (\n"
        "    load_image_dataset,\n"
        "    batch_resize_images,\n"
        "    images_to_numpy,\n"
        "    normalize_minmax,\n"
        "    flatten_images,\n"
        "    ImagePCA,\n"
        "    run_image_pipeline\n"
        ")\n\n"
        "print('Image preprocessing modules loaded!')"
    ))

    # Step 1: Loading
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 1: Loading Real Handwritten Digit Images"
    ))
    cells.append(nbf.v4.new_code_cell(
        "raw_images, labels, file_paths, class_to_idx = load_image_dataset('../data/image/raw')\n\n"
        "fig, axes = plt.subplots(1, 6, figsize=(15, 3))\n"
        "for i, ax in enumerate(axes):\n"
        "    idx = i * (len(raw_images) // 6)\n"
        "    ax.imshow(raw_images[idx])\n"
        "    ax.set_title(f'Digit: {labels[idx]}\\nSize: {raw_images[idx].size}')\n"
        "    ax.axis('off')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # Step 2: Resizing & Normalization
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 2: Resizing and Pixel Normalization"
    ))
    cells.append(nbf.v4.new_code_cell(
        "resized_images = batch_resize_images(raw_images, target_size=(28, 28), preserve_aspect=True)\n"
        "img_tensor = images_to_numpy(resized_images)\n"
        "norm_tensor = normalize_minmax(img_tensor)\n"
        "print(f'Tensor shape: {norm_tensor.shape}')\n"
        "print(f'Pixel range: [{norm_tensor.min():.2f}, {norm_tensor.max():.2f}]')"
    ))

    # Step 3: PCA
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 3: PCA Dimensionality Reduction & Scree Plot"
    ))
    cells.append(nbf.v4.new_code_cell(
        "X_flat = flatten_images(norm_tensor)\n"
        "pca_engine = ImagePCA(n_components=0.95)\n"
        "X_pca = pca_engine.fit_transform(X_flat)\n"
        "print(f'Raw pixel dimensions: {X_flat.shape[1]}')\n"
        "print(f'PCA reduced dimensions: {X_pca.shape[1]} ({(1 - X_pca.shape[1]/X_flat.shape[1])*100:.1f}% compression)')\n"
        "pca_engine.get_variance_summary(top_k=8)"
    ))

    # Step 4: Full Image Pipeline Execution
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 4: Full Image Pipeline Execution (SVM RBF + KNN vs Baseline)"
    ))
    cells.append(nbf.v4.new_code_cell(
        "image_results = run_image_pipeline('../data/image/raw')\n\n"
        "from src.evaluation.visualizer import plot_image_pca_and_reconstruction\n"
        "plot_image_pca_and_reconstruction(image_results, output_path='../reports/image_pca_and_reconstruction.png')\n\n"
        "from src.evaluation.comparison import generate_modality_comparison\n"
        "comp_df = generate_modality_comparison(image_results['baseline_metrics'], image_results['preprocessed_metrics'], 'Image (Handwritten Digits)')\n"
        "comp_df"
    ))

    nb.cells = cells
    with open("notebooks/03_image_preprocessing.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("-> Generated notebooks/03_image_preprocessing.ipynb")


def create_master_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "# Webinar 2: Data Preprocessing — Master End-to-End Demonstration\n"
        "### Real Practical Datasets: IBM Telco Churn | 20 Newsgroups | MNIST Digits\n"
        "### Modern Algorithms: XGBoost | LinearSVC (SVM) | SVM RBF + KNN\n\n"
        "## Architecture\n"
        "```\n"
        "                      +-----------------------------+\n"
        "                      |          WEBINAR 2          |\n"
        "                      |      DATA PREPROCESSING     |\n"
        "                      +--------------+--------------+\n"
        "                                     |\n"
        "         +---------------------------+---------------------------+\n"
        "         |                           |                           |\n"
        "         v                           v                           v\n"
        "+-----------------+         +-----------------+         +-----------------+\n"
        "|     TABULAR     |         |      TEXT       |         |      IMAGE      |\n"
        "+-----------------+         +-----------------+         +-----------------+\n"
        "| • Profiling     |         | • Cleaning      |         | • Loading       |\n"
        "| • Encoding      |         | • TF-IDF        |         | • Resize        |\n"
        "| • Scaling       |         | • Numerical Feat|         | • Normalize     |\n"
        "| • Imbalance     |         | • Imbalance     |         | • PCA           |\n"
        "| • Train (XGBoost|         +--------+--------+         | • Reduced Feat  |\n"
        "+--------+--------+                  |                  +--------+--------+\n"
        "         |                           +-----------+               |\n"
        "         |                                       |               |\n"
        "         +---------------------------+-----------+---------------+\n"
        "                                     |\n"
        "                                     v\n"
        "                        +-------------------------+\n"
        "                        |       Evaluation        |\n"
        "                        +------------+------------+\n"
        "                                     |\n"
        "                                     v\n"
        "                        +-------------------------+\n"
        "                        |  Compare Before vs After|\n"
        "                        +-------------------------+\n"
        "```"
    ))

    # Cell 1: Imports & Setup
    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "import os\n"
        "sys.path.append(os.path.abspath('..'))\n\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n\n"
        "from src.tabular import run_tabular_pipeline\n"
        "from src.text import run_text_pipeline\n"
        "from src.image import run_image_pipeline\n"
        "from src.evaluation import build_master_comparison_report, print_master_comparison, generate_all_report_visuals\n\n"
        "print('All Webinar 2 modules loaded successfully!')"
    ))

    # Execution of all 3 pipelines
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Running All 3 Preprocessing Pipelines on Real Datasets"
    ))
    cells.append(nbf.v4.new_code_cell(
        "print('--- 1. Running Tabular Pipeline (IBM Telco Churn + XGBoost) ---')\n"
        "tabular_results = run_tabular_pipeline('../data/tabular/telco_churn_raw.csv')\n\n"
        "print('\\n--- 2. Running Text Pipeline (20 Newsgroups + LinearSVC) ---')\n"
        "text_results = run_text_pipeline('../data/text/newsgroups_raw.csv')\n\n"
        "print('\\n--- 3. Running Image Pipeline (Handwritten Digits + SVM RBF) ---')\n"
        "image_results = run_image_pipeline('../data/image/raw')\n\n"
        "print('\\nAll 3 pipelines completed successfully!')"
    ))

    # Downstream Convergence: Master Before vs After Comparison
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Downstream Convergence: Master Before vs After Comparison"
    ))
    cells.append(nbf.v4.new_code_cell(
        "master_df = build_master_comparison_report(tabular_results, text_results, image_results)\n"
        "print_master_comparison(master_df)\n"
        "master_df"
    ))

    # Visualizing Reports
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Publication-Grade Visual Reports"
    ))
    cells.append(nbf.v4.new_code_cell(
        "generate_all_report_visuals(tabular_results, text_results, image_results, master_df)\n\n"
        "from IPython.display import Image, display\n"
        "print('1. Benchmark Comparison Across Modalities:')\n"
        "display(Image(filename='../reports/before_vs_after_benchmarks.png'))\n\n"
        "print('2. Tabular Feature Scaling & Outlier Diagnostics:')\n"
        "display(Image(filename='../reports/tabular_scaling_and_outliers.png'))\n\n"
        "print('3. Image PCA Scree Plot & Reconstruction:')\n"
        "display(Image(filename='../reports/image_pca_and_reconstruction.png'))\n\n"
        "print('4. Confusion Matrices (Before vs After):')\n"
        "display(Image(filename='../reports/confusion_matrices_comparison.png'))"
    ))

    nb.cells = cells
    with open("notebooks/04_end_to_end_webinar2.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("-> Generated notebooks/04_end_to_end_webinar2.ipynb")


def build_all():
    create_tabular_notebook()
    create_text_notebook()
    create_image_notebook()
    create_master_notebook()
    print("All 4 interactive Jupyter notebooks built successfully!")


if __name__ == "__main__":
    build_all()
