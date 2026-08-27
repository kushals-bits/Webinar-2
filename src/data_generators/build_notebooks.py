"""
Script to build all 4 interactive Jupyter notebooks for Webinar 2: Data Preprocessing.
Uses nbformat to construct well-documented, runnable notebooks with rich explanations and plots.
"""

import os
import nbformat as nbf


def create_tabular_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "# Webinar 2: Data Preprocessing — Track 1: Tabular Pipeline\n"
        "This notebook covers the complete tabular preprocessing lifecycle:\n"
        "1. **Profiling**: Missing value audit, outlier diagnostics (IQR & Z-score), skewness, and cardinality.\n"
        "2. **Encoding**: Handling dirty data, One-Hot Encoding for nominal columns, Ordinal Encoding for ordered categories.\n"
        "3. **Scaling**: Comparing `StandardScaler`, `MinMaxScaler`, and `RobustScaler` on skewed features with extreme outliers.\n"
        "4. **Imbalance**: Handling skewed class distribution using **SMOTE** (Synthetic Minority Over-sampling Technique).\n"
        "5. **Train Model & Evaluation**: Comparing Baseline (unscaled, imbalanced) vs Preprocessed (encoded, robust-scaled, balanced) model performance."
    ))

    # Cell 1: Imports
    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "import os\n"
        "# Ensure src is on python path\n"
        "sys.path.append(os.path.abspath('..'))\n\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "from src.tabular import (\n"
        "    profile_dataframe,\n"
        "    print_profiling_report,\n"
        "    clean_raw_tabular_data,\n"
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
        "## Step 1: Data Profiling & Health Audit\n"
        "Before applying transformations, we inspect missingness, statistical distributions, skewness, and extreme outliers."
    ))
    cells.append(nbf.v4.new_code_cell(
        "df_raw = pd.read_csv('../data/tabular/customer_churn_raw.csv')\n"
        "print(f'Raw Tabular Dataset Shape: {df_raw.shape}')\n"
        "df_raw.head()"
    ))
    cells.append(nbf.v4.new_code_cell(
        "profile = profile_dataframe(df_raw, target_col='churn')\n"
        "print_profiling_report(profile)"
    ))

    # Step 2: Encoding
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 2: Data Cleaning & Categorical Encoding\n"
        "- Clean unphysical ages and string-formatted values in `total_charges` (`$`, spaces).\n"
        "- Apply `OneHotEncoder` for nominal columns (`payment_method`, `internet_service`, `tech_support`).\n"
        "- Apply `OrdinalEncoder` for ordered columns (`contract_type`: Month-to-month < One year < Two year)."
    ))
    cells.append(nbf.v4.new_code_cell(
        "df_cleaned = clean_raw_tabular_data(df_raw)\n"
        "X_clean = df_cleaned.drop(columns=['churn', 'customer_id'])\n"
        "y = df_cleaned['churn'].map({'Yes': 1, 'No': 0})\n\n"
        "encoder = TabularEncoder(\n"
        "    nominal_cols=['payment_method', 'internet_service', 'tech_support'],\n"
        "    ordinal_cols=['contract_type'],\n"
        "    ordinal_categories={'contract_type': ['Month-to-month', 'One year', 'Two year']}\n"
        ")\n"
        "X_encoded = encoder.fit_transform(X_clean)\n"
        "print(f'Encoded Feature Matrix Shape: {X_encoded.shape}')\n"
        "X_encoded.head()"
    ))

    # Step 3: Scaling
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 3: Feature Scaling Comparison\n"
        "Compare `StandardScaler`, `MinMaxScaler`, and `RobustScaler` on numerical features with extreme outliers."
    ))
    cells.append(nbf.v4.new_code_cell(
        "scaler_comparison = compare_scalers(X_encoded, num_cols=['monthly_charges', 'age', 'tenure_months'])\n"
        "scaler_comparison"
    ))
    cells.append(nbf.v4.new_code_cell(
        "from src.evaluation.visualizer import plot_tabular_scaling_and_outliers\n"
        "plot_tabular_scaling_and_outliers(df_raw, num_col='monthly_charges', output_path='../reports/tabular_scaling_and_outliers.png')\n\n"
        "# Apply RobustScaler\n"
        "scaler = TabularScaler(method='robust')\n"
        "X_scaled = scaler.fit_transform(X_encoded)\n"
        "X_scaled.head()"
    ))

    # Step 4: Imbalance
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 4: Handling Class Imbalance (SMOTE)\n"
        "Customer churn is naturally imbalanced. We use SMOTE to oversample minority churn cases in feature space."
    ))
    cells.append(nbf.v4.new_code_cell(
        "print('Original Class Distribution:')\n"
        "print(y.value_counts(normalize=True) * 100)\n\n"
        "X_balanced, y_balanced = balance_dataset(X_scaled, y, method='smote')\n"
        "print('\\nBalanced Class Distribution (After SMOTE):')\n"
        "print(y_balanced.value_counts())"
    ))

    # Step 5: Model Training & Evaluation
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 5: Model Training & Before vs After Evaluation\n"
        "Compare baseline model against the full preprocessed pipeline."
    ))
    cells.append(nbf.v4.new_code_cell(
        "tabular_results = run_tabular_pipeline('../data/tabular/customer_churn_raw.csv')\n\n"
        "from src.evaluation.comparison import generate_modality_comparison\n"
        "comp_df = generate_modality_comparison(tabular_results['baseline_metrics'], tabular_results['preprocessed_metrics'], 'Tabular')\n"
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
        "This notebook covers the complete NLP text preprocessing workflow:\n"
        "1. **Text Cleaning**: Removing HTML tags, URLs, emojis, noisy punctuation, lowercasing, and stop words.\n"
        "2. **TF-IDF Vectorization**: Extracting unigram + bigram representations with sublinear TF scaling.\n"
        "3. **Numerical Feature Engineering**: Extracting text length, word count, uppercase shouting ratio, and sentiment heuristics.\n"
        "4. **Imbalance Handling**: Addressing sentiment class imbalance using SMOTE on text embeddings.\n"
        "5. **Model Evaluation**: Comparing raw Bag-of-Words vs Cleaned + TF-IDF + Numerical Engineered Features."
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
        "## Step 1: Inspect Raw Dirty Text Data\n"
        "Observing real-world noise: HTML tags (`<br/>`, `<b>`), URLs, emojis, uppercase shouting, and contractions."
    ))
    cells.append(nbf.v4.new_code_cell(
        "df_text = pd.read_csv('../data/text/product_reviews_raw.csv')\n"
        "print(f'Total reviews: {len(df_text)}')\n"
        "df_text.head(10)"
    ))

    # Step 2: Cleaning Pipeline
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 2: Cleaning and Normalization\n"
        "Apply regular expressions, HTML stripping, contraction expansion, and stopword filtering."
    ))
    cells.append(nbf.v4.new_code_cell(
        "sample_raw = df_text['review_text'].iloc[0]\n"
        "sample_cleaned = clean_text(sample_raw)\n"
        "print(f'BEFORE CLEANING:\\n{sample_raw}\\n')\n"
        "print(f'AFTER CLEANING:\\n{sample_cleaned}')"
    ))
    cells.append(nbf.v4.new_code_cell(
        "cleaned_texts = batch_clean_texts(df_text['review_text'])\n"
        "df_text['cleaned_review'] = cleaned_texts\n"
        "df_text[['review_text', 'cleaned_review']].head()"
    ))

    # Step 3: TF-IDF
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 3: TF-IDF Vectorization with N-Grams\n"
        "Convert text into weighted vectors capturing both individual words and 2-word phrases."
    ))
    cells.append(nbf.v4.new_code_cell(
        "tfidf = TFIDFProcessor(max_features=300, ngram_range=(1, 2), sublinear_tf=True)\n"
        "tfidf_matrix = tfidf.fit_transform(cleaned_texts)\n"
        "print(f'TF-IDF Matrix Shape: {tfidf_matrix.shape}')\n"
        "top_kw = tfidf.get_top_keywords(cleaned_texts, top_n=10)\n"
        "top_kw"
    ))

    # Step 4: Numerical Features
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 4: Numerical Feature Engineering\n"
        "Extract structured numerical features directly from text characteristics."
    ))
    cells.append(nbf.v4.new_code_cell(
        "num_feats = extract_numerical_text_features(df_text['review_text'])\n"
        "num_feats.head()"
    ))

    # Step 5: Full NLP Pipeline Execution
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 5: Full NLP Pipeline Execution & Evaluation"
    ))
    cells.append(nbf.v4.new_code_cell(
        "text_results = run_text_pipeline('../data/text/product_reviews_raw.csv')\n\n"
        "from src.evaluation.comparison import generate_modality_comparison\n"
        "comp_df = generate_modality_comparison(text_results['baseline_metrics'], text_results['preprocessed_metrics'], 'Text')\n"
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
        "This notebook covers the computer vision preprocessing and dimensionality reduction workflow:\n"
        "1. **Loading**: Reading multi-class images from directory structures and ensuring RGB channel consistency.\n"
        "2. **Resize**: Aspect-ratio preserving letterboxing vs direct resize.\n"
        "3. **Normalize**: Pixel intensity scaling $[0, 255] \\to [0.0, 1.0]$ and channel-wise standardization.\n"
        "4. **PCA**: Principal Component Analysis for dimensionality reduction (>99% feature compression).\n"
        "5. **Reduced Features & Reconstruction**: Scree plots, variance retention, and image reconstruction from principal components."
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
        "## Step 1: Loading & Visualizing Raw Images"
    ))
    cells.append(nbf.v4.new_code_cell(
        "raw_images, labels, file_paths, class_to_idx = load_image_dataset('../data/image/raw')\n\n"
        "fig, axes = plt.subplots(1, 5, figsize=(15, 3))\n"
        "for i, ax in enumerate(axes):\n"
        "    ax.imshow(raw_images[i])\n"
        "    ax.set_title(f'{labels[i]}\\nSize: {raw_images[i].size}')\n"
        "    ax.axis('off')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # Step 2: Resizing
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 2: Standardizing Image Resolution (Letterbox vs Direct)"
    ))
    cells.append(nbf.v4.new_code_cell(
        "resized_images = batch_resize_images(raw_images, target_size=(64, 64), preserve_aspect=True)\n"
        "print(f'Standardized size: {resized_images[0].size}')"
    ))

    # Step 3: Normalization
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 3: Pixel Intensity Normalization [0.0, 1.0]"
    ))
    cells.append(nbf.v4.new_code_cell(
        "img_tensor = images_to_numpy(resized_images)\n"
        "norm_tensor = normalize_minmax(img_tensor)\n"
        "print(f'Raw min/max: {img_tensor.min():.1f}, {img_tensor.max():.1f}')\n"
        "print(f'Normalized min/max: {norm_tensor.min():.1f}, {norm_tensor.max():.1f}')"
    ))

    # Step 4: PCA
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 4: PCA Dimensionality Reduction & Scree Plot\n"
        "Reduce 12,288 flattened pixel dimensions to principal components retaining 95% variance."
    ))
    cells.append(nbf.v4.new_code_cell(
        "X_flat = flatten_images(norm_tensor)\n"
        "pca_engine = ImagePCA(n_components=0.95)\n"
        "X_pca = pca_engine.fit_transform(X_flat)\n"
        "print(f'Original dimensions: {X_flat.shape[1]}')\n"
        "print(f'Reduced PCA dimensions: {X_pca.shape[1]} ({(1 - X_pca.shape[1]/X_flat.shape[1])*100:.1f}% reduction)')\n"
        "pca_engine.get_variance_summary(top_k=8)"
    ))

    # Step 5: Reconstruction & End-to-End Run
    cells.append(nbf.v4.new_markdown_cell(
        "## Step 5: Image Reconstruction from PCA & Model Evaluation"
    ))
    cells.append(nbf.v4.new_code_cell(
        "image_results = run_image_pipeline('../data/image/raw')\n\n"
        "from src.evaluation.visualizer import plot_image_pca_and_reconstruction\n"
        "plot_image_pca_and_reconstruction(image_results, output_path='../reports/image_pca_and_reconstruction.png')\n\n"
        "from src.evaluation.comparison import generate_modality_comparison\n"
        "comp_df = generate_modality_comparison(image_results['baseline_metrics'], image_results['preprocessed_metrics'], 'Image')\n"
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
        "# Webinar 2: Data Preprocessing — Master End-to-End Demonstration\n\n"
        "## Overview & Architecture\n"
        "This master notebook demonstrates the complete unified workflow for **Webinar 2: Data Preprocessing**:\n\n"
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
        "| • Train Model   |         +--------+--------+         | • Reduced Feat  |\n"
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
        "## 1. Running All Preprocessing Pipelines\n"
        "Executing Tabular, Text, and Image pipelines."
    ))
    cells.append(nbf.v4.new_code_cell(
        "print('--- Running Tabular Pipeline ---')\n"
        "tabular_results = run_tabular_pipeline('../data/tabular/customer_churn_raw.csv')\n\n"
        "print('--- Running Text Pipeline ---')\n"
        "text_results = run_text_pipeline('../data/text/product_reviews_raw.csv')\n\n"
        "print('--- Running Image Pipeline ---')\n"
        "image_results = run_image_pipeline('../data/image/raw')\n\n"
        "print('All 3 pipelines executed successfully!')"
    ))

    # Downstream Convergence: Master Before vs After Comparison
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Downstream Convergence: Master Before vs After Comparison\n"
        "Synthesizing results into a unified comparative benchmark table."
    ))
    cells.append(nbf.v4.new_code_cell(
        "master_df = build_master_comparison_report(tabular_results, text_results, image_results)\n"
        "print_master_comparison(master_df)\n"
        "master_df"
    ))

    # Visualizing Reports
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Comparative Visualizations\n"
        "Generating publication-quality comparison charts."
    ))
    cells.append(nbf.v4.new_code_cell(
        "generate_all_report_visuals(tabular_results, text_results, image_results, master_df)\n\n"
        "from IPython.display import Image, display\n"
        "print('Benchmark Comparison:')\n"
        "display(Image(filename='../reports/before_vs_after_benchmarks.png'))\n\n"
        "print('Tabular Scaling & Outliers:')\n"
        "display(Image(filename='../reports/tabular_scaling_and_outliers.png'))\n\n"
        "print('Image PCA & Reconstruction:')\n"
        "display(Image(filename='../reports/image_pca_and_reconstruction.png'))\n\n"
        "print('Confusion Matrices:')\n"
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
    print("All 4 Jupyter Notebooks built successfully!")


if __name__ == "__main__":
    build_all()
