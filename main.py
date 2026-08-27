"""
Main Execution Script for Webinar 2: Data Preprocessing (UPDATED)
Uses REAL practical datasets and different ML/NLP algorithms vs Webinar 1.

DATASETS:
  - Tabular: IBM Telco Customer Churn (7,043 real records)
  - Text:    20 Newsgroups — sci.med vs alt.atheism (real internet forum posts)
  - Image:   MNIST Handwritten Digits 0/1/2 (real human handwriting)

ALGORITHMS (different from Webinar 1's Logistic Regression + Random Forest):
  - Tabular: XGBoost / LightGBM / GradientBoosting (Boosting Ensembles)
  - Text:    LinearSVC (Support Vector Machine) + SGD Classifier
  - Image:   SVM with RBF Kernel + KNeighborsClassifier (on PCA features)
"""

import os
import sys
import pandas as pd

from src.data_generators.download_real_datasets import download_all
from src.tabular import profile_dataframe, print_profiling_report, run_tabular_pipeline
from src.text import run_text_pipeline
from src.image import run_image_pipeline
from src.evaluation import build_master_comparison_report, print_master_comparison, generate_all_report_visuals


TABULAR_CSV = "data/tabular/telco_churn_raw.csv"
TEXT_CSV = "data/text/newsgroups_raw.csv"
IMAGE_DIR = "data/image/raw"


def main():
    print("*" * 90)
    print(" " * 15 + "WEBINAR 2: DATA PREPROCESSING — REAL DATASETS + NEW ALGORITHMS")
    print("*" * 90)
    print("\nDatasets  : IBM Telco Churn | 20 Newsgroups | MNIST Digits")
    print("Algorithms: XGBoost | LinearSVC (SVM) | SVM RBF + KNN")
    print("*" * 90)

    # 1. Download / verify all real datasets
    datasets_exist = (
        os.path.exists(TABULAR_CSV) and
        os.path.exists(TEXT_CSV) and
        os.path.exists(IMAGE_DIR) and
        any(os.listdir(IMAGE_DIR)) if os.path.exists(IMAGE_DIR) else False
    )
    if not datasets_exist:
        print("\nDownloading real practical datasets...")
        download_all()
    else:
        print("\nReal datasets verified on disk.")

    # ─────────────────────────────────────────────────────────────────
    # TRACK 1: TABULAR PREPROCESSING (IBM Telco Churn + XGBoost)
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(">>> TRACK 1: TABULAR — IBM Telco Customer Churn")
    print("    Dataset: 7,043 real customer records | Algorithm: XGBoost vs Logistic Regression")
    print("=" * 90)

    df_tabular_raw = pd.read_csv(TABULAR_CSV)
    profile = profile_dataframe(df_tabular_raw, target_col="Churn")
    print_profiling_report(profile)

    print("\nRunning: Cleaning -> One-Hot/Ordinal Encoding -> RobustScaling -> SMOTE -> XGBoost...")
    tabular_results = run_tabular_pipeline(TABULAR_CSV)
    print(f"Model trained: {tabular_results['model_name']}")

    # ─────────────────────────────────────────────────────────────────
    # TRACK 2: TEXT PREPROCESSING (20 Newsgroups + LinearSVC)
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(">>> TRACK 2: TEXT - 20 Newsgroups (sci.med vs alt.atheism)")
    print("    Dataset: Real internet forum posts | Algorithm: LinearSVC vs SGD Classifier")
    print("=" * 90)
    print("Running: Cleaning -> TF-IDF (1,2)-grams -> Numerical Features -> SMOTE -> LinearSVC...")
    text_results = run_text_pipeline(TEXT_CSV)
    print("Top 10 TF-IDF Keywords:")
    print(text_results["tfidf_top_keywords"].to_string(index=False))

    # ─────────────────────────────────────────────────────────────────
    # TRACK 3: IMAGE PREPROCESSING (MNIST + SVM RBF)
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(">>> TRACK 3: IMAGE - MNIST Handwritten Digits (0, 1, 2)")
    print("    Dataset: Real handwritten digits | Algorithm: SVM RBF + KNN vs Logistic Regression")
    print("=" * 90)
    print("Running: Loading -> Resize -> [0,1] Normalization -> PCA -> SVM RBF + KNN...")
    image_results = run_image_pipeline(IMAGE_DIR)
    pca_summary = image_results["image_pca"].get_variance_summary(top_k=5)
    print("Top PCA Components Variance:")
    print(pca_summary.to_string(index=False))
    print(f"KNN (k=5) on PCA features — F1 Macro: {image_results['knn_metrics']['F1-Score (Macro)']:.4f}")

    # ─────────────────────────────────────────────────────────────────
    # DOWNSTREAM: EVALUATION & BEFORE vs AFTER COMPARISON
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(">>> DOWNSTREAM: EVALUATION & BEFORE VS AFTER COMPARISON")
    print("=" * 90)
    master_df = build_master_comparison_report(tabular_results, text_results, image_results)
    print_master_comparison(master_df)

    os.makedirs("reports", exist_ok=True)
    master_df.to_csv("reports/master_before_vs_after_metrics.csv", index=False)
    print("Saved: reports/master_before_vs_after_metrics.csv")

    generate_all_report_visuals(tabular_results, text_results, image_results, master_df)

    print("\n" + "*" * 90)
    print(" " * 20 + "WEBINAR 2 (REAL DATASETS + NEW ALGORITHMS) — COMPLETED!")
    print("*" * 90)


if __name__ == "__main__":
    main()
