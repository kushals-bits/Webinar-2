"""
Main Execution Script for Webinar 2: Data Preprocessing
Orchestrates Tabular, Text, and Image preprocessing pipelines, runs evaluation, and generates Before vs After comparative reports.
"""

import os
import sys
import pandas as pd

from src.data_generators import generate_all
from src.tabular import profile_dataframe, print_profiling_report, run_tabular_pipeline
from src.text import run_text_pipeline
from src.image import run_image_pipeline
from src.evaluation import build_master_comparison_report, print_master_comparison, generate_all_report_visuals


def main():
    print("*" * 90)
    print(" " * 20 + "STARTING WEBINAR 2: DATA PREPROCESSING PIPELINE")
    print("*" * 90)

    # 1. Ensure datasets exist
    if not (os.path.exists("data/tabular/customer_churn_raw.csv") and
            os.path.exists("data/text/product_reviews_raw.csv") and
            os.path.exists("data/image/raw")):
        print("\nDatasets not found. Generating synthetic datasets...")
        generate_all()
    else:
        print("\nDatasets verified on disk.")

    # -------------------------------------------------------------
    # TRACK 1: TABULAR PREPROCESSING PIPELINE
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print(">>> EXECUTING TRACK 1: TABULAR PREPROCESSING PIPELINE")
    print("=" * 90)
    
    # Profiling
    df_tabular_raw = pd.read_csv("data/tabular/customer_churn_raw.csv")
    profile = profile_dataframe(df_tabular_raw, target_col="churn")
    print_profiling_report(profile)

    # Execute Tabular Pipeline (Encoding -> Scaling -> Imbalance -> Model)
    print("\nRunning Tabular: Encoding -> RobustScaling -> SMOTE -> Model Training...")
    tabular_results = run_tabular_pipeline()
    print("Tabular Pipeline completed successfully.")

    # -------------------------------------------------------------
    # TRACK 2: TEXT PREPROCESSING PIPELINE
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print(">>> EXECUTING TRACK 2: TEXT PREPROCESSING PIPELINE")
    print("=" * 90)
    print("Running Text: Cleaning -> TF-IDF (1,2) -> Numerical Features -> SMOTE -> Model Training...")
    text_results = run_text_pipeline()
    print("Top TF-IDF Keywords extracted:")
    print(text_results["tfidf_top_keywords"].to_string(index=False))
    print("Text Pipeline completed successfully.")

    # -------------------------------------------------------------
    # TRACK 3: IMAGE PREPROCESSING PIPELINE
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print(">>> EXECUTING TRACK 3: IMAGE PREPROCESSING PIPELINE")
    print("=" * 90)
    print("Running Image: Loading -> Aspect-Preserving Resize -> [0,1] Normalization -> PCA -> Model Training...")
    image_results = run_image_pipeline()
    pca_summary = image_results["image_pca"].get_variance_summary(top_k=5)
    print("Top Principal Components Explained Variance:")
    print(pca_summary.to_string(index=False))
    print("Image Pipeline completed successfully.")

    # -------------------------------------------------------------
    # DOWNSTREAM CONVERGENCE: EVALUATION & BEFORE VS AFTER COMPARISON
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print(">>> DOWNSTREAM CONVERGENCE: EVALUATION & BEFORE VS AFTER COMPARISON")
    print("=" * 90)
    master_comparison_df = build_master_comparison_report(tabular_results, text_results, image_results)
    print_master_comparison(master_comparison_df)

    # Save summary report to CSV
    os.makedirs("reports", exist_ok=True)
    master_comparison_df.to_csv("reports/master_before_vs_after_metrics.csv", index=False)
    print("Saved metrics summary to: reports/master_before_vs_after_metrics.csv")

    # Generate Publication-Quality Visual Charts
    generate_all_report_visuals(tabular_results, text_results, image_results, master_comparison_df)

    print("\n" + "*" * 90)
    print(" " * 20 + "WEBINAR 2 PIPELINE EXECUTION FINISHED SUCCESSFULLY!")
    print("*" * 90)


if __name__ == "__main__":
    main()
