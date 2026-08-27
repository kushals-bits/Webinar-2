"""
Tabular Model Training Module
Trains and evaluates baseline models vs fully preprocessed models.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from .encoding import clean_raw_tabular_data, TabularEncoder
from .scaling import TabularScaler
from .imbalance import balance_dataset


def run_tabular_pipeline(csv_path="data/tabular/customer_churn_raw.csv", target_col="churn", random_state=42):
    """
    Executes the entire Tabular Preprocessing and Modeling pipeline:
    1. Loads raw data
    2. Builds Baseline Model (Raw, unscaled, naive imputation, imbalanced)
    3. Runs Full Preprocessing:
       - Cleaning dirty strings and out-of-range records
       - One-Hot and Ordinal Encoding
       - Feature Scaling (RobustScaler)
       - Imbalance handling (SMOTE)
    4. Trains Preprocessed Model (Logistic Regression & Random Forest)
    5. Returns evaluation metrics for both Baseline and Preprocessed
    """
    df_raw = pd.read_csv(csv_path)
    
    # Target encoding: 'Yes' -> 1, 'No' -> 0
    y_raw = df_raw[target_col].map({"Yes": 1, "No": 0})
    X_raw_df = df_raw.drop(columns=[target_col, "customer_id"])

    # Split train and test before any processing to prevent leakage
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw_df, y_raw, test_size=0.25, random_state=random_state, stratify=y_raw
    )

    # -------------------------------------------------------------
    # 1. BASELINE PIPELINE (Naive imputation, no proper scaling/balancing)
    # -------------------------------------------------------------
    X_train_base = X_train_raw.copy()
    X_test_base = X_test_raw.copy()

    # Naively convert or factorize all columns to numeric for baseline model
    for c in X_train_base.columns:
        # If it's total_charges or string-formatted numeric, try naive to_numeric first
        converted_train = pd.to_numeric(X_train_base[c].astype(str).str.extract(r'(\d+\.?\d*)', expand=False), errors="coerce")
        converted_test = pd.to_numeric(X_test_base[c].astype(str).str.extract(r'(\d+\.?\d*)', expand=False), errors="coerce")
        
        if converted_train.notna().sum() > 0.5 * len(X_train_base):
            # Numeric column with noise
            X_train_base[c] = converted_train.fillna(0)
            X_test_base[c] = converted_test.fillna(0)
        else:
            # Pure categorical column: naive factorize
            codes_train, uniques = pd.factorize(X_train_base[c].astype(str).fillna("Missing"))
            X_train_base[c] = codes_train
            # Map test using training categories
            cat_map = {val: i for i, val in enumerate(uniques)}
            X_test_base[c] = X_test_base[c].astype(str).map(cat_map).fillna(-1)

    X_train_base = X_train_base.astype(float).fillna(0)
    X_test_base = X_test_base.astype(float).fillna(0)

    # Fit Baseline Model
    baseline_model = LogisticRegression(max_iter=1000, random_state=random_state)
    baseline_model.fit(X_train_base, y_train)
    y_pred_base = baseline_model.predict(X_test_base)
    y_proba_base = baseline_model.predict_proba(X_test_base)[:, 1]

    baseline_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_base),
        "Precision": precision_score(y_test, y_pred_base, zero_division=0),
        "Recall": recall_score(y_test, y_pred_base, zero_division=0),
        "F1-Score (Macro)": f1_score(y_test, y_pred_base, average="macro", zero_division=0),
        "F1-Score (Minority)": f1_score(y_test, y_pred_base, pos_label=1, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba_base),
        "y_true": y_test,
        "y_pred": y_pred_base,
        "y_proba": y_proba_base,
        "confusion_matrix": confusion_matrix(y_test, y_pred_base)
    }

    # -------------------------------------------------------------
    # 2. FULL PREPROCESSED PIPELINE
    # -------------------------------------------------------------
    # Step A: Clean dirty values
    X_train_clean = clean_raw_tabular_data(X_train_raw)
    X_test_clean = clean_raw_tabular_data(X_test_raw)

    # Step B: Categorical Encoding
    nominal_cols = ["payment_method", "internet_service", "tech_support"]
    ordinal_cols = ["contract_type"]
    ordinal_order = {"contract_type": ["Month-to-month", "One year", "Two year"]}

    encoder = TabularEncoder(
        nominal_cols=nominal_cols,
        ordinal_cols=ordinal_cols,
        ordinal_categories=ordinal_order
    )
    X_train_encoded = encoder.fit_transform(X_train_clean)
    X_test_encoded = encoder.transform(X_test_clean)

    # Step C: Feature Scaling
    scaler = TabularScaler(method="robust")
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    # Step D: Handle Class Imbalance (SMOTE on training set only!)
    X_train_balanced, y_train_balanced = balance_dataset(
        X_train_scaled, y_train, method="smote", random_state=random_state
    )

    # Step E: Model Training
    preprocessed_model = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=6)
    preprocessed_model.fit(X_train_balanced, y_train_balanced)
    
    y_pred_proc = preprocessed_model.predict(X_test_scaled)
    y_proba_proc = preprocessed_model.predict_proba(X_test_scaled)[:, 1]

    preprocessed_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_proc),
        "Precision": precision_score(y_test, y_pred_proc, zero_division=0),
        "Recall": recall_score(y_test, y_pred_proc, zero_division=0),
        "F1-Score (Macro)": f1_score(y_test, y_pred_proc, average="macro", zero_division=0),
        "F1-Score (Minority)": f1_score(y_test, y_pred_proc, pos_label=1, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba_proc),
        "y_true": y_test,
        "y_pred": y_pred_proc,
        "y_proba": y_proba_proc,
        "confusion_matrix": confusion_matrix(y_test, y_pred_proc)
    }

    # Save cleaned processed dataframe to disk
    processed_full_df = pd.concat([X_train_scaled, X_test_scaled], axis=0).reset_index(drop=True)
    processed_full_df["churn"] = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
    processed_full_df.to_csv("data/tabular/customer_churn_processed.csv", index=False)

    return {
        "baseline_metrics": baseline_metrics,
        "preprocessed_metrics": preprocessed_metrics,
        "feature_names": encoder.feature_names_out_,
        "feature_importances": preprocessed_model.feature_importances_,
        "preprocessed_model": preprocessed_model,
        "baseline_model": baseline_model
    }
