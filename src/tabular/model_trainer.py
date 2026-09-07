"""
Tabular Model Training Module — Webinar 2 (UPDATED)
Algorithms used (DIFFERENT from Webinar 1):
  - Webinar 1: Logistic Regression, Random Forest
  - Webinar 2: XGBoost, LightGBM, Gradient Boosting (Ensemble Boosting methods)

Dataset: IBM Telco Customer Churn (Real 7,043 customer records)
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from sklearn.ensemble import GradientBoostingClassifier

from .encoding import clean_raw_tabular_data, TabularEncoder
from .scaling import TabularScaler
from .imbalance import balance_dataset


def _eval_metrics(y_test, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score (Macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "F1-Score (Minority)": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "y_true": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }


def _prepare_telco_data(csv_path):
    """Cleans and preps the IBM Telco Customer Churn CSV for encoding."""
    df = pd.read_csv(csv_path)

    # Drop customerID (identifier, not a feature)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges is stored as string (spaces for missing) — convert
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")

    # Target encoding
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])
    return X, y


def run_tabular_pipeline(csv_path="data/tabular/telco_churn_raw.csv", random_state=42):
    """
    IBM Telco Customer Churn — Full pipeline with Boosting algorithms.

    ALGORITHMS (Webinar 2 — different from Webinar 1's Logistic Regression + Random Forest):
    ─────────────────────────────────────────────────────────────────────────────────
    Baseline:     LogisticRegression         (naive, no preprocessing, imbalanced)
    Preprocessed: XGBoostClassifier          (if available)
                  LightGBM Classifier        (if available)
                  GradientBoostingClassifier (sklearn fallback)
    ─────────────────────────────────────────────────────────────────────────────────

    Steps:
    1. Download real IBM Telco dataset (7,043 records)
    2. Clean dirty strings (TotalCharges whitespace, binary Yes/No columns)
    3. One-Hot Encode nominals, Ordinal Encode Contract type
    4. RobustScaler on continuous numerical features
    5. SMOTE to address ~27% churn class imbalance
    6. Train XGBoost (or GBM fallback) vs Baseline Logistic Regression
    """
    # Load dataset
    if not pd.io.common.file_exists(csv_path):
        raise FileNotFoundError(
            f"Telco Churn dataset not found at {csv_path}. "
            "Run: python src/data_generators/download_real_datasets.py"
        )

    X_raw, y = _prepare_telco_data(csv_path)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.25, random_state=random_state, stratify=y
    )

    # ─────────────────────────────────────────
    # BASELINE: Logistic Regression (naive, no preprocessing)
    # ─────────────────────────────────────────
    X_train_b = X_train_raw.copy()
    X_test_b = X_test_raw.copy()

    for c in X_train_b.columns:
        conv_tr = pd.to_numeric(X_train_b[c].astype(str).str.extract(r"(\d+\.?\d*)", expand=False), errors="coerce")
        conv_te = pd.to_numeric(X_test_b[c].astype(str).str.extract(r"(\d+\.?\d*)", expand=False), errors="coerce")
        if conv_tr.notna().sum() > 0.5 * len(X_train_b):
            X_train_b[c] = conv_tr.fillna(0)
            X_test_b[c] = conv_te.fillna(0)
        else:
            codes, uniques = pd.factorize(X_train_b[c].astype(str).fillna("Missing"))
            X_train_b[c] = codes
            cat_map = {v: i for i, v in enumerate(uniques)}
            X_test_b[c] = X_test_b[c].astype(str).map(cat_map).fillna(-1)

    X_train_b = X_train_b.astype(float).fillna(0)
    X_test_b = X_test_b.astype(float).fillna(0)

    base_model = LogisticRegression(max_iter=1000, random_state=random_state)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        base_model.fit(X_train_b, y_train)
    y_pred_base = base_model.predict(X_test_b)
    y_proba_base = base_model.predict_proba(X_test_b)[:, 1]
    baseline_metrics = _eval_metrics(y_test, y_pred_base, y_proba_base)
    baseline_metrics["model_name"] = "Logistic Regression (Baseline)"

    # ─────────────────────────────────────────
    # PREPROCESSED: XGBoost / LightGBM / GBM (Boosting ensemble)
    # ─────────────────────────────────────────
    # Detect which binary yes/no cols to handle
    binary_yes_no = [c for c in X_train_raw.columns
                     if set(X_train_raw[c].dropna().unique()).issubset({"Yes", "No", "No internet service", "No phone service"})]
    nominal_cols = [c for c in X_train_raw.select_dtypes("object").columns
                    if c != "Contract" and c not in binary_yes_no]
    ordinal_cols = ["Contract"] if "Contract" in X_train_raw.columns else []
    ordinal_cats = {"Contract": ["Month-to-month", "One year", "Two year"]}

    # Map binary yes/no to 0/1 directly
    X_train_clean = X_train_raw.copy()
    X_test_clean = X_test_raw.copy()
    binary_map = {"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}
    for c in binary_yes_no:
        X_train_clean[c] = X_train_clean[c].map(binary_map).fillna(0)
        X_test_clean[c] = X_test_clean[c].map(binary_map).fillna(0)

    encoder = TabularEncoder(
        nominal_cols=nominal_cols,
        ordinal_cols=ordinal_cols,
        ordinal_categories=ordinal_cats
    )
    X_train_enc = encoder.fit_transform(X_train_clean)
    X_test_enc = encoder.transform(X_test_clean)

    scaler = TabularScaler(method="robust")
    X_train_sc = scaler.fit_transform(X_train_enc)
    X_test_sc = scaler.transform(X_test_enc)

    X_train_bal, y_train_bal = balance_dataset(X_train_sc, y_train, method="smote", random_state=random_state)

    # Select best available boosting classifier
    if XGBOOST_AVAILABLE:
        proc_model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=random_state, verbosity=0
        )
        model_name = "XGBoost Classifier"
    elif LIGHTGBM_AVAILABLE:
        proc_model = LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, verbosity=-1
        )
        model_name = "LightGBM Classifier"
    else:
        proc_model = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.8, random_state=random_state
        )
        model_name = "Gradient Boosting Classifier (sklearn)"

    proc_model.fit(X_train_bal, y_train_bal)
    y_pred_proc = proc_model.predict(X_test_sc)
    y_proba_proc = proc_model.predict_proba(X_test_sc)[:, 1]
    preprocessed_metrics = _eval_metrics(y_test, y_pred_proc, y_proba_proc)
    preprocessed_metrics["model_name"] = model_name

    print(f"  Tabular | Baseline: {baseline_metrics['model_name']} -> Preprocessed: {model_name}")

    # Save cleaned output
    out_dir = os.path.dirname(os.path.normpath(csv_path)) if csv_path and os.path.dirname(os.path.normpath(csv_path)) else "data/tabular"
    os.makedirs(out_dir, exist_ok=True)
    X_test_sc["Churn"] = y_test.values
    X_test_sc.to_csv(os.path.join(out_dir, "telco_churn_processed.csv"), index=False)

    # Feature importances
    feat_importances = None
    if hasattr(proc_model, "feature_importances_"):
        feat_importances = proc_model.feature_importances_

    return {
        "baseline_metrics": baseline_metrics,
        "preprocessed_metrics": preprocessed_metrics,
        "feature_names": encoder.feature_names_out_,
        "feature_importances": feat_importances,
        "preprocessed_model": proc_model,
        "baseline_model": base_model,
        "model_name": model_name
    }


import os
