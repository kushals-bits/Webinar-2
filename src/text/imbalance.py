"""
Text NLP Pipeline Module — Webinar 2 (UPDATED)
Algorithms used (DIFFERENT from Webinar 1):
  - Webinar 1: Basic Logistic Regression on tokenized text
  - Webinar 2: LinearSVC + Complement Naive Bayes + SGD Classifier

Dataset: 20 Newsgroups (real internet forum posts — sci.med vs alt.atheism)
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

from .cleaning import batch_clean_texts
from .vectorization import TFIDFProcessor
from .feature_engineering import extract_numerical_text_features
from ..tabular.imbalance import balance_dataset


def run_text_pipeline(csv_path="data/text/newsgroups_raw.csv",
                      text_col="raw_text", target_col="target", random_state=42):
    """
    20 Newsgroups (sci.med vs alt.atheism) — Full NLP pipeline with SVM/NB algorithms.

    ALGORITHMS (Webinar 2 — different from Webinar 1's Logistic Regression):
    ─────────────────────────────────────────────────────────────────────────────────
    Baseline:     Bag-of-Words (raw text, uncleaned) + CountVectorizer
                  → SGD Classifier (Stochastic Gradient Descent)
    Preprocessed: Cleaned text + TF-IDF (1,2)-grams
                  → LinearSVC (Support Vector Machine with linear kernel)
                     calibrated with Platt scaling for probability output
    ─────────────────────────────────────────────────────────────────────────────────

    Steps:
    1. Load real 20 Newsgroups corpus (internet forum posts, raw noise)
    2. Baseline: raw CountVectorizer → SGD Classifier
    3. Preprocessed: Clean → TF-IDF bigrams → Numerical features → SMOTE → LinearSVC
    """
    df = pd.read_csv(csv_path)
    texts = df[text_col].fillna("")
    y = df[target_col]

    # Map target names for display
    classes = sorted(y.unique())
    class_names = ["sci.med", "alt.atheism"] if set(classes) == {0, 1} else [str(c) for c in classes]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, y, test_size=0.25, random_state=random_state, stratify=y
    )

    # ─────────────────────────────────────────
    # BASELINE: Raw BoW → SGD Classifier (imbalanced, uncleaned)
    # ─────────────────────────────────────────
    base_cv = CountVectorizer(max_features=500, lowercase=False, strip_accents=None)
    X_train_bow = base_cv.fit_transform(X_train_raw).toarray()
    X_test_bow = base_cv.transform(X_test_raw).toarray()

    # SGDClassifier with hinge loss ≡ linear SVM, but different to Webinar 1's LogReg
    base_model = SGDClassifier(loss="modified_huber", max_iter=1000, tol=1e-3, random_state=random_state, n_jobs=-1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        base_model.fit(X_train_bow, y_train)
    y_pred_base = base_model.predict(X_test_bow)
    y_proba_base = base_model.predict_proba(X_test_bow)[:, 1]

    baseline_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_base),
        "Precision": precision_score(y_test, y_pred_base, zero_division=0),
        "Recall": recall_score(y_test, y_pred_base, zero_division=0),
        "F1-Score (Macro)": f1_score(y_test, y_pred_base, average="macro", zero_division=0),
        "F1-Score (Minority)": f1_score(y_test, y_pred_base, pos_label=classes[-1], zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba_base),
        "y_true": y_test, "y_pred": y_pred_base, "y_proba": y_proba_base,
        "confusion_matrix": confusion_matrix(y_test, y_pred_base),
        "model_name": "SGD Classifier (Baseline, Raw BoW)"
    }

    # ─────────────────────────────────────────
    # PREPROCESSED: Cleaned TF-IDF → LinearSVC (calibrated)
    # ─────────────────────────────────────────
    # Step A: Text Cleaning
    clean_train = batch_clean_texts(X_train_raw)
    clean_test = batch_clean_texts(X_test_raw)

    # Step B: TF-IDF with unigrams + bigrams, sublinear TF
    tfidf = TFIDFProcessor(max_features=500, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(clean_train)
    X_test_tfidf = tfidf.transform(clean_test)

    # Step C: Numerical Linguistic Features (scaled)
    num_train = extract_numerical_text_features(X_train_raw)
    num_test = extract_numerical_text_features(X_test_raw)
    num_scaler = RobustScaler()
    X_train_num = num_scaler.fit_transform(num_train)
    X_test_num = num_scaler.transform(num_test)

    # Step D: Concatenate TF-IDF + numerical
    X_train_combined = np.hstack([X_train_tfidf, X_train_num])
    X_test_combined = np.hstack([X_test_tfidf, X_test_num])

    # Step E: SMOTE balancing
    X_train_bal, y_train_bal = balance_dataset(
        pd.DataFrame(X_train_combined), y_train, method="smote", random_state=random_state
    )

    # Step F: LinearSVC (calibrated with Platt scaling for predict_proba)
    svc = LinearSVC(C=1.0, max_iter=2000, random_state=random_state)
    proc_model = CalibratedClassifierCV(svc, cv=3)
    proc_model.fit(X_train_bal, y_train_bal)

    y_pred_proc = proc_model.predict(X_test_combined)
    y_proba_proc = proc_model.predict_proba(X_test_combined)[:, 1]

    preprocessed_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_proc),
        "Precision": precision_score(y_test, y_pred_proc, zero_division=0),
        "Recall": recall_score(y_test, y_pred_proc, zero_division=0),
        "F1-Score (Macro)": f1_score(y_test, y_pred_proc, average="macro", zero_division=0),
        "F1-Score (Minority)": f1_score(y_test, y_pred_proc, pos_label=classes[-1], zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba_proc),
        "y_true": y_test, "y_pred": y_pred_proc, "y_proba": y_proba_proc,
        "confusion_matrix": confusion_matrix(y_test, y_pred_proc),
        "model_name": "LinearSVC (Calibrated, Cleaned TF-IDF)"
    }

    print(f"  Text | Baseline: SGD Classifier -> Preprocessed: LinearSVC (Calibrated)")
    print(f"  Top TF-IDF keywords: {', '.join(tfidf.get_top_keywords(clean_train, top_n=5)['term'].tolist())}")

    # Save cleaned output
    out_dir = os.path.dirname(os.path.normpath(csv_path)) if csv_path and os.path.dirname(os.path.normpath(csv_path)) else "data/text"
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({
        "doc_id": df["doc_id"],
        "raw_text": texts,
        "cleaned_text": batch_clean_texts(texts),
        "category": df.get("category", y)
    }).to_csv(os.path.join(out_dir, "newsgroups_clean.csv"), index=False)

    return {
        "baseline_metrics": baseline_metrics,
        "preprocessed_metrics": preprocessed_metrics,
        "tfidf_top_keywords": tfidf.get_top_keywords(clean_train, top_n=10),
        "preprocessed_model": proc_model,
        "baseline_model": base_model,
        "class_names": class_names
    }
