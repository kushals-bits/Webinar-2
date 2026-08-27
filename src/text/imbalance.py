"""
Text Imbalance & Modeling Pipeline Module
Trains baseline text classifier vs fully preprocessed NLP pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from .cleaning import batch_clean_texts
from .vectorization import TFIDFProcessor
from .feature_engineering import extract_numerical_text_features
from ..tabular.imbalance import balance_dataset


def run_text_pipeline(csv_path="data/text/product_reviews_raw.csv", text_col="review_text", target_col="sentiment", random_state=42):
    """
    Executes the entire Text Preprocessing & Modeling Pipeline:
    1. Loads raw text data
    2. Builds Baseline Model:
       - Raw text directly passed to basic Bag-of-Words (CountVectorizer)
       - No HTML stripping, no lowercasing/stopword removal, no imbalance handling
       - Standard Logistic Regression
    3. Runs Full Preprocessed Pipeline:
       - Text Cleaning (HTML, URLs, contractions, stopwords, punctuation)
       - TF-IDF Vectorization with unigrams + bigrams and sublinear term frequency
       - Numerical Linguistic Feature Extraction & Scaling
       - Feature Concatenation (TF-IDF + Scaled Numerical Features)
       - Class Imbalance Handling with SMOTE
    4. Evaluates and returns metrics comparison.
    """
    df = pd.read_csv(csv_path)
    
    # Map target: 'Positive' -> 1, 'Negative' -> 0
    y = df[target_col].map({"Positive": 1, "Negative": 0})
    texts = df[text_col]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, y, test_size=0.25, random_state=random_state, stratify=y
    )

    # -------------------------------------------------------------
    # 1. BASELINE PIPELINE (Raw, uncleaned BoW, imbalanced)
    # -------------------------------------------------------------
    baseline_cv = CountVectorizer(lowercase=False, max_features=300)
    X_train_base = baseline_cv.fit_transform(X_train_raw).toarray()
    X_test_base = baseline_cv.transform(X_test_raw).toarray()

    base_model = LogisticRegression(max_iter=500, random_state=random_state)
    base_model.fit(X_train_base, y_train)
    y_pred_base = base_model.predict(X_test_base)
    y_proba_base = base_model.predict_proba(X_test_base)[:, 1]

    baseline_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_base),
        "Precision": precision_score(y_test, y_pred_base, zero_division=0),
        "Recall": recall_score(y_test, y_pred_base, zero_division=0),
        "F1-Score (Macro)": f1_score(y_test, y_pred_base, average="macro", zero_division=0),
        "F1-Score (Minority)": f1_score(y_test, y_pred_base, pos_label=0, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba_base),
        "y_true": y_test,
        "y_pred": y_pred_base,
        "y_proba": y_proba_base,
        "confusion_matrix": confusion_matrix(y_test, y_pred_base)
    }

    # -------------------------------------------------------------
    # 2. PREPROCESSED PIPELINE (Cleaned, TF-IDF + Numerical Feats, SMOTE)
    # -------------------------------------------------------------
    # Step A: Cleaning
    clean_train_texts = batch_clean_texts(X_train_raw)
    clean_test_texts = batch_clean_texts(X_test_raw)

    # Step B: TF-IDF
    tfidf = TFIDFProcessor(max_features=300, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(clean_train_texts)
    X_test_tfidf = tfidf.transform(clean_test_texts)

    # Step C: Numerical Feature Extraction & Scaling
    num_train_df = extract_numerical_text_features(X_train_raw)
    num_test_df = extract_numerical_text_features(X_test_raw)

    num_scaler = RobustScaler()
    X_train_num = num_scaler.fit_transform(num_train_df)
    X_test_num = num_scaler.transform(num_test_df)

    # Step D: Feature Concatenation
    X_train_combined = np.hstack([X_train_tfidf, X_train_num])
    X_test_combined = np.hstack([X_test_tfidf, X_test_num])

    # Step E: Handle Imbalance with SMOTE
    X_train_balanced, y_train_balanced = balance_dataset(
        pd.DataFrame(X_train_combined), y_train, method="smote", random_state=random_state
    )

    # Step F: Train Tuned Classifier
    proc_model = LogisticRegression(C=1.5, max_iter=1000, random_state=random_state)
    proc_model.fit(X_train_balanced, y_train_balanced)

    y_pred_proc = proc_model.predict(X_test_combined)
    y_proba_proc = proc_model.predict_proba(X_test_combined)[:, 1]

    preprocessed_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_proc),
        "Precision": precision_score(y_test, y_pred_proc, zero_division=0),
        "Recall": recall_score(y_test, y_pred_proc, zero_division=0),
        "F1-Score (Macro)": f1_score(y_test, y_pred_proc, average="macro", zero_division=0),
        "F1-Score (Minority)": f1_score(y_test, y_pred_proc, pos_label=0, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba_proc),
        "y_true": y_test,
        "y_pred": y_pred_proc,
        "y_proba": y_proba_proc,
        "confusion_matrix": confusion_matrix(y_test, y_pred_proc)
    }

    # Save cleaned text dataframe
    df_clean_out = pd.DataFrame({
        "review_id": df["review_id"],
        "raw_text": df[text_col],
        "cleaned_text": batch_clean_texts(df[text_col]),
        "sentiment": df[target_col]
    })
    df_clean_out.to_csv("data/text/product_reviews_clean.csv", index=False)

    return {
        "baseline_metrics": baseline_metrics,
        "preprocessed_metrics": preprocessed_metrics,
        "tfidf_top_keywords": tfidf.get_top_keywords(clean_train_texts, top_n=10),
        "preprocessed_model": proc_model,
        "baseline_model": base_model
    }
