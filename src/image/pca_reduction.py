"""
Image PCA & Modeling Pipeline — Webinar 2 (UPDATED)
Algorithms used (DIFFERENT from Webinar 1):
  - Webinar 1: Logistic Regression on pixel features
  - Webinar 2: SVM (RBF kernel) + KNeighborsClassifier on PCA-reduced features

Dataset: MNIST Handwritten Digits (real human handwriting — digits 0, 1, 2)
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

from .loading import load_image_dataset
from .resizing import batch_resize_images
from .normalization import images_to_numpy, normalize_minmax, flatten_images


class ImagePCA:
    """
    PCA transformer for image feature compression:
    - Fits PCA on flattened image vectors
    - Calculates cumulative explained variance
    - Reduces high-dimensional raw pixel space to compact representation
    - Supports inverse reconstruction back to image pixel space
    """
    def __init__(self, n_components=0.95, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.explained_variance_ratio_ = None
        self.cumulative_variance_ratio_ = None

    def fit(self, X: np.ndarray):
        self.pca.fit(X)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        return self.pca.inverse_transform(X_reduced)

    def get_variance_summary(self, top_k=10) -> pd.DataFrame:
        k = min(top_k, len(self.explained_variance_ratio_))
        return pd.DataFrame({
            "Principal Component": [f"PC_{i+1}" for i in range(k)],
            "Explained Variance": [round(float(v), 4) for v in self.explained_variance_ratio_[:k]],
            "Cumulative Variance": [round(float(v), 4) for v in self.cumulative_variance_ratio_[:k]]
        })


def run_image_pipeline(root_dir="data/image/raw", target_size=(28, 28), n_components=0.95, random_state=42):
    """
    MNIST Handwritten Digits (0, 1, 2) — Full Image Preprocessing pipeline with SVM & KNN.

    ALGORITHMS (Webinar 2 — different from Webinar 1):
    ─────────────────────────────────────────────────────────────────────────────────
    Baseline:     Raw pixel features (28×28 = 784 dims, unscaled)
                  → LogisticRegression (100 iterations only, struggles on raw pixels)
    Preprocessed: Normalized + PCA-Reduced features (≤30 dims, 95% variance)
                  → SVM with RBF Kernel (support vector classification)
                  → K-Nearest Neighbors (KNN, k=5) on PCA features
    ─────────────────────────────────────────────────────────────────────────────────

    Steps:
    1. Load MNIST digit images (real handwritten digits 0, 1, 2)
    2. Standardize resolution (letterboxing to 28×28)
    3. Normalize pixels [0, 255] → [0.0, 1.0]
    4. PCA to retain 95% variance → drastically reduce dimensions
    5. SVM RBF vs KNN vs Baseline Logistic Regression
    """
    raw_images, labels, file_paths, class_to_idx = load_image_dataset(root_dir)
    y_encoded = np.array([class_to_idx[lbl] for lbl in labels])

    indices = np.arange(len(raw_images))
    train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=random_state, stratify=y_encoded)

    train_images = [raw_images[i] for i in train_idx]
    test_images = [raw_images[i] for i in test_idx]
    y_train = y_encoded[train_idx]
    y_test = y_encoded[test_idx]

    # ─────────────────────────────────────────
    # BASELINE: Raw unscaled pixels → Logistic Regression
    # ─────────────────────────────────────────
    train_base = batch_resize_images(train_images, target_size=target_size, preserve_aspect=False)
    test_base = batch_resize_images(test_images, target_size=target_size, preserve_aspect=False)
    X_train_raw_arr = flatten_images(images_to_numpy(train_base))  # raw [0, 255]
    X_test_raw_arr = flatten_images(images_to_numpy(test_base))

    base_clf = LogisticRegression(max_iter=200, random_state=random_state, solver="saga")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        base_clf.fit(X_train_raw_arr, y_train)
    y_pred_base = base_clf.predict(X_test_raw_arr)

    baseline_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_base),
        "Precision": precision_score(y_test, y_pred_base, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred_base, average="macro", zero_division=0),
        "F1-Score (Macro)": f1_score(y_test, y_pred_base, average="macro", zero_division=0),
        "Feature Dimensions": X_train_raw_arr.shape[1],
        "confusion_matrix": confusion_matrix(y_test, y_pred_base),
        "model_name": "Logistic Regression (Baseline, Raw Pixels)"
    }

    # ─────────────────────────────────────────
    # PREPROCESSED: Normalized + PCA → SVM (RBF) + KNN
    # ─────────────────────────────────────────
    train_proc = batch_resize_images(train_images, target_size=target_size, preserve_aspect=True)
    test_proc = batch_resize_images(test_images, target_size=target_size, preserve_aspect=True)

    X_train_norm = normalize_minmax(images_to_numpy(train_proc))
    X_test_norm = normalize_minmax(images_to_numpy(test_proc))
    X_train_flat = flatten_images(X_train_norm)
    X_test_flat = flatten_images(X_test_norm)

    # PCA
    image_pca = ImagePCA(n_components=n_components, random_state=random_state)
    X_train_pca = image_pca.fit_transform(X_train_flat)
    X_test_pca = image_pca.transform(X_test_flat)

    # Scale PCA features for SVM
    feat_scaler = StandardScaler()
    X_train_pca_sc = feat_scaler.fit_transform(X_train_pca)
    X_test_pca_sc = feat_scaler.transform(X_test_pca)

    # SVM with RBF kernel
    svm_clf = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=random_state)
    svm_clf.fit(X_train_pca_sc, y_train)
    y_pred_svm = svm_clf.predict(X_test_pca_sc)

    # KNN (k=5) on PCA features
    knn_clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn_clf.fit(X_train_pca, y_train)
    y_pred_knn = knn_clf.predict(X_test_pca)

    # Use SVM as the "preprocessed" result (typically better on compact PCA features)
    preprocessed_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_svm),
        "Precision": precision_score(y_test, y_pred_svm, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred_svm, average="macro", zero_division=0),
        "F1-Score (Macro)": f1_score(y_test, y_pred_svm, average="macro", zero_division=0),
        "Feature Dimensions": X_train_pca_sc.shape[1],
        "Dimensionality Reduction": f"{(1 - X_train_pca_sc.shape[1]/X_train_raw_arr.shape[1])*100:.1f}%",
        "confusion_matrix": confusion_matrix(y_test, y_pred_svm),
        "model_name": "SVM RBF Kernel (Preprocessed, PCA)"
    }

    knn_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_knn),
        "F1-Score (Macro)": f1_score(y_test, y_pred_knn, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_knn),
        "model_name": "KNN (k=5, PCA Features)"
    }

    print(f"  Image | Baseline: Logistic Regression (raw {X_train_raw_arr.shape[1]} dims) "
          f"-> Preprocessed: SVM RBF ({X_train_pca_sc.shape[1]} PCA dims)")

    proc_dir = os.path.join(os.path.dirname(os.path.normpath(root_dir)), "processed") if root_dir else "data/image/processed"
    os.makedirs(proc_dir, exist_ok=True)
    np.save(os.path.join(proc_dir, "image_pca_features.npy"), X_train_pca)
    np.save(os.path.join(proc_dir, "image_labels.npy"), y_train)

    return {
        "baseline_metrics": baseline_metrics,
        "preprocessed_metrics": preprocessed_metrics,
        "knn_metrics": knn_metrics,
        "image_pca": image_pca,
        "classes": list(class_to_idx.keys()),
        "sample_original": X_test_flat[0],
        "sample_reconstruction": image_pca.inverse_transform(X_test_pca[0:1])[0],
        "target_size": target_size
    }
