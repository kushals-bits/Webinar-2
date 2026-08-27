"""
Image PCA Dimensionality Reduction & Modeling Module
Applies PCA to flatten normalized image features, analyzes variance, and benchmarks classification.
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from .loading import load_image_dataset
from .resizing import batch_resize_images
from .normalization import images_to_numpy, normalize_minmax, flatten_images


class ImagePCA:
    """
    PCA transformer for image feature compression:
    - Fits PCA on flattened image vectors
    - Calculates cumulative explained variance
    - Reduces high-dimensional raw pixel space (e.g. 64x64x3 = 12,288 dims) to compact representation (e.g. 30-50 dims)
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
        """Reconstructs flattened image vectors from PCA reduced space."""
        return self.pca.inverse_transform(X_reduced)

    def get_variance_summary(self, top_k=10) -> pd.DataFrame:
        """Returns summary table of top principal components variance."""
        k = min(top_k, len(self.explained_variance_ratio_))
        return pd.DataFrame({
            "Principal Component": [f"PC_{i+1}" for i in range(k)],
            "Explained Variance": [round(float(v), 4) for v in self.explained_variance_ratio_[:k]],
            "Cumulative Variance": [round(float(v), 4) for v in self.cumulative_variance_ratio_[:k]]
        })


def run_image_pipeline(root_dir="data/image/raw", target_size=(64, 64), n_components=0.95, random_state=42):
    """
    Executes the entire Image Preprocessing & Modeling Pipeline:
    1. Loads multi-class images (arbitrary raw resolutions)
    2. Builds Baseline Model:
       - Naive direct resize without aspect preservation, raw [0, 255] pixels, all 12,288 dimensions
       - Basic classifier
    3. Runs Full Preprocessed Pipeline:
       - Aspect-ratio preserving letterboxing resize
       - Min-Max pixel intensity normalization [0.0, 1.0]
       - PCA Dimensionality Reduction (retaining specified variance threshold)
       - Trains Classifier on Reduced Features
    4. Evaluates and compares performance, dimensions, and execution efficiency.
    """
    raw_images, labels, file_paths, class_to_idx = load_image_dataset(root_dir)
    y_encoded = np.array([class_to_idx[lbl] for lbl in labels])

    # Split indices
    indices = np.arange(len(raw_images))
    train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=random_state, stratify=y_encoded)

    train_images = [raw_images[i] for i in train_idx]
    test_images = [raw_images[i] for i in test_idx]
    y_train = y_encoded[train_idx]
    y_test = y_encoded[test_idx]

    # -------------------------------------------------------------
    # 1. BASELINE PIPELINE (Raw unscaled direct resize, all pixels)
    # -------------------------------------------------------------
    train_resized_base = batch_resize_images(train_images, target_size=target_size, preserve_aspect=False)
    test_resized_base = batch_resize_images(test_images, target_size=target_size, preserve_aspect=False)

    X_train_raw_arr = flatten_images(images_to_numpy(train_resized_base))  # Raw [0, 255]
    X_test_raw_arr = flatten_images(images_to_numpy(test_resized_base))

    base_clf = LogisticRegression(max_iter=100, random_state=random_state)
    base_clf.fit(X_train_raw_arr, y_train)
    y_pred_base = base_clf.predict(X_test_raw_arr)

    baseline_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_base),
        "Precision": precision_score(y_test, y_pred_base, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred_base, average="macro", zero_division=0),
        "F1-Score (Macro)": f1_score(y_test, y_pred_base, average="macro", zero_division=0),
        "Feature Dimensions": X_train_raw_arr.shape[1],
        "confusion_matrix": confusion_matrix(y_test, y_pred_base)
    }

    # -------------------------------------------------------------
    # 2. PREPROCESSED PIPELINE (Aspect-preserving resize, [0,1] normalization, PCA)
    # -------------------------------------------------------------
    train_resized_proc = batch_resize_images(train_images, target_size=target_size, preserve_aspect=True)
    test_resized_proc = batch_resize_images(test_images, target_size=target_size, preserve_aspect=True)

    X_train_norm = normalize_minmax(images_to_numpy(train_resized_proc))
    X_test_norm = normalize_minmax(images_to_numpy(test_resized_proc))

    X_train_flat = flatten_images(X_train_norm)
    X_test_flat = flatten_images(X_test_norm)

    # Fit PCA
    image_pca = ImagePCA(n_components=n_components, random_state=random_state)
    X_train_pca = image_pca.fit_transform(X_train_flat)
    X_test_pca = image_pca.transform(X_test_flat)

    # Train Classifier on Reduced Features
    proc_clf = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=6)
    proc_clf.fit(X_train_pca, y_train)
    y_pred_proc = proc_clf.predict(X_test_pca)

    preprocessed_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_proc),
        "Precision": precision_score(y_test, y_pred_proc, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred_proc, average="macro", zero_division=0),
        "F1-Score (Macro)": f1_score(y_test, y_pred_proc, average="macro", zero_division=0),
        "Feature Dimensions": X_train_pca.shape[1],
        "Dimensionality Reduction": f"{(1 - (X_train_pca.shape[1] / X_train_raw_arr.shape[1])) * 100:.1f}%",
        "confusion_matrix": confusion_matrix(y_test, y_pred_proc)
    }

    # Save processed PCA features to processed directory
    os.makedirs("data/image/processed", exist_ok=True)
    np.save("data/image/processed/image_pca_features.npy", X_train_pca)
    np.save("data/image/processed/image_labels.npy", y_train)

    return {
        "baseline_metrics": baseline_metrics,
        "preprocessed_metrics": preprocessed_metrics,
        "image_pca": image_pca,
        "classes": list(class_to_idx.keys()),
        "sample_original": X_test_flat[0],
        "sample_reconstruction": image_pca.inverse_transform(X_test_pca[0:1])[0],
        "target_size": target_size
    }
