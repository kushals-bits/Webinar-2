"""
Tabular Class Imbalance Handling Module
Implements SMOTE, Random Over-Sampling, Random Under-Sampling, and Cost-Sensitive Class Weighting.
"""

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


def calculate_class_weights(y) -> dict:
    """Computes balanced class weights inversely proportional to class frequencies."""
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


def balance_dataset(X: pd.DataFrame, y: pd.Series, method: str = "smote", random_state: int = 42):
    """
    Balances the feature matrix X and target y using specified technique:
    - 'smote': Synthetic Minority Over-sampling Technique
    - 'oversample': Random Over-sampling
    - 'undersample': Random Under-sampling
    - 'none': Returns original data unchanged
    """
    method = method.lower()
    if method == "none":
        return X, y

    if not IMBLEARN_AVAILABLE:
        print("Warning: imbalanced-learn not found. Using simple duplicate-based resampling fallback.")
        # Fallback implementation
        df_combined = pd.concat([X, y.rename("target")], axis=1)
        minority_class = y.value_counts().idxmin()
        majority_count = y.value_counts().max()
        df_majority = df_combined[df_combined["target"] != minority_class]
        df_minority = df_combined[df_combined["target"] == minority_class]
        
        if method in ["smote", "oversample"]:
            df_minority_resampled = df_minority.sample(majority_count, replace=True, random_state=random_state)
            balanced_df = pd.concat([df_majority, df_minority_resampled]).sample(frac=1.0, random_state=random_state)
        elif method == "undersample":
            minority_count = y.value_counts().min()
            df_majority_resampled = df_majority.sample(minority_count, replace=False, random_state=random_state)
            balanced_df = pd.concat([df_majority_resampled, df_minority]).sample(frac=1.0, random_state=random_state)
        else:
            balanced_df = df_combined

        X_res = balanced_df.drop(columns=["target"])
        y_res = balanced_df["target"]
        return X_res, y_res

    # Use imblearn
    feature_names = X.columns if isinstance(X, pd.DataFrame) else None
    
    if method == "smote":
        sampler = SMOTE(random_state=random_state)
    elif method == "oversample":
        sampler = RandomOverSampler(random_state=random_state)
    elif method == "undersample":
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError(f"Unknown balancing method '{method}'. Choose from 'smote', 'oversample', 'undersample', 'none'.")

    X_res, y_res = sampler.fit_resample(X, y)
    
    if feature_names is not None:
        X_res = pd.DataFrame(X_res, columns=feature_names)
    if isinstance(y, pd.Series):
        y_res = pd.Series(y_res, name=y.name)

    return X_res, y_res
