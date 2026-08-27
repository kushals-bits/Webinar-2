"""
Evaluation Metrics Module
Calculates standard and robust performance metrics across classification tasks.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def compute_classification_metrics(y_true, y_pred, y_proba=None, pos_label=1) -> dict:
    """Computes a comprehensive dictionary of classification evaluation metrics."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision (Macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (Macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-Score (Macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-Score (Weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }

    # Binary specific minority metrics if applicable
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2 and pos_label in unique_classes:
        metrics["Precision (Minority)"] = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        metrics["Recall (Minority)"] = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        metrics["F1-Score (Minority)"] = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)

    if y_proba is not None:
        try:
            if len(unique_classes) == 2:
                metrics["ROC-AUC"] = roc_auc_score(y_true, y_proba)
            else:
                metrics["ROC-AUC (OVR)"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
        except Exception:
            metrics["ROC-AUC"] = None

    return metrics


def format_metrics_table(metrics_dict: dict) -> pd.DataFrame:
    """Formats metrics dictionary into a clean 2-column DataFrame."""
    rows = []
    for k, v in metrics_dict.items():
        if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
            rows.append({"Metric": k, "Value": f"{v:.4f}"})
    return pd.DataFrame(rows)
