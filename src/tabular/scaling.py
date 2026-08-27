"""
Tabular Feature Scaling Module
Implements and compares StandardScaler, MinMaxScaler, and RobustScaler.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def compare_scalers(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    """
    Compares the effect of StandardScaler, MinMaxScaler, and RobustScaler on numerical features:
    Shows original vs transformed mean, std, min, median, max, IQR.
    """
    data = df[num_cols].dropna()
    results = []

    scalers = {
        "Raw (No Scaling)": None,
        "StandardScaler (Z-Score)": StandardScaler(),
        "MinMaxScaler ([0, 1])": MinMaxScaler(),
        "RobustScaler (IQR-based)": RobustScaler()
    }

    for name, scaler in scalers.items():
        if scaler is None:
            transformed = data.values
        else:
            transformed = scaler.fit_transform(data)

        for i, col in enumerate(num_cols):
            vals = transformed[:, i]
            q25, q75 = np.percentile(vals, [25, 75])
            results.append({
                "Scaler": name,
                "Feature": col,
                "Mean": round(float(np.mean(vals)), 4),
                "Std": round(float(np.std(vals)), 4),
                "Min": round(float(np.min(vals)), 4),
                "Median": round(float(np.median(vals)), 4),
                "Max": round(float(np.max(vals)), 4),
                "IQR": round(float(q75 - q25), 4)
            })

    return pd.DataFrame(results)


class TabularScaler:
    """
    Scaler wrapper supporting 'standard', 'minmax', and 'robust' methods.
    Preserves DataFrame column names and index.
    """
    def __init__(self, method="robust", columns=None):
        self.method = method.lower()
        self.columns = columns
        if self.method == "standard":
            self.scaler = StandardScaler()
        elif self.method == "minmax":
            self.scaler = MinMaxScaler()
        elif self.method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method '{method}'. Choose from 'standard', 'minmax', 'robust'.")

    def fit(self, X: pd.DataFrame, y=None):
        cols_to_scale = self.columns or X.select_dtypes(include=[np.number]).columns.tolist()
        self.target_cols_ = cols_to_scale
        self.scaler.fit(X[cols_to_scale])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        scaled_arr = self.scaler.transform(X[self.target_cols_])
        X_out[self.target_cols_] = scaled_arr
        return X_out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
