"""
Tabular Data Profiling Module
Comprehensive data health audit, missing value analysis, skewness, outliers, and distribution diagnostics.
"""

import numpy as np
import pandas as pd


def detect_outliers_iqr(series, threshold=1.5):
    """Detects outlier indices and bounds using the Interquartile Range (IQR) method."""
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return 0, None, None, pd.Series([False] * len(series), index=series.index)
    q25 = np.percentile(clean_series, 25)
    q75 = np.percentile(clean_series, 75)
    iqr = q75 - q25
    lower_bound = q25 - threshold * iqr
    upper_bound = q75 + threshold * iqr
    outlier_mask = (series < lower_bound) | (series > upper_bound)
    return outlier_mask.sum(), lower_bound, upper_bound, outlier_mask


def detect_outliers_zscore(series, threshold=3.0):
    """Detects outliers using standard Z-score method."""
    clean_series = series.dropna()
    if len(clean_series) == 0 or clean_series.std() == 0:
        return 0, pd.Series([False] * len(series), index=series.index)
    mean = clean_series.mean()
    std = clean_series.std()
    z_scores = np.abs((series - mean) / std)
    outlier_mask = z_scores > threshold
    return outlier_mask.sum(), outlier_mask


def profile_dataframe(df: pd.DataFrame, target_col: str = None) -> dict:
    """
    Performs full profiling of a tabular DataFrame:
    - Shape and memory usage
    - Data types
    - Missing value analysis (count, percentage)
    - Numerical statistics (mean, std, min, median, max, skewness, kurtosis)
    - Outlier counts (IQR & Z-score)
    - Categorical statistics (cardinality, top values, frequency)
    - Target distribution (if target_col specified)
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # 1. Column Overview & Missingness
    col_summary = []
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = (null_count / total_rows) * 100
        n_unique = df[col].nunique()
        dtype = str(df[col].dtype)
        
        entry = {
            "column": col,
            "dtype": dtype,
            "null_count": null_count,
            "null_pct": round(null_pct, 2),
            "unique_values": n_unique,
            "type_category": "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"
        }
        col_summary.append(entry)
    
    summary_df = pd.DataFrame(col_summary)

    # 2. Numerical Diagnostics (Outliers, Skew, Kurtosis)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_diagnostics = []
    for col in num_cols:
        series = df[col]
        iqr_count, lb, ub, _ = detect_outliers_iqr(series)
        z_count, _ = detect_outliers_zscore(series)
        skew = series.skew()
        kurt = series.kurtosis()
        
        num_diagnostics.append({
            "column": col,
            "mean": round(series.mean(), 2),
            "std": round(series.std(), 2),
            "min": round(series.min(), 2),
            "median": round(series.median(), 2),
            "max": round(series.max(), 2),
            "skewness": round(skew, 2) if not np.isnan(skew) else None,
            "kurtosis": round(kurt, 2) if not np.isnan(kurt) else None,
            "iqr_outliers": iqr_count,
            "iqr_lower": round(lb, 2) if lb is not None else None,
            "iqr_upper": round(ub, 2) if ub is not None else None,
            "zscore_outliers": z_count
        })
    num_df = pd.DataFrame(num_diagnostics)

    # 3. Categorical Diagnostics
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    cat_diagnostics = []
    for col in cat_cols:
        series = df[col].dropna()
        top_val = series.mode().iloc[0] if len(series) > 0 else None
        top_freq = series.value_counts().iloc[0] if len(series) > 0 else 0
        cat_diagnostics.append({
            "column": col,
            "cardinality": df[col].nunique(),
            "top_value": top_val,
            "top_frequency": top_freq,
            "top_percentage": round((top_freq / total_rows) * 100, 2) if total_rows > 0 else 0
        })
    cat_df = pd.DataFrame(cat_diagnostics)

    # 4. Target Class Imbalance Check
    target_info = None
    if target_col and target_col in df.columns:
        counts = df[target_col].value_counts(dropna=False)
        pcts = df[target_col].value_counts(normalize=True, dropna=False) * 100
        target_info = {
            "target_column": target_col,
            "class_counts": counts.to_dict(),
            "class_percentages": {k: round(v, 2) for k, v in pcts.items()},
            "imbalance_ratio": round(counts.max() / counts.min(), 2) if len(counts) > 1 and counts.min() > 0 else None
        }

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "column_summary": summary_df,
        "numerical_diagnostics": num_df,
        "categorical_diagnostics": cat_df,
        "target_info": target_info
    }


def print_profiling_report(profile: dict):
    """Prints a structured ASCII report of the profiling results."""
    print("=" * 80)
    print(f"TABULAR DATA HEALTH & PROFILING AUDIT REPORT")
    print(f"Total Records: {profile['total_rows']} | Total Features: {profile['total_cols']}")
    print("=" * 80)
    
    print("\n--- 1. Missing Values & Feature Types ---")
    print(profile["column_summary"].to_string(index=False))
    
    if not profile["numerical_diagnostics"].empty:
        print("\n--- 2. Numerical Features (Distribution, Skewness, Outliers) ---")
        print(profile["numerical_diagnostics"].to_string(index=False))
        
    if not profile["categorical_diagnostics"].empty:
        print("\n--- 3. Categorical Features (Cardinality & Distribution) ---")
        print(profile["categorical_diagnostics"].to_string(index=False))
        
    if profile["target_info"]:
        t = profile["target_info"]
        print("\n--- 4. Target Imbalance Analysis ---")
        print(f"Target Feature: '{t['target_column']}'")
        for cls, count in t["class_counts"].items():
            pct = t["class_percentages"].get(cls, 0)
            print(f"  Class '{cls}': {count} instances ({pct}%)")
        print(f"  Imbalance Ratio (Majority : Minority): {t['imbalance_ratio']}:1")
    print("=" * 80)
