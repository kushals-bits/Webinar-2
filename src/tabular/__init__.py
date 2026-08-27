"""
Tabular Preprocessing Package for Webinar 2: Data Preprocessing.
Covers: Profiling, Encoding, Scaling, Imbalance handling, and Model Training.
"""

from .profiling import profile_dataframe, print_profiling_report, detect_outliers_iqr, detect_outliers_zscore
from .encoding import clean_raw_tabular_data, TabularEncoder
from .scaling import TabularScaler, compare_scalers
from .imbalance import balance_dataset, calculate_class_weights
from .model_trainer import run_tabular_pipeline

__all__ = [
    "profile_dataframe",
    "print_profiling_report",
    "detect_outliers_iqr",
    "detect_outliers_zscore",
    "clean_raw_tabular_data",
    "TabularEncoder",
    "TabularScaler",
    "compare_scalers",
    "balance_dataset",
    "calculate_class_weights",
    "run_tabular_pipeline"
]
