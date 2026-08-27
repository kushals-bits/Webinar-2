"""
Evaluation Package for Webinar 2: Data Preprocessing.
Covers: Metrics calculation, Before vs After Comparison, and Visualization.
"""

from .metrics import compute_classification_metrics, format_metrics_table
from .comparison import generate_modality_comparison, build_master_comparison_report, print_master_comparison
from .visualizer import (
    plot_before_vs_after_benchmarks,
    plot_tabular_scaling_and_outliers,
    plot_image_pca_and_reconstruction,
    plot_confusion_matrices,
    generate_all_report_visuals
)

__all__ = [
    "compute_classification_metrics",
    "format_metrics_table",
    "generate_modality_comparison",
    "build_master_comparison_report",
    "print_master_comparison",
    "plot_before_vs_after_benchmarks",
    "plot_tabular_scaling_and_outliers",
    "plot_image_pca_and_reconstruction",
    "plot_confusion_matrices",
    "generate_all_report_visuals"
]
