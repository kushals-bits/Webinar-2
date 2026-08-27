"""
Before vs After Benchmark Comparison Engine
Computes performance deltas and formats cross-pipeline benchmark tables.
"""

import pandas as pd


def generate_modality_comparison(baseline_metrics: dict, preprocessed_metrics: dict, modality_name: str) -> pd.DataFrame:
    """
    Generates a Before vs After comparison table for a single modality:
    Metric | Baseline (Before) | Preprocessed (After) | Delta (Change) | % Improvement
    """
    common_metrics = [
        k for k in baseline_metrics.keys()
        if k in preprocessed_metrics and isinstance(baseline_metrics[k], (int, float)) and not isinstance(baseline_metrics[k], bool)
    ]

    rows = []
    for metric in common_metrics:
        before = baseline_metrics[metric]
        after = preprocessed_metrics[metric]
        delta = after - before
        pct_change = (delta / before * 100) if before != 0 else 0.0

        rows.append({
            "Modality": modality_name,
            "Metric": metric,
            "Baseline (Before)": round(before, 4),
            "Preprocessed (After)": round(after, 4),
            "Delta": f"{delta:+.4f}",
            "Relative Improvement": f"{pct_change:+.1f}%"
        })

    return pd.DataFrame(rows)


def build_master_comparison_report(tabular_res: dict, text_res: dict, image_res: dict) -> pd.DataFrame:
    """Combines Tabular, Text, and Image Before vs After results into a unified summary table."""
    tab_df = generate_modality_comparison(
        tabular_res["baseline_metrics"], tabular_res["preprocessed_metrics"], "1. Tabular"
    )
    text_df = generate_modality_comparison(
        text_res["baseline_metrics"], text_res["preprocessed_metrics"], "2. Text"
    )
    img_df = generate_modality_comparison(
        image_res["baseline_metrics"], image_res["preprocessed_metrics"], "3. Image"
    )

    return pd.concat([tab_df, text_df, img_df], axis=0).reset_index(drop=True)


def print_master_comparison(master_df: pd.DataFrame):
    """Prints a structured ASCII master comparison table."""
    print("\n" + "=" * 90)
    print(" " * 25 + "WEBINAR 2: BEFORE VS AFTER PREPROCESSING COMPARISON")
    print("=" * 90)
    
    for mod in master_df["Modality"].unique():
        sub = master_df[master_df["Modality"] == mod].drop(columns=["Modality"])
        print(f"\n>>> MODALITY: {mod}")
        print("-" * 75)
        print(sub.to_string(index=False))
        print("-" * 75)
    print("=" * 90 + "\n")
