"""
Visualizer Module for Webinar 2: Data Preprocessing
Generates publication-quality charts and plots for Before vs After comparisons across modalities.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# Configure aesthetic plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True
})


def plot_before_vs_after_benchmarks(master_df: pd.DataFrame, output_path="reports/before_vs_after_benchmarks.png"):
    """
    Plots grouped bar charts comparing Baseline (Before) vs Preprocessed (After) metrics
    for Tabular, Text, and Image modalities side-by-side.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    modalities = master_df["Modality"].unique()
    n_mods = len(modalities)

    fig, axes = plt.subplots(1, n_mods, figsize=(6 * n_mods, 5), sharey=False)
    if n_mods == 1:
        axes = [axes]

    colors = ["#e74c3c", "#2ecc71"]  # Red for Before, Green for After

    for ax, mod in zip(axes, modalities):
        sub = master_df[master_df["Modality"] == mod]
        # Filter for standard percentage metrics
        plot_metrics = sub[sub["Metric"].isin(["Accuracy", "Precision", "Recall", "F1-Score (Macro)", "ROC-AUC", "F1-Score (Minority)"])]
        if plot_metrics.empty:
            plot_metrics = sub

        x = np.arange(len(plot_metrics))
        width = 0.35

        rects1 = ax.bar(x - width/2, plot_metrics["Baseline (Before)"], width, label="Baseline (Before)", color=colors[0], alpha=0.85)
        rects2 = ax.bar(x + width/2, plot_metrics["Preprocessed (After)"], width, label="Preprocessed (After)", color=colors[1], alpha=0.85)

        ax.set_title(f"{mod} Performance", fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_metrics["Metric"], rotation=25, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score (0.0 - 1.0)")
        ax.legend(loc="upper left")

        # Add values on top of bars
        for rect in rects1:
            h = rect.get_height()
            if not np.isnan(h):
                ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for rect in rects2:
            h = rect.get_height()
            if not np.isnan(h):
                ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.suptitle("Webinar 2: End-to-End Pipeline Evaluation (Before vs After Preprocessing)", fontweight="bold", y=1.03)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved benchmark chart to: {output_path}")


def plot_tabular_scaling_and_outliers(df_raw: pd.DataFrame, num_col="monthly_charges", output_path="reports/tabular_scaling_and_outliers.png"):
    """
    Plots distribution comparison across StandardScaler, MinMaxScaler, and RobustScaler,
    illustrating how RobustScaler handles extreme outliers effectively.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clean_series = df_raw[num_col].dropna().values.reshape(-1, 1)

    std_scaled = StandardScaler().fit_transform(clean_series).flatten()
    minmax_scaled = MinMaxScaler().fit_transform(clean_series).flatten()
    robust_scaled = RobustScaler().fit_transform(clean_series).flatten()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Raw Distribution
    sns.histplot(clean_series.flatten(), ax=axes[0, 0], kde=True, color="#34495e", bins=30)
    axes[0, 0].set_title(f"1. Raw '{num_col}' (With Outliers)", fontweight="bold")
    axes[0, 0].set_xlabel("Raw Value")

    # 2. StandardScaler
    sns.histplot(std_scaled, ax=axes[0, 1], kde=True, color="#3498db", bins=30)
    axes[0, 1].set_title("2. StandardScaler (Z-Score: Distorted by Outliers)", fontweight="bold")
    axes[0, 1].set_xlabel("Scaled Value (Mean=0, Std=1)")

    # 3. MinMaxScaler
    sns.histplot(minmax_scaled, ax=axes[1, 0], kde=True, color="#e67e22", bins=30)
    axes[1, 0].set_title("3. MinMaxScaler ([0, 1]: Compressed by Max Outlier)", fontweight="bold")
    axes[1, 0].set_xlabel("Scaled Value ([0, 1])")

    # 4. RobustScaler
    sns.histplot(robust_scaled, ax=axes[1, 1], kde=True, color="#27ae60", bins=30)
    axes[1, 1].set_title("4. RobustScaler (Median/IQR: Immune to Outliers)", fontweight="bold")
    axes[1, 1].set_xlabel("Scaled Value (Median=0, IQR=1)")

    plt.suptitle(f"Feature Scaling Comparison on '{num_col}'", fontweight="bold", y=1.02)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved scaling comparison plot to: {output_path}")


def plot_image_pca_and_reconstruction(image_res: dict, output_path="reports/image_pca_and_reconstruction.png"):
    """
    Plots:
    1. PCA Cumulative Explained Variance (Scree Plot)
    2. Side-by-side visualization of Original Normalized Image vs PCA Reconstructed Image
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pca_obj = image_res["image_pca"]
    cum_var = pca_obj.cumulative_variance_ratio_
    target_size = image_res["target_size"]
    h, w = target_size

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Subplot 1: Scree Plot
    axes[0].plot(range(1, len(cum_var) + 1), cum_var * 100, marker="o", markersize=3, color="#2980b9", lw=2)
    axes[0].axhline(y=95, color="#e74c3c", linestyle="--", label="95% Variance Threshold")
    axes[0].set_title("PCA Cumulative Explained Variance", fontweight="bold")
    axes[0].set_xlabel("Number of Principal Components")
    axes[0].set_ylabel("Cumulative Variance (%)")
    axes[0].set_ylim(0, 105)
    axes[0].legend(loc="lower right")

    # Subplot 2: Original Image
    orig_img_arr = np.clip(image_res["sample_original"].reshape(h, w, 3), 0.0, 1.0)
    axes[1].imshow(orig_img_arr)
    axes[1].set_title(f"Original Normalized Image\n({h}x{w}x3 = {h*w*3} dims)", fontweight="bold")
    axes[1].axis("off")

    # Subplot 3: Reconstructed Image
    recon_img_arr = np.clip(image_res["sample_reconstruction"].reshape(h, w, 3), 0.0, 1.0)
    axes[2].imshow(recon_img_arr)
    n_comps = len(cum_var)
    axes[2].set_title(f"Reconstructed from PCA\n({n_comps} components ~95% Var)", fontweight="bold")
    axes[2].axis("off")

    plt.suptitle("Image Preprocessing: Normalization, PCA Compression & Reconstruction", fontweight="bold", y=1.03)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved Image PCA plot to: {output_path}")


def plot_confusion_matrices(tabular_res: dict, text_res: dict, image_res: dict, output_path="reports/confusion_matrices_comparison.png"):
    """Plots side-by-side confusion matrices for Baseline vs Preprocessed across all modalities."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(11, 13))

    modalities = [
        ("Tabular (Customer Churn)", tabular_res, ["No", "Yes"]),
        ("Text (Review Sentiment)", text_res, ["Negative", "Positive"]),
        ("Image (Shape Classes)", image_res, image_res.get("classes", ["C1", "C2", "C3"]))
    ]

    for row_idx, (title, res, labels) in enumerate(modalities):
        # Baseline CM
        cm_base = res["baseline_metrics"]["confusion_matrix"]
        sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", ax=axes[row_idx, 0],
                    xticklabels=labels, yticklabels=labels, cbar=False)
        axes[row_idx, 0].set_title(f"{title}\nBaseline (Before)", fontweight="bold")
        axes[row_idx, 0].set_ylabel("True Label")
        axes[row_idx, 0].set_xlabel("Predicted Label")

        # Preprocessed CM
        cm_proc = res["preprocessed_metrics"]["confusion_matrix"]
        sns.heatmap(cm_proc, annot=True, fmt="d", cmap="Greens", ax=axes[row_idx, 1],
                    xticklabels=labels, yticklabels=labels, cbar=False)
        axes[row_idx, 1].set_title(f"{title}\nPreprocessed (After)", fontweight="bold")
        axes[row_idx, 1].set_ylabel("True Label")
        axes[row_idx, 1].set_xlabel("Predicted Label")

    plt.suptitle("Confusion Matrices: Baseline vs Preprocessed Pipelines", fontweight="bold", y=1.01)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved confusion matrices to: {output_path}")


def generate_all_report_visuals(tabular_res: dict, text_res: dict, image_res: dict, master_df: pd.DataFrame):
    """Generates all 4 visual report charts and saves them to reports/."""
    print("Generating comprehensive visual report charts...")
    plot_before_vs_after_benchmarks(master_df)
    
    # Load raw tabular for scaling demo
    if os.path.exists("data/tabular/telco_churn_raw.csv"):
        df_raw = pd.read_csv("data/tabular/telco_churn_raw.csv")
        num_col = "MonthlyCharges" if "MonthlyCharges" in df_raw.columns else df_raw.select_dtypes(include=[np.number]).columns[0]
    else:
        df_raw = pd.read_csv("data/tabular/customer_churn_raw.csv")
        num_col = "monthly_charges" if "monthly_charges" in df_raw.columns else df_raw.select_dtypes(include=[np.number]).columns[0]

    plot_tabular_scaling_and_outliers(df_raw, num_col=num_col)
    plot_image_pca_and_reconstruction(image_res)
    plot_confusion_matrices(tabular_res, text_res, image_res)
    print("All visual reports generated in 'reports/'!")
