"""
Visualization Module for Causal Medical AI.
Generates publication-quality figures:
- CATE distributions vs Ground Truth
- Covariate balance Love Plot (SMD before/after weighting)
- Cumulative Gain / Qini Curves
- Rosenbaum Sensitivity Bound curves
- Benchmark performance comparisons
- Subgroup Treatment Effect Forest Plot
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple


def set_plotting_style():
    """Configure clean publication-ready plotting style."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update(
        {
            "figure.autolayout": True,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def plot_cate_distribution(
    predictions_dict: Dict[str, np.ndarray],
    true_cate: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot kernel density estimates of predicted CATE across estimators."""
    set_plotting_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    palette = sns.color_palette("tab10")

    if true_cate is not None:
        sns.kdeplot(
            true_cate,
            ax=ax,
            label="Ground Truth CATE",
            color="black",
            linewidth=2.5,
            linestyle="--",
        )

    for i, (name, preds) in enumerate(predictions_dict.items()):
        sns.kdeplot(
            preds,
            ax=ax,
            label=name,
            color=palette[i % len(palette)],
            linewidth=1.8,
            alpha=0.85,
        )

    ax.set_title("Distribution of Estimated Individual Treatment Effects (CATE)", weight="bold")
    ax.set_xlabel("Predicted Treatment Effect (Cardiovascular Risk Reduction %)")
    ax.set_ylabel("Density")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_love_plot(
    smd_df: pd.DataFrame,
    threshold: float = 0.10,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate Love Plot displaying Standardized Mean Differences before and after IPW weighting.
    """
    set_plotting_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    df_sorted = smd_df.sort_values(by="abs_unweighted_smd", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df_sorted))

    ax.scatter(
        df_sorted["abs_unweighted_smd"],
        y_pos,
        color="#d62728",
        label="Unadjusted (Observational)",
        s=60,
        zorder=3,
    )
    ax.scatter(
        df_sorted["abs_weighted_smd"],
        y_pos,
        color="#2ca02c",
        label="Adjusted (IPW Weighted)",
        s=60,
        zorder=3,
    )

    # Connect dots
    for i in range(len(df_sorted)):
        ax.plot(
            [df_sorted.loc[i, "abs_unweighted_smd"], df_sorted.loc[i, "abs_weighted_smd"]],
            [y_pos[i], y_pos[i]],
            color="gray",
            linestyle=":",
            alpha=0.6,
        )

    ax.axvline(threshold, color="#333333", linestyle="--", linewidth=1.2, label=f"Balance Threshold ({threshold})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["feature"])
    ax.set_xlabel("Absolute Standardized Mean Difference (SMD)")
    ax.set_title("Covariate Balance Diagnostic (Love Plot)", weight="bold")
    ax.legend(loc="lower right", frameon=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_qini_curves(
    qini_dict: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot cumulative uplift / Qini curves for treatment prioritization."""
    set_plotting_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))

    palette = sns.color_palette("tab10")

    for i, (name, (fractions, gains, auuc)) in enumerate(qini_dict.items()):
        ax.plot(
            fractions * 100.0,
            gains,
            label=f"{name} (AUUC={auuc:.2f})",
            color=palette[i % len(palette)],
            linewidth=2.0,
        )

    # Random assignment baseline
    ax.plot([0, 100], [0, max([g[-1] for _, (_, g, _) in qini_dict.items()]) if qini_dict else 0],
            color="black", linestyle=":", label="Random Assignment")

    ax.set_title("Cumulative Uplift / Qini Curves for Personalized Treatment", weight="bold")
    ax.set_xlabel("Percentage of Population Treated (%)")
    ax.set_ylabel("Cumulative Clinical Gain")
    ax.legend(loc="upper left", frameon=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_rosenbaum_sensitivity(
    rosenbaum_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Rosenbaum Sensitivity Bounds (Gamma vs p-value upper bound)."""
    set_plotting_style()
    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.plot(
        rosenbaum_df["gamma"],
        rosenbaum_df["p_value_upper"],
        marker="o",
        color="#1f77b4",
        linewidth=2.2,
        label="Upper Bound P-value",
    )
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1.5, label="Significance Threshold (p = 0.05)")

    ax.set_title(r"Rosenbaum Sensitivity Analysis for Hidden Confounding ($\Gamma$)", weight="bold")
    ax.set_xlabel(r"Hidden Bias Odds Ratio ($\Gamma$)")
    ax.set_ylabel("P-value Upper Bound")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", frameon=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_benchmark_metrics(
    benchmark_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar plot comparing benchmark performance across estimators."""
    set_plotting_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    metrics = [
        ("pehe", "PEHE (Lower is Better)", axes[0, 0], False),
        ("ate_abs_error", "ATE Absolute Error (Lower is Better)", axes[0, 1], False),
        ("policy_value", "Policy Value (Higher is Better)", axes[1, 0], True),
        ("auuc", "AUUC / Qini (Higher is Better)", axes[1, 1], True),
    ]

    for metric, title, ax, higher_better in metrics:
        if metric in benchmark_df.columns:
            sorted_df = benchmark_df.sort_values(by=metric, ascending=not higher_better)
            colors = sns.color_palette("Blues_r" if higher_better else "Blues", len(sorted_df))
            bars = ax.barh(sorted_df["estimator"], sorted_df[metric], color=colors, edgecolor="#333333")
            ax.set_title(title, weight="bold")
            # Annotate values
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.01 * (max(sorted_df[metric]) - min(sorted_df[metric]) + 1e-4),
                        bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_subgroup_forest(
    subgroups_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Forest plot of heterogeneous treatment effects across key clinical subgroups."""
    set_plotting_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    y_pos = np.arange(len(subgroups_df))
    ax.errorbar(
        subgroups_df["mean_cate"],
        y_pos,
        xerr=[
            subgroups_df["mean_cate"] - subgroups_df["ci_lower"],
            subgroups_df["ci_upper"] - subgroups_df["mean_cate"],
        ],
        fmt="o",
        color="#2b5c8f",
        ecolor="#2b5c8f",
        elinewidth=2.0,
        capsize=4,
        markersize=7,
    )

    ax.axvline(0, color="black", linestyle="--", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(subgroups_df["subgroup"])
    ax.set_xlabel("Estimated Treatment Effect (Risk Reduction %)")
    ax.set_title("Subgroup Analysis: Heterogeneous Treatment Effects (95% CI)", weight="bold")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
