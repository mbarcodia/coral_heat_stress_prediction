import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import shap
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from xgboost import XGBClassifier
from scipy import stats
import os
from pathlib import Path


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_test_xgboost(X_train, Y_train, X_test, Y_test, fd, weights_train, fold, weeks_per_year):
    """Train XGBClassifier and return model + predictions + fold MAE."""
    xgb_model = XGBClassifier(**fd)
    xgb_model.fit(X_train, Y_train, sample_weight=weights_train)

    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)

    print(f"Fold {fold + 1} – TRUTH")
    print(format_week_delimited(Y_test, weeks_per_year))
    print(f"Fold {fold + 1} – XGB PREDICTIONS")
    print(format_week_delimited(np.round(y_pred_test, 0), weeks_per_year))

    mode_train = stats.mode(np.round(y_pred_train, 0))
    mae_xgb = mean_absolute_error(Y_test, y_pred_test)
    print(f"Fold {fold + 1} – MAE XGB: {mae_xgb:.2f}")

    return xgb_model, y_pred_train, y_pred_test, mode_train, mae_xgb


def train_test_linear(X_train, Y_train, X_test, Y_test, weights_train, fold, weeks_per_year):
    """Train multinomial LogisticRegression and return model + predictions + fold MAE."""
    model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
    model.fit(X_train, Y_train, sample_weight=weights_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print(f"Fold {fold + 1} – LINEAR PREDICTIONS")
    print(format_week_delimited(np.round(y_pred_test, 0), weeks_per_year))

    mae_lin = mean_absolute_error(Y_test, y_pred_test)
    print(f"Fold {fold + 1} – MAE Linear: {mae_lin:.2f}")

    return model, y_pred_train, y_pred_test, mae_lin


# ─────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────

def calculate_weekly_errors(Y_true, Y_pred, array_weeks, weeks_per_year):
    """Mean absolute error per true label week (0–val_no_onset)."""
    mean_error_per_week = np.zeros(len(array_weeks))
    for count, week in enumerate(array_weeks):
        indices = np.where(Y_true == week)[0]
        if indices.size > 0:
            mean_error_per_week[count] = mean_absolute_error(Y_true[indices], Y_pred[indices])
        else:
            mean_error_per_week[count] = np.nan
    return mean_error_per_week


def apply_onset_mask(labels, preds_xgb, preds_lin, preds_baseline, weeks_per_year):
    """Mask predictions to NaN for all weeks after the first true onset in each year.

    Once a bleaching onset is observed (label == 0), subsequent predictions in
    that year are excluded from metric computation because the event has already
    occurred.
    """
    masked_xgb = preds_xgb.astype(float).copy()
    masked_lin = preds_lin.astype(float).copy()
    masked_base = preds_baseline.astype(float).copy()

    for start_idx in range(0, len(labels), weeks_per_year):
        end_idx = min(start_idx + weeks_per_year, len(labels))
        chunk = labels[start_idx:end_idx]
        if 0 in chunk:
            first_zero = np.where(chunk == 0)[0][0]
            post_onset = range(start_idx + first_zero + 1, end_idx)
            for idx in post_onset:
                masked_xgb[idx] = np.nan
                masked_lin[idx] = np.nan
                masked_base[idx] = np.nan

    return masked_xgb, masked_lin, masked_base


def apply_onset_mask_single(labels, preds, weeks_per_year):
    """Mask a single prediction array — used by the hyperparameter sweep."""
    masked = preds.astype(float).copy()
    for start_idx in range(0, len(labels), weeks_per_year):
        end_idx = min(start_idx + weeks_per_year, len(labels))
        chunk = labels[start_idx:end_idx]
        if 0 in chunk:
            first_zero = np.where(chunk == 0)[0][0]
            masked[start_idx + first_zero + 1 : end_idx] = np.nan
    return masked


# ─────────────────────────────────────────────
# Feature naming
# ─────────────────────────────────────────────

def rename_features(feature_names):
    """Map internal column names to human-readable labels for plots."""
    rename_map = {
        "zonalWindInter": "zonal wind",
        "meridionalWindInter": "meridional wind",
        "meanDhw": "DHW",
        "airTempInter": "air temp",
        "meanHs": "HotSpot",
        "oniInter": "ENSO Index",
        "meanDhw_lag_1": "DHW 1wk lag",
        "meanDhw_lag_2": "DHW 2wk lag",
        "meanSsta": "SST anomaly",
        "loop_curr_sst": "Loop Current Index",
        "solar_flux": "solar flux",
        "dswrf": "solar flux",
        "week": "week of year",
    }
    new_names = []
    for name in feature_names:
        new_name = name
        for key, val in rename_map.items():
            if key in name:
                new_name = new_name.replace(key, val)
        new_names.append(new_name)
    return new_names


# ─────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────

def format_week_delimited(array, weeks_per_year):
    """Print a flat array as a grid of weeks_per_year columns (one row per year)."""
    return "\n".join(
        " ".join(
            f"{int(array[i]):2}" for i in range(j, min(j + weeks_per_year, len(array)))
        )
        for j in range(0, len(array), weeks_per_year)
    )


# ─────────────────────────────────────────────
# SHAP — per-fold multiclass summary
# ─────────────────────────────────────────────

def save_shap_plot_multiclass(
    explainer, X_test, feature_names, fold, base_values, test_years, output_dir, class_names
):
    """Save one SHAP beeswarm plot per class for a single fold."""
    output_dir = Path(output_dir)
    X_test_np = np.array(X_test)
    shap_values = explainer.shap_values(X_test, check_additivity=False)

    for class_idx, class_name in enumerate(class_names):
        base_val_str = (
            np.round(base_values[class_idx], 3)
            if hasattr(base_values, "__len__") and len(base_values) > 1
            else np.round(base_values, 3)
        )
        plt.figure()
        shap.summary_plot(
            shap_values[:, :, class_idx],
            X_test_np,
            feature_names=feature_names,
            show=False,
        )
        plt.title(
            f"Fold {fold + 1} SHAP – Class {class_name} | "
            f"Test Years: {test_years[0]}–{test_years[1]} | Base: {base_val_str}"
        )
        plt.savefig(
            output_dir / f"shap_fold_{fold + 1}_class_{class_name}.png", dpi=300
        )
        plt.close()


# ─────────────────────────────────────────────
# SHAP — combined all-folds summary figure
# ─────────────────────────────────────────────

def save_shap_summary_figure(
    all_shap_values,
    all_folds_expected_values,
    X_values,
    renamed_features,
    val_no_onset,
    output_path,
    site_display_name="",
):
    """Generate the publication-quality 8-panel SHAP summary figure.

    Parameters
    ----------
    all_shap_values : ndarray, shape (n_samples, n_features, n_classes)
        SHAP values concatenated across all folds.
    all_folds_expected_values : ndarray, shape (n_folds, n_classes)
        Per-fold expected (base) values.
    X_values : ndarray, shape (n_samples, n_features)
        Raw feature values (for coloring).
    renamed_features : list of str
        Human-readable feature names.
    val_no_onset : int
        The class index that represents "no onset" (e.g., 7).
    output_path : str or Path
        Full path to save the figure.
    site_display_name : str
        Site name used in the figure title (e.g., "Sand Key").
    """
    num_classes = val_no_onset + 1  # 0 … val_no_onset
    letters = [chr(ord("a") + i) for i in range(num_classes)]
    nrows, ncols = 4, 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, 22), sharex=True)
    axes = axes.flatten()

    shap_vals = all_shap_values  # (n_samples, n_features, n_classes)
    num_features = len(renamed_features)
    cmap = cm.get_cmap("magma_r")
    expected_avg = np.mean(all_folds_expected_values, axis=0)

    for class_idx in range(num_classes):
        ax = axes[class_idx]
        shap_vals_class = shap_vals[:, :, class_idx]
        mean_abs_shap = np.abs(shap_vals_class).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:num_features][::-1]

        for i, feat_idx in enumerate(top_indices):
            feat_vals = X_values[:, feat_idx]
            norm = Normalize(vmin=np.percentile(feat_vals, 1), vmax=np.percentile(feat_vals, 99))
            color_vals = cmap(norm(feat_vals))
            y_jitter = i + np.random.uniform(-0.05, 0.05, size=len(feat_vals))
            ax.scatter(
                shap_vals_class[:, feat_idx],
                y_jitter,
                color=color_vals,
                s=30,
                alpha=0.8,
                rasterized=True,
            )

        ax.axvline(x=0, color="gray", linestyle="--", linewidth=1)
        ax.set_yticks(range(len(top_indices)))
        ax.set_xlim(-2, 2)

        # Axis labels (left two columns only)
        if class_idx % ncols in (0, 1):
            ax.set_yticklabels([renamed_features[i] for i in top_indices], fontsize=19)
        else:
            ax.set_yticklabels([])

        ax.set_xticks(np.linspace(-2, 2, 9))
        ax.tick_params(axis="x", labelsize=18, labelbottom=True)
        ax.set_xlabel("SHAP value", fontsize=18)

        # Titles
        if class_idx == 1:
            title_str = r"$\bf{1\ week\ until\ onset}$"
        elif class_idx == val_no_onset:
            title_str = r"$\bf{no\ onset}$"
        else:
            s = "s" if class_idx != 1 else ""
            title_str = rf"$\bf{{{class_idx}\ week{s}\ until\ onset}}$"

        ax.set_title(title_str, fontsize=22, loc="center")
        ax.text(1.0, 1.02, f"base value={np.round(expected_avg[class_idx], 2)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=16)
        ax.text(0.0, 1.02, f"{letters[class_idx]})",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=20)

    # Wider x-axis for onset class (class 0)
    axes[0]._shared_axes["x"].remove(axes[0])
    axes[0].set_xlim(-5, 3)
    axes[0].set_xticks(np.linspace(-5, 3, 9))

    # Colorbar
    cbar_ax = fig.add_axes([0.93, 0.05, 0.01, 0.9])
    sm = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.yaxis.set_ticks([])
    cbar.ax.text(1.5, 0, "Low", va="bottom", ha="left", fontsize=24, transform=cbar.ax.transAxes)
    cbar.ax.text(1.5, 1, "High", va="top", ha="left", fontsize=28, transform=cbar.ax.transAxes)
    cbar.set_label("Feature value", fontsize=28, labelpad=24)

    plt.subplots_adjust(left=0.07, right=0.91, top=0.95, bottom=0.05, hspace=0.3, wspace=0.32)
    title = "Explainable AI Feature Ranking by Prediction Class"
    if site_display_name:
        title += f" — {site_display_name}"
    plt.suptitle(title, fontsize=34, y=1.0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"SHAP summary figure saved to {output_path}")


# ─────────────────────────────────────────────
# Performance figures
# ─────────────────────────────────────────────

def save_performance_figure(
    labels,
    masked_preds_xgb,
    masked_preds_lin,
    masked_preds_baseline,
    mean_error_per_week_xgb,
    mean_error_per_week_lin,
    mean_error_per_week_freq,
    metrics,
    val_no_onset,
    tolerance,
    output_path,
    site_display_name="",
):
    """Save the combined scatter (MAE vs weeks until onset) + bar (metrics) figure."""
    model_colors = {"XGB": "deeppink", "Linear": "gray", "Frequency": "darkgreen"}
    metric_labels = [
        f"Onset Accuracy\n(±{tolerance} week)",
        "No Onset Accuracy",
        "False Alarm Rate",
        "Missed Onset Rate",
    ]

    fig, (ax_bar, ax_scatter) = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 1.5]}
    )

    # --- Scatter: MAE vs weeks until onset ---
    for model, color, marker, data in zip(
        ["Linear", "Frequency", "XGB"],
        ["gray", "darkgreen", "deeppink"],
        ["o", "s", "D"],
        [mean_error_per_week_lin, mean_error_per_week_freq, mean_error_per_week_xgb],
    ):
        ax_scatter.scatter(
            np.arange(0, val_no_onset),
            data[:val_no_onset],
            color=color, marker=marker, s=160, label=model,
        )
    ax_scatter.plot(
        np.arange(0, val_no_onset), np.arange(0, val_no_onset),
        color="grey", linestyle="--", alpha=0.5,
    )
    ax_scatter.set_xlim(-0.5, val_no_onset - 0.5)
    ax_scatter.set_xticks(np.arange(0, val_no_onset))
    ax_scatter.set_ylim(-0.5, val_no_onset - 0.5)
    ax_scatter.set_xlabel("Weeks until onset", fontsize=20)
    ax_scatter.set_ylabel("Mean Absolute Error", fontsize=20)
    ax_scatter.set_title("Prediction Performance for Years with Stress Onset", fontsize=18)
    ax_scatter.tick_params(labelsize=16)
    handles, labs = ax_scatter.get_legend_handles_labels()
    ax_scatter.legend([handles[2], handles[0], handles[1]],
                      [labs[2], labs[0], labs[1]], fontsize=16, loc="upper right")

    # --- Bar: classification metrics ---
    bar_width = 0.2
    x_indexes = np.arange(len(metric_labels))
    for i, (model, color) in enumerate(model_colors.items()):
        ax_bar.bar(
            x_indexes + i * bar_width - bar_width,
            metrics[model][0:4],
            width=bar_width,
            label=f"{model}; F1: {metrics[model][-1]:.2f}",
            color=color, alpha=0.7,
        )
    ax_bar.set_ylabel("Percentage (%)", fontsize=14)
    ax_bar.set_title("Performance Metrics for All Predictions", fontsize=18)
    ax_bar.set_xticks(x_indexes)
    ax_bar.set_xticklabels(metric_labels, fontsize=13)
    ax_bar.set_ylim(0, 100)
    ax_bar.legend(fontsize=13, loc="upper right")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.5)
    ax_bar.tick_params(axis="y", labelsize=13)

    title = f"Performance — {site_display_name}" if site_display_name else "Performance"
    plt.suptitle(title, fontsize=20)
    plt.subplots_adjust(hspace=0.45)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Performance figure saved to {output_path}")


def save_metrics_table(df_metrics, output_path):
    """Save a formatted metrics table as a PNG."""
    columns = df_metrics.columns.tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df_metrics.values,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        colColours=["#DDDDDD"] * len(columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.4, 1.4)
    for col in range(len(columns)):
        cell = table[0, col]
        cell.set_fontsize(14)
        cell.set_text_props(weight="bold")
        cell.set_height(0.15)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Metrics table saved to {output_path}")
