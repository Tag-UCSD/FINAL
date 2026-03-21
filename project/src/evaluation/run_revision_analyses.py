#!/usr/bin/env python3
"""
Revision analyses for the paper update.

This script rebuilds the missing summary artifacts requested by review:
  - mean-prediction baseline
  - linear regression baseline
  - retrained Model B / D predictions from saved best hyperparameters
  - bootstrap confidence intervals
  - paired Wilcoxon tests and Cohen's d
  - random-indicator permutation control
  - VLM score diagnostics
  - revised comparison / pipeline figures
"""

import json
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon, entropy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "project" / "data"
OUT_RESULTS = ROOT / "project" / "outputs" / "results"
OUT_FIGURES = ROOT / "project" / "outputs" / "figures"

DATASET_PATH = DATA / "assembled_dataset" / "pilot_dataset.parquet"
SPLIT_PATH = DATA / "assembled_dataset" / "split_summary.json"
LGBM_SUMMARY_PATH = OUT_RESULTS / "lgbm_training_summary.json"
LGBM_IND_SUMMARY_PATH = OUT_RESULTS / "lgbm_indicators_training_summary.json"

AFFORDANCES = ["L059", "L079", "L091", "L130", "L141"]
AFF_NAMES = {
    "L059": "Sleep",
    "L079": "Cook",
    "L091": "Computer Work",
    "L130": "Conversation",
    "L141": "Yoga/Stretching",
}
SEED = 42
N_BOOT = 1000


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def get_raw_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col.startswith(("presence_", "count_", "dist_", "depth_diff_")):
            cols.append(col)
        elif col in (
            "total_object_count", "total_stuff_count", "num_unique_classes",
            "free_floor_fraction", "largest_object_area", "furniture_clutter_index",
            "mean_object_area", "std_object_area", "scene_complexity",
            "vertical_spread", "depth_available", "scene_depth_median_m",
            "scene_depth_range_m", "scene_depth_std_m",
        ):
            cols.append(col)
    return cols


def get_indicator_cols(df: pd.DataFrame, aff_id: str) -> list[str]:
    return [c for c in df.columns if c.startswith(f"ind_{aff_id}_")]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        r = 0.0
        rho = 0.0
    else:
        r = float(pearsonr(y_true, y_pred).statistic)
        rho = float(spearmanr(y_true, y_pred).statistic)
        if np.isnan(r):
            r = 0.0
        if np.isnan(rho):
            rho = 0.0
    return {"rmse": rmse, "mae": mae, "pearson_r": r, "spearman_rho": rho}


def bootstrap_metric_cis(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = N_BOOT) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    rmse_vals = []
    mae_vals = []
    pearson_vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        m = compute_metrics(yt, yp)
        rmse_vals.append(m["rmse"])
        mae_vals.append(m["mae"])
        pearson_vals.append(m["pearson_r"])
    return {
        "rmse": tuple(np.quantile(rmse_vals, [0.025, 0.975])),
        "mae": tuple(np.quantile(mae_vals, [0.025, 0.975])),
        "pearson_r": tuple(np.quantile(pearson_vals, [0.025, 0.975])),
    }


def bootstrap_rmse_diff_ci(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        rmse_a = np.sqrt(mean_squared_error(yt, pred_a[idx]))
        rmse_b = np.sqrt(mean_squared_error(yt, pred_b[idx]))
        vals.append(rmse_b - rmse_a)
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    sd = diff.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(diff.mean() / sd)


def format_split_summary(df: pd.DataFrame) -> dict:
    summary = {
        "total_rows": int(len(df)),
        "n_images": int(df["image_id"].nunique()),
        "n_affordances": len(AFFORDANCES),
        "affordance_ids": AFFORDANCES,
        "splits": {},
        "n_features_raw": len(get_raw_feature_cols(df)),
        "n_indicator_features": int(sum(c.startswith("ind_") for c in df.columns)),
        "cluster_split": {},
    }
    for split in ["train", "val", "test"]:
        dsplit = df[df["split"] == split]
        summary["splits"][split] = {
            "n_scenes": int(dsplit["scene_name"].nunique()),
            "n_rows": int(len(dsplit)),
        }
        summary["cluster_split"][split] = {
            str(k): int(v)
            for k, v in dsplit.drop_duplicates("scene_name")["cluster_assignment"].value_counts().items()
        }
    return summary


def load_params(path: Path) -> dict[str, dict]:
    with open(path) as f:
        data = json.load(f)
    return {row["affordance_id"]: row["best_params"] for row in data}


def savefig(fig, stem: Path) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"))
    plt.close(fig)


def main() -> None:
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATASET_PATH)
    raw_cols = get_raw_feature_cols(df)
    params_b = load_params(LGBM_SUMMARY_PATH)
    params_d = load_params(LGBM_IND_SUMMARY_PATH)

    with open(SPLIT_PATH, "w") as f:
        json.dump(format_split_summary(df), f, indent=2)

    mean_rows = []
    linear_rows = []
    test_rows = []
    stats_rows = []
    perm_rows = []
    ci_rows = []

    rng = np.random.default_rng(SEED)

    for aff_id in AFFORDANCES:
        df_aff = df[df["affordance_id"] == aff_id].copy()
        df_train = df_aff[df_aff["split"] == "train"].copy()
        df_val = df_aff[df_aff["split"] == "val"].copy()
        df_trainval = df_aff[df_aff["split"].isin(["train", "val"])].copy()
        df_test = df_aff[df_aff["split"] == "test"].copy()

        y_train = df_train["vlm_score"].to_numpy(dtype=float)
        y_trainval = df_trainval["vlm_score"].to_numpy(dtype=float)
        y_test = df_test["vlm_score"].to_numpy(dtype=float)

        X_trainval_raw = df_trainval[raw_cols].to_numpy(dtype=np.float32)
        X_test_raw = df_test[raw_cols].to_numpy(dtype=np.float32)

        ind_cols = get_indicator_cols(df, aff_id)
        X_trainval_d = df_trainval[raw_cols + ind_cols].to_numpy(dtype=np.float32)
        X_test_d = df_test[raw_cols + ind_cols].to_numpy(dtype=np.float32)

        mean_pred = np.full_like(y_test, y_train.mean(), dtype=float)
        mean_metrics = compute_metrics(y_test, mean_pred)
        mean_rows.append({
            "affordance_id": aff_id,
            "affordance_name": AFF_NAMES[aff_id],
            **{k: round(v, 4) for k, v in mean_metrics.items()},
            "train_mean": round(float(y_train.mean()), 4),
        })

        lin = LinearRegression()
        lin.fit(X_trainval_raw, y_trainval)
        lin_pred = lin.predict(X_test_raw)
        lin_metrics = compute_metrics(y_test, lin_pred)
        linear_rows.append({
            "affordance_id": aff_id,
            "affordance_name": AFF_NAMES[aff_id],
            **{k: round(v, 4) for k, v in lin_metrics.items()},
        })

        model_b = lgb.LGBMRegressor(**params_b[aff_id])
        model_b.fit(X_trainval_raw, y_trainval, callbacks=[lgb.log_evaluation(-1)])
        pred_b = model_b.predict(X_test_raw)
        metrics_b = compute_metrics(y_test, pred_b)

        model_d = lgb.LGBMRegressor(**params_d[aff_id])
        model_d.fit(X_trainval_d, y_trainval, callbacks=[lgb.log_evaluation(-1)])
        pred_d = model_d.predict(X_test_d)
        metrics_d = compute_metrics(y_test, pred_d)

        # Permutation control: preserve dimensionality and marginal sparsity, break image-indicator semantics.
        X_train_perm = df_trainval[raw_cols + ind_cols].copy()
        X_test_perm = df_test[raw_cols + ind_cols].copy()
        for col in ind_cols:
            X_train_perm[col] = rng.permutation(X_train_perm[col].to_numpy())
            X_test_perm[col] = rng.permutation(X_test_perm[col].to_numpy())
        model_perm = lgb.LGBMRegressor(**params_d[aff_id])
        model_perm.fit(X_train_perm.to_numpy(dtype=np.float32), y_trainval, callbacks=[lgb.log_evaluation(-1)])
        pred_perm = model_perm.predict(X_test_perm.to_numpy(dtype=np.float32))
        metrics_perm = compute_metrics(y_test, pred_perm)
        perm_rows.append({
            "affordance_id": aff_id,
            "affordance_name": AFF_NAMES[aff_id],
            "rmse_b": round(metrics_b["rmse"], 4),
            "rmse_d": round(metrics_d["rmse"], 4),
            "rmse_d_permuted": round(metrics_perm["rmse"], 4),
            "delta_rmse_d_vs_b": round(metrics_d["rmse"] - metrics_b["rmse"], 4),
            "delta_rmse_perm_vs_b": round(metrics_perm["rmse"] - metrics_b["rmse"], 4),
            "pearson_r_d": round(metrics_d["pearson_r"], 4),
            "pearson_r_d_permuted": round(metrics_perm["pearson_r"], 4),
            "n_indicator_cols": len(ind_cols),
        })

        ci_b = bootstrap_metric_cis(y_test, pred_b)
        ci_d = bootstrap_metric_cis(y_test, pred_d)
        ci_mean = bootstrap_metric_cis(y_test, mean_pred)
        ci_lin = bootstrap_metric_cis(y_test, lin_pred)
        for model_name, metrics, ci in [
            ("Mean", mean_metrics, ci_mean),
            ("LinearRegression", lin_metrics, ci_lin),
            ("LightGBM", metrics_b, ci_b),
            ("Indicator-LGBM", metrics_d, ci_d),
        ]:
            ci_rows.append({
                "affordance_id": aff_id,
                "affordance_name": AFF_NAMES[aff_id],
                "model": model_name,
                "rmse": round(metrics["rmse"], 4),
                "rmse_ci_low": round(ci["rmse"][0], 4),
                "rmse_ci_high": round(ci["rmse"][1], 4),
                "mae": round(metrics["mae"], 4),
                "mae_ci_low": round(ci["mae"][0], 4),
                "mae_ci_high": round(ci["mae"][1], 4),
                "pearson_r": round(metrics["pearson_r"], 4),
                "pearson_r_ci_low": round(ci["pearson_r"][0], 4),
                "pearson_r_ci_high": round(ci["pearson_r"][1], 4),
            })

        abs_err_b = np.abs(y_test - pred_b)
        abs_err_d = np.abs(y_test - pred_d)
        w_stat, w_p = wilcoxon(abs_err_b, abs_err_d, zero_method="wilcox", alternative="greater")
        rmse_diff_ci = bootstrap_rmse_diff_ci(y_test, pred_b, pred_d)
        stats_rows.append({
            "affordance_id": aff_id,
            "affordance_name": AFF_NAMES[aff_id],
            "rmse_b": round(metrics_b["rmse"], 4),
            "rmse_d": round(metrics_d["rmse"], 4),
            "delta_rmse_d_minus_b": round(metrics_d["rmse"] - metrics_b["rmse"], 4),
            "delta_rmse_ci_low": round(rmse_diff_ci[0], 4),
            "delta_rmse_ci_high": round(rmse_diff_ci[1], 4),
            "wilcoxon_stat": round(float(w_stat), 4),
            "wilcoxon_p": float(w_p),
            "cohens_d_abs_error": round(cohens_d_paired(abs_err_b, abs_err_d), 4),
            "mean_abs_error_b": round(float(abs_err_b.mean()), 4),
            "mean_abs_error_d": round(float(abs_err_d.mean()), 4),
            "n_test": int(len(y_test)),
        })

        for image_id, y, mp, lp, pb, pdp, pp in zip(
            df_test["image_id"], y_test, mean_pred, lin_pred, pred_b, pred_d, pred_perm
        ):
            test_rows.append({
                "image_id": image_id,
                "affordance_id": aff_id,
                "affordance_name": AFF_NAMES[aff_id],
                "y_true": round(float(y), 4),
                "pred_mean": round(float(mp), 4),
                "pred_linear": round(float(lp), 4),
                "pred_b": round(float(pb), 4),
                "pred_d": round(float(pdp), 4),
                "pred_d_permuted": round(float(pp), 4),
            })

    mean_df = pd.DataFrame(mean_rows)
    linear_df = pd.DataFrame(linear_rows)
    ci_df = pd.DataFrame(ci_rows)
    stats_df = pd.DataFrame(stats_rows)
    perm_df = pd.DataFrame(perm_rows)
    pred_df = pd.DataFrame(test_rows)

    mean_df.to_csv(OUT_RESULTS / "baseline_mean_prediction.csv", index=False)
    linear_df.to_csv(OUT_RESULTS / "baseline_linear_regression.csv", index=False)
    ci_df.to_csv(OUT_RESULTS / "metric_confidence_intervals.csv", index=False)
    stats_df.to_csv(OUT_RESULTS / "statistical_tests.csv", index=False)
    perm_df.to_csv(OUT_RESULTS / "indicator_permutation_test.csv", index=False)
    pred_df.to_csv(OUT_RESULTS / "revision_test_predictions.csv", index=False)

    diag_rows = []
    score_wide = df.pivot(index="image_id", columns="affordance_id", values="vlm_score")
    for aff_id in AFFORDANCES:
        vals = df[df["affordance_id"] == aff_id]["vlm_score"].to_numpy()
        counts = pd.Series(vals).value_counts().sort_index()
        probs = counts / counts.sum()
        diag_rows.append({
            "analysis": "affordance_distribution",
            "affordance_id": aff_id,
            "affordance_name": AFF_NAMES[aff_id],
            "metric": "entropy_bits",
            "value": round(float(entropy(probs, base=2)), 4),
        })
        diag_rows.append({
            "analysis": "affordance_distribution",
            "affordance_id": aff_id,
            "affordance_name": AFF_NAMES[aff_id],
            "metric": "fraction_score_1",
            "value": round(float(np.mean(vals == 1)), 4),
        })
        diag_rows.append({
            "analysis": "affordance_distribution",
            "affordance_id": aff_id,
            "affordance_name": AFF_NAMES[aff_id],
            "metric": "fraction_score_7",
            "value": round(float(np.mean(vals == 7)), 4),
        })
    for a, b in combinations(AFFORDANCES, 2):
        r = score_wide[[a, b]].corr(method="pearson").iloc[0, 1]
        diag_rows.append({
            "analysis": "inter_affordance_correlation",
            "affordance_id": f"{a}-{b}",
            "affordance_name": f"{AFF_NAMES[a]} vs {AFF_NAMES[b]}",
            "metric": "pearson_r",
            "value": round(float(r), 4),
        })
    pd.DataFrame(diag_rows).to_csv(OUT_RESULTS / "vlm_score_diagnostics.csv", index=False)

    # Revised comparison figure with CI bars for reproducible non-CNN baselines.
    plot_df = ci_df[ci_df["model"].isin(["Mean", "LinearRegression", "LightGBM", "Indicator-LGBM"])].copy()
    model_order = ["Mean", "LinearRegression", "LightGBM", "Indicator-LGBM"]
    model_colors = {
        "Mean": "#B5B5B5",
        "LinearRegression": "#6C8EAD",
        "LightGBM": "#DD8452",
        "Indicator-LGBM": "#55A868",
    }
    x = np.arange(len(AFFORDANCES))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric, ylabel in [
        (axes[0], "rmse", "RMSE (score points)"),
        (axes[1], "pearson_r", "Pearson r"),
    ]:
        for i, model in enumerate(model_order):
            subset = plot_df[plot_df["model"] == model].set_index("affordance_id").loc[AFFORDANCES]
            vals = subset[metric].to_numpy()
            lo = subset[f"{metric}_ci_low"].to_numpy()
            hi = subset[f"{metric}_ci_high"].to_numpy()
            yerr = np.vstack([vals - lo, hi - vals])
            ax.bar(x + (i - 1.5) * width, vals, width=width, color=model_colors[model], label=model, alpha=0.9)
            ax.errorbar(x + (i - 1.5) * width, vals, yerr=yerr, fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{aid}\n{AFF_NAMES[aid]}" for aid in AFFORDANCES])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("Test RMSE with 95% bootstrap CIs")
    axes[1].set_title("Test Pearson r with 95% bootstrap CIs")
    axes[1].legend(ncol=2, frameon=False, loc="lower right")
    plt.tight_layout()
    savefig(fig, OUT_FIGURES / "experiment1_model_comparison")

    # Revised indicator figure.
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    subset = stats_df.set_index("affordance_id").loc[AFFORDANCES]
    vals = subset["delta_rmse_d_minus_b"].to_numpy()
    lo = subset["delta_rmse_ci_low"].to_numpy()
    hi = subset["delta_rmse_ci_high"].to_numpy()
    yerr = np.vstack([vals - lo, hi - vals])
    ax.bar(np.arange(len(AFFORDANCES)), vals, color="#55A868", alpha=0.9)
    ax.errorbar(np.arange(len(AFFORDANCES)), vals, yerr=yerr, fmt="none", ecolor="black", capsize=3, elinewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(AFFORDANCES)))
    ax.set_xticklabels([f"{aid}\n{AFF_NAMES[aid]}" for aid in AFFORDANCES])
    ax.set_ylabel("ΔRMSE (Model D - Model B)")
    ax.set_title("Indicator distillation effect with 95% bootstrap CIs")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    savefig(fig, OUT_FIGURES / "experiment5_indicator_value")

    # Simple pipeline figure.
    fig, ax = plt.subplots(figsize=(10.5, 2.8))
    ax.axis("off")
    boxes = [
        (0.04, "Hypersim scene image\n(scene-level split)"),
        (0.28, "Qwen2-VL-7B teacher\nscore + indicator checklist"),
        (0.54, "Mask2Former features\n310 structured scene cues"),
        (0.78, "Student regressors\nMean / Linear / LightGBM / Indicator-LGBM"),
    ]
    for x0, label in boxes:
        rect = plt.Rectangle((x0, 0.32), 0.18, 0.34, facecolor="#F7F3E8", edgecolor="#333333", linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x0 + 0.09, 0.49, label, ha="center", va="center", fontsize=10)
    for x0 in [0.22, 0.48, 0.72]:
        ax.annotate("", xy=(x0 + 0.05, 0.49), xytext=(x0, 0.49), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(0.41, 0.76, "distilled supervision", ha="center", va="center", fontsize=9)
    ax.text(0.63, 0.76, "structured representation", ha="center", va="center", fontsize=9)
    ax.text(0.87, 0.18, "Evaluate on held-out scenes with RMSE, MAE, r, rho, and paired tests", ha="center", va="center", fontsize=9)
    plt.tight_layout()
    savefig(fig, OUT_FIGURES / "pipeline_overview")


if __name__ == "__main__":
    main()
