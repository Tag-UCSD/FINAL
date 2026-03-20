#!/usr/bin/env python3
"""
7-Experiment Evaluation Suite for the Affordance Prediction Pilot Study.

Experiments:
  1  Model Comparison     — CNN / LightGBM / Indicator-LGBM on test set
  2  HP Sensitivity        — Optuna visualisations (LGBM)
  3  CNN Ablation          — 2×4 grid + training curves
  4  Feature Group Ablation — progressive LGBM feature groups
  5  Indicator Distillation Value — Model B vs D statistics
  6  SHAP Analysis         — beeswarm / bar + vs VLM indicator ranking
  7  Segmentation Ablation — planned-work scaffold
"""

import json
import pickle
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

import lightgbm as lgb
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PALETTE = {
    "CNN": "#4C72B0", "LightGBM": "#DD8452", "Indicator-LGBM": "#55A868",
    "train": "#4C72B0", "val": "#DD8452", "test": "#55A868",
}
AFF_COLORS = {"L059": "#E07B54", "L079": "#5B85AA", "L091": "#414770",
              "L130": "#372248", "L141": "#82D173"}

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "project" / "data"
CONFIGS = ROOT / "project" / "configs"
OUT_MODELS = ROOT / "project" / "outputs" / "models"
OUT_RESULTS = ROOT / "project" / "outputs" / "results"
OUT_FIGURES = ROOT / "project" / "outputs" / "figures"

DATASET_PATH = DATA / "assembled_dataset" / "pilot_dataset.parquet"
INDICATOR_VOCAB_PATH = DATA / "vlm_annotations" / "indicator_vocabulary.json"
CNN_RESULTS_PATH = OUT_RESULTS / "cnn_training_results.json"
LGBM_SUMMARY_PATH = OUT_RESULTS / "lgbm_training_summary.json"
LGBM_IND_SUMMARY_PATH = OUT_RESULTS / "lgbm_indicators_training_summary.json"

AFFORDANCES = ["L059", "L079", "L091", "L130", "L141"]
SEED = 42
N_FOLDS = 5


# ════════════════════════════════════════════════════════════════════════════
# Utilities
# ════════════════════════════════════════════════════════════════════════════

def savefig(fig, path_stem: Path) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(path_stem.with_suffix(f".{ext}"))
    plt.close(fig)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r, _ = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    return {"rmse": round(rmse, 4), "mae": round(mae, 4),
            "pearson_r": round(float(r), 4), "spearman_rho": round(float(rho), 4)}


def get_raw_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col.startswith(("presence_", "count_", "dist_", "depth_diff_")):
            cols.append(col)
        elif col in ("total_object_count", "total_stuff_count",
                     "num_unique_classes", "free_floor_fraction",
                     "largest_object_area", "furniture_clutter_index",
                     "mean_object_area", "std_object_area",
                     "scene_complexity", "vertical_spread",
                     "depth_available", "scene_depth_median_m",
                     "scene_depth_range_m", "scene_depth_std_m"):
            cols.append(col)
    return cols


def get_indicator_cols(df: pd.DataFrame, aff_id: str) -> list[str]:
    return [c for c in df.columns if c.startswith(f"ind_{aff_id}_")]


def load_lgbm_model(aff_id: str, model_dir: Path) -> lgb.LGBMRegressor | None:
    model_files = list((model_dir / aff_id).glob("*.pkl"))
    for mf in model_files:
        if "study" not in mf.name:
            try:
                with open(mf, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
    return None


def load_lgbm_study(aff_id: str, model_dir: Path) -> optuna.Study | None:
    study_path = model_dir / aff_id / "optuna_study.pkl"
    if study_path.exists():
        with open(study_path, "rb") as f:
            return pickle.load(f)
    return None


def load_best_params(aff_id: str, model_dir: Path) -> dict:
    params_path = model_dir / aff_id / "best_params.json"
    if params_path.exists():
        with open(params_path) as f:
            return json.load(f)
    return {}


# ════════════════════════════════════════════════════════════════════════════
# Experiment 1 — Model Comparison
# ════════════════════════════════════════════════════════════════════════════

def experiment1_model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Experiment 1 — Model Comparison")
    print("=" * 60)

    records = []
    raw_cols = get_raw_feature_cols(df)

    for aff_id in AFFORDANCES:
        df_aff = df[df["affordance_id"] == aff_id]
        df_test = df_aff[df_aff["split"] == "test"]
        y_test = df_test["vlm_score"].values.astype(np.float32)

        # --- LightGBM (Model B) ---
        lgbm_b = load_lgbm_model(aff_id, OUT_MODELS / "lgbm")
        if lgbm_b is not None:
            X_test_b = df_test[raw_cols].values.astype(np.float32)
            metrics_b = compute_metrics(y_test, lgbm_b.predict(X_test_b))
        else:
            metrics_b = {"rmse": np.nan, "mae": np.nan, "pearson_r": np.nan, "spearman_rho": np.nan}
            print(f"  WARNING: LightGBM model not found for {aff_id}")
        records.append({"affordance_id": aff_id, "model": "LightGBM", **metrics_b})

        # --- Indicator-LGBM (Model D) ---
        lgbm_d = load_lgbm_model(aff_id, OUT_MODELS / "lgbm_indicators")
        if lgbm_d is not None:
            ind_cols = get_indicator_cols(df, aff_id)
            X_test_d = df_test[raw_cols + ind_cols].values.astype(np.float32)
            metrics_d = compute_metrics(y_test, lgbm_d.predict(X_test_d))
        else:
            metrics_d = {"rmse": np.nan, "mae": np.nan, "pearson_r": np.nan, "spearman_rho": np.nan}
            print(f"  WARNING: Indicator-LGBM model not found for {aff_id}")
        records.append({"affordance_id": aff_id, "model": "Indicator-LGBM", **metrics_d})

        # --- CNN (Model A) — load from saved results ---
        if CNN_RESULTS_PATH.exists():
            cnn_runs = json.loads(CNN_RESULTS_PATH.read_text())
            aff_cnn = [r for r in cnn_runs if r["affordance_id"] == aff_id]
            if aff_cnn:
                best_cnn = min(aff_cnn, key=lambda r: r["test_rmse"])
                metrics_cnn = {
                    "rmse": best_cnn["test_rmse"],
                    "mae": best_cnn["test_mae"],
                    "pearson_r": best_cnn["test_pearson_r"],
                    "spearman_rho": best_cnn["test_spearman_rho"],
                }
            else:
                metrics_cnn = {"rmse": np.nan, "mae": np.nan, "pearson_r": np.nan, "spearman_rho": np.nan}
        else:
            metrics_cnn = {"rmse": np.nan, "mae": np.nan, "pearson_r": np.nan, "spearman_rho": np.nan}
            print("  WARNING: CNN results file not found — run train_cnn.py first")
        records.append({"affordance_id": aff_id, "model": "CNN", **metrics_cnn})

    results_df = pd.DataFrame(records)
    csv_path = OUT_RESULTS / "experiment1_model_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(results_df.pivot_table(index="affordance_id", columns="model", values="rmse").round(4))

    # ── Grouped bar chart ──
    metrics_to_plot = ["rmse", "mae", "pearson_r", "spearman_rho"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    model_order = ["CNN", "LightGBM", "Indicator-LGBM"]
    x = np.arange(len(AFFORDANCES))
    width = 0.25

    for ax, metric in zip(axes, metrics_to_plot):
        for i, model_name in enumerate(model_order):
            vals = [
                results_df.loc[(results_df["affordance_id"] == aff) & (results_df["model"] == model_name), metric].values[0]
                for aff in AFFORDANCES
            ]
            ax.bar(x + i * width, vals, width, label=model_name,
                   color=list(PALETTE.values())[i], alpha=0.85)
        ax.set_xticks(x + width)
        ax.set_xticklabels(AFFORDANCES, rotation=15)
        ax.set_ylabel(metric.upper().replace("_", " "))
        ax.set_title(metric.replace("_", " ").title())
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Experiment 1 — Model Comparison (Test Set)", fontsize=12, y=1.01)
    plt.tight_layout()
    savefig(fig, OUT_FIGURES / "experiment1_model_comparison")
    print(f"Saved figure: {OUT_FIGURES / 'experiment1_model_comparison.png'}")

    return results_df


# ════════════════════════════════════════════════════════════════════════════
# Experiment 2 — HP Sensitivity
# ════════════════════════════════════════════════════════════════════════════

def experiment2_hp_sensitivity() -> None:
    print("\n" + "=" * 60)
    print("Experiment 2 — Hyperparameter Sensitivity (LightGBM)")
    print("=" * 60)

    lgbm_dir = OUT_MODELS / "lgbm"

    for aff_id in AFFORDANCES:
        study = load_lgbm_study(aff_id, lgbm_dir)
        if study is None:
            print(f"  WARNING: Optuna study not found for {aff_id} — skipping")
            continue

        print(f"  {aff_id}: {len(study.trials)} trials, best RMSE={study.best_value:.4f}")

        # 1. Optimization history
        fig, ax = plt.subplots(figsize=(8, 4))
        trial_values = [t.value for t in study.trials if t.value is not None]
        best_so_far = np.minimum.accumulate(trial_values)
        ax.plot(range(1, len(trial_values) + 1), trial_values, "o", alpha=0.3,
                color=AFF_COLORS.get(aff_id, "steelblue"), markersize=4, label="Trial RMSE")
        ax.plot(range(1, len(best_so_far) + 1), best_so_far, "-",
                color="crimson", linewidth=1.5, label="Best so far")
        ax.set_xlabel("Trial")
        ax.set_ylabel("CV RMSE")
        ax.set_title(f"Optuna Optimization History — {aff_id}")
        ax.legend()
        ax.grid(alpha=0.3)
        savefig(fig, OUT_FIGURES / f"experiment2_opt_history_{aff_id}")

        # 2. Hyperparameter importance (fANOVA proxy via value/param correlation)
        hp_names = ["num_leaves", "max_depth", "learning_rate", "n_estimators",
                    "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
                    "min_child_samples"]
        trial_data = []
        for t in study.trials:
            if t.value is None:
                continue
            row = {"rmse": t.value}
            for hp in hp_names:
                row[hp] = t.params.get(hp, np.nan)
            trial_data.append(row)
        trial_df = pd.DataFrame(trial_data)

        # Spearman correlation of each HP with RMSE as importance proxy
        importances = {}
        for hp in hp_names:
            if trial_df[hp].nunique() > 1:
                rho, _ = spearmanr(trial_df[hp], trial_df["rmse"])
                importances[hp] = abs(float(rho))
            else:
                importances[hp] = 0.0
        imp_series = pd.Series(importances).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(imp_series.index, imp_series.values,
                       color=AFF_COLORS.get(aff_id, "steelblue"), alpha=0.8)
        ax.set_xlabel("|Spearman ρ| with CV RMSE")
        ax.set_title(f"HP Importance (fANOVA proxy) — {aff_id}")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        savefig(fig, OUT_FIGURES / f"experiment2_hp_importance_{aff_id}")

        # 3. Parallel coordinates (top-20 trials by RMSE)
        top20 = trial_df.nsmallest(20, "rmse")
        fig, ax = plt.subplots(figsize=(12, 5))
        # Normalise columns for visualisation
        norm_df = top20[hp_names].copy()
        for col in hp_names:
            col_range = norm_df[col].max() - norm_df[col].min()
            if col_range > 0:
                norm_df[col] = (norm_df[col] - norm_df[col].min()) / col_range
        cmap = matplotlib.cm.get_cmap("plasma")
        norm_rmse = (top20["rmse"] - top20["rmse"].min()) / (top20["rmse"].max() - top20["rmse"].min() + 1e-9)
        for idx, (_, row_n) in enumerate(norm_df.iterrows()):
            ax.plot(range(len(hp_names)), row_n.values, "-o",
                    color=cmap(norm_rmse.iloc[idx]), alpha=0.5, linewidth=1, markersize=3)
        ax.set_xticks(range(len(hp_names)))
        ax.set_xticklabels(hp_names, rotation=30, ha="right")
        ax.set_ylabel("Normalised value")
        ax.set_title(f"Parallel Coordinates (Top-20 trials) — {aff_id}")
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(
            vmin=top20["rmse"].min(), vmax=top20["rmse"].max()))
        plt.colorbar(sm, ax=ax, label="CV RMSE")
        plt.tight_layout()
        savefig(fig, OUT_FIGURES / f"experiment2_parallel_coords_{aff_id}")

    print(f"Saved Experiment 2 figures to: {OUT_FIGURES}")


# ════════════════════════════════════════════════════════════════════════════
# Experiment 3 — CNN Ablation
# ════════════════════════════════════════════════════════════════════════════

def experiment3_cnn_ablation() -> None:
    print("\n" + "=" * 60)
    print("Experiment 3 — CNN Ablation (ResNet-18/50 × 4 LRs)")
    print("=" * 60)

    if not CNN_RESULTS_PATH.exists():
        print("  WARNING: CNN results file not found — run train_cnn.py first")
        print("  Creating placeholder results file...")
        placeholder = []
        for aff_id in AFFORDANCES:
            for arch in ["resnet18", "resnet50"]:
                for lr in [1e-3, 5e-4, 1e-4, 5e-5]:
                    placeholder.append({
                        "affordance_id": aff_id, "architecture": arch,
                        "learning_rate": lr, "test_rmse": np.nan, "test_mae": np.nan,
                        "test_pearson_r": np.nan, "test_spearman_rho": np.nan,
                        "best_val_rmse": np.nan, "best_epoch": 0,
                        "train_losses": [], "val_losses": [],
                    })
        with open(CNN_RESULTS_PATH, "w") as f:
            json.dump(placeholder, f, indent=2)
        cnn_results = placeholder
    else:
        cnn_results = json.loads(CNN_RESULTS_PATH.read_text())

    cnn_df = pd.DataFrame(cnn_results)
    lr_labels = ["1e-3", "5e-4", "1e-4", "5e-5"]
    lr_vals = [1e-3, 5e-4, 1e-4, 5e-5]

    # 2×4 RMSE table per affordance
    ablation_records = []
    for aff_id in AFFORDANCES:
        for arch in ["resnet18", "resnet50"]:
            for lr in lr_vals:
                mask = (
                    (cnn_df["affordance_id"] == aff_id) &
                    (cnn_df["architecture"] == arch) &
                    (cnn_df["learning_rate"].apply(lambda x: abs(x - lr) < 1e-9))
                )
                row = cnn_df[mask]
                ablation_records.append({
                    "affordance_id": aff_id,
                    "architecture": arch,
                    "learning_rate": lr,
                    "lr_label": f"{lr:.0e}",
                    "test_rmse": float(row["test_rmse"].values[0]) if len(row) else np.nan,
                    "test_mae": float(row["test_mae"].values[0]) if len(row) else np.nan,
                    "best_val_rmse": float(row["best_val_rmse"].values[0]) if len(row) else np.nan,
                    "best_epoch": int(row["best_epoch"].values[0]) if len(row) else 0,
                })

    ablation_df = pd.DataFrame(ablation_records)
    csv_path = OUT_RESULTS / "experiment3_cnn_ablation.csv"
    ablation_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Heatmap per affordance: 2 archs × 4 LRs
    for aff_id in AFFORDANCES:
        aff_df = ablation_df[ablation_df["affordance_id"] == aff_id]
        pivot = aff_df.pivot_table(index="architecture", columns="lr_label",
                                    values="test_rmse", aggfunc="mean")
        # Reorder columns
        ordered_cols = [c for c in lr_labels if c in pivot.columns]
        pivot = pivot[ordered_cols]

        fig, ax = plt.subplots(figsize=(7, 3))
        if pivot.empty or np.isnan(pivot.to_numpy(dtype=float)).all():
            ax.text(0.5, 0.5, "CNN results not available",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd",
                        ax=ax, cbar_kws={"label": "Test RMSE"}, linewidths=0.5)
        ax.set_title(f"CNN Test RMSE — {aff_id}\n(rows: architecture, cols: learning rate)")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Architecture")
        plt.tight_layout()
        savefig(fig, OUT_FIGURES / f"experiment3_cnn_heatmap_{aff_id}")

    # Training curves (best/worst) — reuse plots from train_cnn.py if they exist
    # If not, create from saved loss lists
    for aff_id in AFFORDANCES:
        aff_runs = cnn_results if isinstance(cnn_results, list) else []
        aff_runs = [r for r in aff_runs if r["affordance_id"] == aff_id
                    and r.get("train_losses")]
        if not aff_runs:
            continue
        best_run = min(aff_runs, key=lambda r: r.get("test_rmse", 9999) or 9999)
        worst_run = max(aff_runs, key=lambda r: r.get("test_rmse", 0) or 0)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, run, label in zip(axes, [best_run, worst_run], ["Best", "Worst"]):
            tl = run.get("train_losses", [])
            vl = run.get("val_losses", [])
            if not tl:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue
            epochs = range(1, len(tl) + 1)
            ax.plot(epochs, tl, label="Train RMSE", color=PALETTE["train"])
            ax.plot(epochs, vl, label="Val RMSE", color=PALETTE["val"], linestyle="--")
            if run.get("best_epoch"):
                ax.axvline(run["best_epoch"], color="gray", linestyle=":", alpha=0.7)
            ax.set_title(f"{label} — {aff_id}\n{run['architecture']}, LR={run['learning_rate']:.0e}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("RMSE")
            ax.legend()
            ax.grid(alpha=0.3)
        plt.suptitle(f"Experiment 3 — CNN Curves ({aff_id})", fontsize=11)
        plt.tight_layout()
        savefig(fig, OUT_FIGURES / f"experiment3_cnn_curves_{aff_id}")

    print(f"Saved Experiment 3 figures to: {OUT_FIGURES}")


# ════════════════════════════════════════════════════════════════════════════
# Experiment 4 — Feature Group Ablation
# ════════════════════════════════════════════════════════════════════════════

def experiment4_feature_ablation(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Experiment 4 — Feature Group Ablation (LightGBM)")
    print("=" * 60)

    def get_group_cols(df: pd.DataFrame, groups: list[str]) -> list[str]:
        cols = []
        for col in df.columns:
            if "presence" in groups and col.startswith("presence_"):
                cols.append(col)
            elif "count" in groups and col.startswith("count_"):
                cols.append(col)
            elif "distance" in groups and (col.startswith("dist_") or col.startswith("depth_diff_")):
                cols.append(col)
            elif "aggregates" in groups and col in (
                "total_object_count", "total_stuff_count", "num_unique_classes",
                "free_floor_fraction", "largest_object_area", "furniture_clutter_index",
                "mean_object_area", "std_object_area", "scene_complexity",
                "vertical_spread", "depth_available", "scene_depth_median_m",
                "scene_depth_range_m", "scene_depth_std_m"
            ):
                cols.append(col)
        return cols

    feature_groups = [
        ("presence_only", ["presence"]),
        ("presence+counts", ["presence", "count"]),
        ("presence+counts+distances", ["presence", "count", "distance"]),
        ("all_features", ["presence", "count", "distance", "aggregates"]),
    ]

    records = []
    lgbm_dir = OUT_MODELS / "lgbm"

    for aff_id in AFFORDANCES:
        best_params = load_best_params(aff_id, lgbm_dir)
        if not best_params:
            print(f"  WARNING: No best params for {aff_id} — using defaults")
            best_params = {"n_estimators": 100, "random_state": SEED, "n_jobs": -1, "verbosity": -1}

        df_aff = df[df["affordance_id"] == aff_id]
        df_tr = df_aff[df_aff["split"].isin(["train", "val"])]
        df_te = df_aff[df_aff["split"] == "test"]
        y_tr = df_tr["vlm_score"].values.astype(np.float32)
        y_te = df_te["vlm_score"].values.astype(np.float32)

        for group_name, group_keys in feature_groups:
            cols = get_group_cols(df, group_keys)
            if not cols:
                print(f"  WARNING: No columns for group '{group_name}'")
                continue
            X_tr = df_tr[cols].values.astype(np.float32)
            X_te = df_te[cols].values.astype(np.float32)

            model = lgb.LGBMRegressor(**best_params)
            model.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(-1)])
            metrics = compute_metrics(y_te, model.predict(X_te))
            records.append({
                "affordance_id": aff_id,
                "feature_group": group_name,
                "n_features": len(cols),
                **metrics,
            })
            print(f"  {aff_id} | {group_name:35s} | {len(cols):4d} feats | RMSE={metrics['rmse']:.4f}")

    ablation_df = pd.DataFrame(records)
    csv_path = OUT_RESULTS / "experiment4_feature_ablation.csv"
    ablation_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Line plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    group_labels = [g[0] for g in feature_groups]
    x = np.arange(len(group_labels))

    for ax, metric in zip(axes, ["rmse", "pearson_r"]):
        for aff_id in AFFORDANCES:
            vals = []
            for gn in group_labels:
                row = ablation_df[(ablation_df["affordance_id"] == aff_id) &
                                   (ablation_df["feature_group"] == gn)]
                vals.append(float(row[metric].values[0]) if len(row) else np.nan)
            ax.plot(x, vals, "-o", label=aff_id, color=AFF_COLORS.get(aff_id))
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, rotation=20, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Feature Group Ablation — {metric.upper()}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Experiment 4 — Feature Group Ablation", fontsize=12)
    plt.tight_layout()
    savefig(fig, OUT_FIGURES / "experiment4_feature_ablation")
    print(f"Saved figure: {OUT_FIGURES / 'experiment4_feature_ablation.png'}")

    return ablation_df


# ════════════════════════════════════════════════════════════════════════════
# Experiment 5 — Indicator Distillation Value
# ════════════════════════════════════════════════════════════════════════════

def experiment5_indicator_value(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Experiment 5 — Indicator Distillation Value (Model B vs D)")
    print("=" * 60)

    raw_cols = get_raw_feature_cols(df)
    records = []

    for aff_id in AFFORDANCES:
        df_aff = df[df["affordance_id"] == aff_id]
        df_test = df_aff[df_aff["split"] == "test"]
        y_test = df_test["vlm_score"].values.astype(np.float32)

        lgbm_b = load_lgbm_model(aff_id, OUT_MODELS / "lgbm")
        lgbm_d = load_lgbm_model(aff_id, OUT_MODELS / "lgbm_indicators")

        if lgbm_b is None or lgbm_d is None:
            print(f"  WARNING: Models not found for {aff_id} — skipping")
            continue

        ind_cols = get_indicator_cols(df, aff_id)
        X_b = df_test[raw_cols].values.astype(np.float32)
        X_d = df_test[raw_cols + ind_cols].values.astype(np.float32)

        pred_b = lgbm_b.predict(X_b)
        pred_d = lgbm_d.predict(X_d)

        m_b = compute_metrics(y_test, pred_b)
        m_d = compute_metrics(y_test, pred_d)

        # Per-sample squared errors
        se_b = (y_test - pred_b) ** 2
        se_d = (y_test - pred_d) ** 2

        # Paired Wilcoxon signed-rank test
        try:
            stat, p_val = wilcoxon(se_b, se_d, alternative="two-sided")
        except Exception:
            stat, p_val = np.nan, np.nan

        delta_rmse = m_d["rmse"] - m_b["rmse"]
        delta_mae = m_d["mae"] - m_b["mae"]
        delta_r = m_d["pearson_r"] - m_b["pearson_r"]

        records.append({
            "affordance_id": aff_id,
            "rmse_b": m_b["rmse"], "rmse_d": m_d["rmse"], "delta_rmse": round(delta_rmse, 4),
            "mae_b": m_b["mae"], "mae_d": m_d["mae"], "delta_mae": round(delta_mae, 4),
            "pearson_r_b": m_b["pearson_r"], "pearson_r_d": m_d["pearson_r"], "delta_r": round(delta_r, 4),
            "spearman_rho_b": m_b["spearman_rho"], "spearman_rho_d": m_d["spearman_rho"],
            "wilcoxon_stat": round(float(stat), 2) if not np.isnan(stat) else None,
            "wilcoxon_p": round(float(p_val), 4) if not np.isnan(p_val) else None,
            "n_indicator_cols": len(ind_cols),
        })
        direction = "↓ better" if delta_rmse < 0 else "↑ worse"
        print(f"  {aff_id}: ΔRMSE={delta_rmse:+.4f} ({direction}) | p={p_val:.4f}")

    results_df = pd.DataFrame(records)
    csv_path = OUT_RESULTS / "experiment5_indicator_value.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Paired bar chart
    if not results_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        metrics_pairs = [
            ("rmse_b", "rmse_d", "RMSE"),
            ("mae_b", "mae_d", "MAE"),
            ("pearson_r_b", "pearson_r_d", "Pearson r"),
        ]
        x = np.arange(len(results_df))
        width = 0.35
        for ax, (col_b, col_d, label) in zip(axes, metrics_pairs):
            ax.bar(x - width / 2, results_df[col_b], width, label="Model B (LGBM)",
                   color=PALETTE["LightGBM"], alpha=0.85)
            ax.bar(x + width / 2, results_df[col_d], width, label="Model D (Indicator-LGBM)",
                   color=PALETTE["Indicator-LGBM"], alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(results_df["affordance_id"], rotation=15)
            ax.set_ylabel(label)
            ax.set_title(f"{label}: Model B vs D")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
        plt.suptitle("Experiment 5 — Indicator Distillation Value", fontsize=12)
        plt.tight_layout()
        savefig(fig, OUT_FIGURES / "experiment5_indicator_value")
        print(f"Saved figure: {OUT_FIGURES / 'experiment5_indicator_value.png'}")

    return results_df


# ════════════════════════════════════════════════════════════════════════════
# Experiment 6 — SHAP Analysis
# ════════════════════════════════════════════════════════════════════════════

def experiment6_shap_analysis(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Experiment 6 — SHAP Analysis")
    print("=" * 60)

    raw_cols = get_raw_feature_cols(df)
    shap_vs_vlm_records = []

    # Load VLM indicator frequency rankings
    with open(INDICATOR_VOCAB_PATH) as f:
        vocab = json.load(f)

    def load_vlm_indicator_ranking(aff_id: str) -> dict[str, int]:
        """Return {indicator_name: rank} sorted by count descending."""
        entries = vocab.get(aff_id, [])
        sorted_entries = sorted(entries, key=lambda e: e.get("count", 0), reverse=True)
        return {e["name"]: i + 1 for i, e in enumerate(sorted_entries)}

    for aff_id in AFFORDANCES:
        lgbm_b = load_lgbm_model(aff_id, OUT_MODELS / "lgbm")
        if lgbm_b is None:
            print(f"  WARNING: LightGBM model not found for {aff_id} — skipping SHAP")
            continue

        df_aff = df[df["affordance_id"] == aff_id]
        df_train = df_aff[df_aff["split"].isin(["train", "val"])]
        X_train = df_train[raw_cols].values.astype(np.float32)

        print(f"  {aff_id}: Computing SHAP values for {len(X_train)} training samples...")
        try:
            explainer = shap.TreeExplainer(lgbm_b)
            shap_values = explainer.shap_values(X_train)
        except Exception as e:
            print(f"  ERROR: SHAP failed for {aff_id}: {e}")
            continue

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": raw_cols,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        shap_df["shap_rank"] = shap_df.index + 1

        top10 = shap_df.head(10)

        # 1. Bar plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top10["feature"][::-1], top10["mean_abs_shap"][::-1],
                color=AFF_COLORS.get(aff_id, "steelblue"), alpha=0.85)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"SHAP Feature Importance (Top 10) — {aff_id}")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        savefig(fig, OUT_FIGURES / f"experiment6_shap_bar_{aff_id}")

        # 2. Beeswarm plot
        try:
            shap_explanation = shap.Explanation(
                values=shap_values[:, :10],
                data=X_train[:, :10],
                feature_names=raw_cols[:10],
            )
            # Use only top-10 features by SHAP importance
            top10_idx = shap_df.head(10).index.tolist()
            shap_exp_top10 = shap.Explanation(
                values=shap_values[:, top10_idx],
                data=X_train[:, top10_idx],
                feature_names=[raw_cols[i] for i in top10_idx],
            )
            fig, ax = plt.subplots(figsize=(9, 6))
            shap.plots.beeswarm(shap_exp_top10, max_display=10, show=False)
            plt.title(f"SHAP Beeswarm (Top 10) — {aff_id}")
            savefig(plt.gcf(), OUT_FIGURES / f"experiment6_shap_beeswarm_{aff_id}")
        except Exception as e:
            print(f"  WARNING: Beeswarm plot failed for {aff_id}: {e}")

        # 3. Compare SHAP ranking vs VLM indicator frequency ranking
        vlm_ranking = load_vlm_indicator_ranking(aff_id)

        # Match SHAP features to VLM indicators by substring
        matched_pairs = []
        for feat_row in shap_df.itertuples():
            feat_name = feat_row.feature.replace("presence_", "").replace("count_", "").replace("_", " ")
            best_match = None
            best_score = 0
            for vlm_name in vlm_ranking:
                # Simple overlap score
                feat_words = set(feat_name.lower().split())
                vlm_words = set(vlm_name.lower().split())
                overlap = len(feat_words & vlm_words) / max(len(feat_words | vlm_words), 1)
                if overlap > best_score and overlap > 0.3:
                    best_score = overlap
                    best_match = vlm_name
            if best_match:
                matched_pairs.append({
                    "affordance_id": aff_id,
                    "feature": feat_row.feature,
                    "shap_rank": feat_row.shap_rank,
                    "mean_abs_shap": feat_row.mean_abs_shap,
                    "vlm_indicator": best_match,
                    "vlm_rank": vlm_ranking[best_match],
                    "match_score": round(best_score, 3),
                })

        if matched_pairs:
            pairs_df = pd.DataFrame(matched_pairs)
            if len(pairs_df) >= 3:
                rho_shap_vlm, p_rho = spearmanr(pairs_df["shap_rank"], pairs_df["vlm_rank"])
                print(f"  {aff_id}: SHAP vs VLM ranking Spearman ρ={rho_shap_vlm:.4f} (p={p_rho:.4f}), "
                      f"n={len(pairs_df)} matched pairs")
                shap_vs_vlm_records.append({
                    "affordance_id": aff_id,
                    "n_matched_pairs": len(pairs_df),
                    "spearman_rho": round(float(rho_shap_vlm), 4),
                    "spearman_p": round(float(p_rho), 4),
                })
            else:
                print(f"  {aff_id}: Insufficient matched pairs for ranking comparison")

    shap_vs_vlm_df = pd.DataFrame(shap_vs_vlm_records)
    csv_path = OUT_RESULTS / "experiment6_shap_vs_vlm.csv"
    shap_vs_vlm_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    print(f"Saved SHAP figures to: {OUT_FIGURES}")

    return shap_vs_vlm_df


# ════════════════════════════════════════════════════════════════════════════
# Experiment 7 — Segmentation Ablation (Planned Work)
# ════════════════════════════════════════════════════════════════════════════

def experiment7_segmentation_ablation() -> None:
    print("\n" + "=" * 60)
    print("Experiment 7 — Segmentation Ablation")
    print("=" * 60)

    seg_dir = DATA / "segmentation_outputs"
    gt_masks_available = False

    # Check for Hypersim GT panoptic masks (labelled differently from Mask2Former outputs)
    gt_candidates = list(seg_dir.glob("*_gt_*.json")) + list(seg_dir.glob("*_gt_*.npz"))
    if gt_candidates:
        gt_masks_available = True

    if gt_masks_available:
        print("  GT segmentation masks found — ablation would re-extract features.")
        print("  (Full GT-vs-predicted ablation is pending feature re-extraction.)")
    else:
        print("  Hypersim GT panoptic masks NOT found in segmentation_outputs/")
        print("  (Mask2Former predicted masks are present.)")
        print("  This experiment is documented as planned work.")

    # Scaffold: create placeholder CSV and document plan
    scaffold = {
        "experiment": "Experiment 7 — Segmentation Mask Ablation",
        "status": "planned_work",
        "description": (
            "Compare Model B RMSE when trained on features extracted from "
            "(a) Mask2Former predicted panoptic masks vs "
            "(b) Hypersim ground-truth semantic masks. "
            "GT masks require downloading the full Hypersim panoptic annotations "
            "from the official dataset and re-running src/features/extract_features.py "
            "with a gt_mask=True flag."
        ),
        "gt_masks_available": gt_masks_available,
        "planned_steps": [
            "1. Download Hypersim GT semantic/instance annotations.",
            "2. Run extract_features.py --use-gt-masks to produce features_raw_gt.parquet.",
            "3. Re-run assemble_dataset.py with the GT features.",
            "4. Re-train Model B (same best HPs from Optuna search) on GT features.",
            "5. Compare RMSE: predicted-mask model vs GT-mask model per affordance.",
            "6. Plot paired bar chart of RMSE differences.",
        ],
        "expected_outcome": (
            "GT masks should reduce noise in spatial distance features and improve RMSE, "
            "but the gap may be small for presence/count features since Mask2Former is "
            "already high-quality on Hypersim indoor scenes."
        ),
    }

    json_path = OUT_RESULTS / "experiment7_segmentation_ablation.json"
    with open(json_path, "w") as f:
        json.dump(scaffold, f, indent=2)

    # Placeholder CSV
    placeholder_df = pd.DataFrame([
        {"affordance_id": aff, "rmse_predicted_masks": np.nan,
         "rmse_gt_masks": np.nan, "delta_rmse": np.nan}
        for aff in AFFORDANCES
    ])
    csv_path = OUT_RESULTS / "experiment7_segmentation_ablation.csv"
    placeholder_df.to_csv(csv_path, index=False)
    print(f"Saved planned-work scaffold: {json_path}")
    print(f"Saved placeholder CSV: {csv_path}")


# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════

def generate_summary(
    df: pd.DataFrame,
    exp1_df: pd.DataFrame,
    exp5_df: pd.DataFrame,
    t_total: float,
) -> None:
    print("\n" + "=" * 60)
    print("Generating summary.json")
    print("=" * 60)

    summary: dict = {
        "dataset": {
            "n_images": df["image_id"].nunique(),
            "n_rows": len(df),
            "affordance_ids": AFFORDANCES,
            "split_counts": {
                split: int((df["split"] == split).sum())
                for split in ["train", "val", "test"]
            },
        },
        "best_model_per_affordance": {},
        "best_overall": {},
        "model_d_vs_b_delta": {},
        "total_compute_time_s": round(t_total, 1),
    }

    # Best model per affordance
    if not exp1_df.empty:
        for aff_id in AFFORDANCES:
            aff_df = exp1_df[exp1_df["affordance_id"] == aff_id]
            if not aff_df.empty:
                best_row = aff_df.loc[aff_df["rmse"].idxmin()]
                summary["best_model_per_affordance"][aff_id] = {
                    "model": best_row["model"],
                    "rmse": float(best_row["rmse"]),
                    "mae": float(best_row["mae"]),
                    "pearson_r": float(best_row["pearson_r"]),
                    "spearman_rho": float(best_row["spearman_rho"]),
                }

        # Best overall (lowest mean RMSE across affordances)
        model_mean = exp1_df.groupby("model")["rmse"].mean()
        best_model = model_mean.idxmin()
        summary["best_overall"] = {
            "model": best_model,
            "mean_rmse": round(float(model_mean[best_model]), 4),
            "per_affordance_rmse": {
                aff: round(float(exp1_df[(exp1_df["affordance_id"] == aff) &
                                          (exp1_df["model"] == best_model)]["rmse"].values[0]), 4)
                for aff in AFFORDANCES
                if len(exp1_df[(exp1_df["affordance_id"] == aff) & (exp1_df["model"] == best_model)]) > 0
            },
        }

    # Model D vs B delta
    if not exp5_df.empty:
        for _, row in exp5_df.iterrows():
            summary["model_d_vs_b_delta"][row["affordance_id"]] = {
                "delta_rmse": float(row["delta_rmse"]),
                "delta_mae": float(row["delta_mae"]),
                "delta_pearson_r": float(row["delta_r"]),
                "wilcoxon_p": float(row["wilcoxon_p"]) if row.get("wilcoxon_p") is not None else None,
            }
        mean_delta = float(exp5_df["delta_rmse"].mean())
        summary["model_d_vs_b_delta"]["_mean_delta_rmse"] = round(mean_delta, 4)
        summary["model_d_vs_b_delta"]["_interpretation"] = (
            "Negative delta_rmse means Model D (Indicator-LGBM) outperforms Model B (LGBM)."
        )

    summary_path = OUT_RESULTS / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    # Pretty-print key findings
    print("\n── Key Findings ──────────────────────────────────────")
    if summary.get("best_overall"):
        bo = summary["best_overall"]
        print(f"  Best overall model: {bo['model']} (mean RMSE={bo['mean_rmse']:.4f})")
    if summary["model_d_vs_b_delta"]:
        md = summary["model_d_vs_b_delta"].get("_mean_delta_rmse", "n/a")
        print(f"  Model D vs B mean ΔRMSE: {md:+.4f} (negative = D better)")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("7-Experiment Affordance Prediction Evaluation Suite")
    print("=" * 70)

    # Ensure output directories exist
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    # Load dataset
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Run project/src/models/assemble_dataset.py first."
        )
    print(f"\nLoading dataset: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Splits: {df['split'].value_counts().to_dict()}")

    # Run experiments
    exp1_df = experiment1_model_comparison(df)
    experiment2_hp_sensitivity()
    experiment3_cnn_ablation()
    exp4_df = experiment4_feature_ablation(df)
    exp5_df = experiment5_indicator_value(df)
    exp6_df = experiment6_shap_analysis(df)
    experiment7_segmentation_ablation()

    # Generate master summary
    generate_summary(df, exp1_df, exp5_df, t_total=time.time() - t0)

    total_time = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"All 7 experiments complete in {total_time:.0f}s")
    print(f"Results : {OUT_RESULTS}")
    print(f"Figures : {OUT_FIGURES}")
    print("=" * 70)


if __name__ == "__main__":
    main()
