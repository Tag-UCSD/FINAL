#!/usr/bin/env python3
"""
Model D — Indicator-Distilled LightGBM.
Same Optuna protocol as Model B, but adds indicator-polarity binary features.
"""

import json
import pickle
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "project" / "data"
OUT_MODELS = ROOT / "project" / "outputs" / "models"
OUT_RESULTS = ROOT / "project" / "outputs" / "results"

DATASET_PATH = DATA / "assembled_dataset" / "pilot_dataset.parquet"

AFFORDANCES = ["L059", "L079", "L091", "L130", "L141"]
N_TRIALS = 100
N_FOLDS = 5
SEED = 42


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
    """Return indicator columns for the given affordance."""
    prefix = f"ind_{aff_id}_"
    return [c for c in df.columns if c.startswith(prefix)]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r, _ = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    return {"rmse": round(rmse, 4), "mae": round(mae, 4),
            "pearson_r": round(float(r), 4), "spearman_rho": round(float(rho), 4)}


def make_objective(X_train: np.ndarray, y_train: np.ndarray) -> callable:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "random_state": SEED,
            "n_jobs": -1,
            "verbosity": -1,
        }
        rmses = []
        for tr_idx, val_idx in kf.split(X_train):
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train[tr_idx], y_train[tr_idx],
                eval_set=[(X_train[val_idx], y_train[val_idx])],
                callbacks=[lgb.early_stopping(20, verbose=False),
                            lgb.log_evaluation(-1)],
            )
            preds = model.predict(X_train[val_idx])
            rmses.append(np.sqrt(mean_squared_error(y_train[val_idx], preds)))
        return float(np.mean(rmses))

    return objective


def train_affordance(
    aff_id: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
) -> dict:
    print(f"\n{'─' * 50}")
    print(f"  Affordance: {aff_id}")
    print(f"  Features: {len(feature_cols)} (raw + indicators)")
    print(f"  Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train["vlm_score"].values.astype(np.float32)
    X_val = df_val[feature_cols].values.astype(np.float32)
    y_val = df_val["vlm_score"].values.astype(np.float32)
    X_test = df_test[feature_cols].values.astype(np.float32)
    y_test = df_test["vlm_score"].values.astype(np.float32)

    X_search = np.vstack([X_train, X_val])
    y_search = np.concatenate([y_train, y_val])

    t_search = time.time()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(make_objective(X_search, y_search), n_trials=N_TRIALS, show_progress_bar=False)
    elapsed_search = time.time() - t_search

    best_params = study.best_params
    best_params["random_state"] = SEED
    best_params["n_jobs"] = -1
    best_params["verbosity"] = -1
    print(f"  Best CV RMSE: {study.best_value:.4f} ({elapsed_search:.0f}s)")

    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_search, y_search, callbacks=[lgb.log_evaluation(-1)])

    test_metrics = compute_metrics(y_test, final_model.predict(X_test))
    val_metrics = compute_metrics(y_val, final_model.predict(X_val))
    print(f"  Test RMSE: {test_metrics['rmse']:.4f} | MAE: {test_metrics['mae']:.4f} "
          f"| r: {test_metrics['pearson_r']:.4f} | ρ: {test_metrics['spearman_rho']:.4f}")

    aff_dir = out_dir / aff_id
    aff_dir.mkdir(parents=True, exist_ok=True)
    with open(aff_dir / "lgbm_indicators_model.pkl", "wb") as f:
        pickle.dump(final_model, f)
    with open(aff_dir / "optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)
    with open(aff_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": final_model.booster_.feature_importance(importance_type="gain"),
        "importance_split": final_model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    fi_df.to_csv(aff_dir / "feature_importance.csv", index=False)

    n_ind = len([c for c in feature_cols if c.startswith(f"ind_{aff_id}_")])
    return {
        "affordance_id": aff_id,
        "best_cv_rmse": round(study.best_value, 4),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_params": best_params,
        "n_features_total": len(feature_cols),
        "n_indicator_features": n_ind,
        "search_time_s": round(elapsed_search, 1),
    }


def main() -> None:
    t0 = time.time()
    print("=" * 60)
    print("Model D — Indicator-Distilled LightGBM Training")
    print("=" * 60)

    OUT_MODELS_D = OUT_MODELS / "lgbm_indicators"
    OUT_MODELS_D.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading dataset: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    print(f"  Shape: {df.shape}")

    raw_cols = get_raw_feature_cols(df)
    print(f"  Raw feature columns: {len(raw_cols)}")

    all_results = []
    for aff_id in AFFORDANCES:
        df_aff = df[df["affordance_id"] == aff_id].copy()
        ind_cols = get_indicator_cols(df, aff_id)
        feature_cols = raw_cols + ind_cols
        print(f"\n  {aff_id}: {len(ind_cols)} indicator columns → {len(feature_cols)} total")

        df_train = df_aff[df_aff["split"] == "train"]
        df_val = df_aff[df_aff["split"] == "val"]
        df_test = df_aff[df_aff["split"] == "test"]

        result = train_affordance(
            aff_id, df_train, df_val, df_test, feature_cols,
            out_dir=OUT_MODELS_D,
        )
        all_results.append(result)

    summary_path = OUT_RESULTS / "lgbm_indicators_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    print("\nModel D — Test Results Summary")
    print(f"{'Affordance':<10} {'RMSE':>8} {'MAE':>8} {'Pearson r':>10} {'Spearman ρ':>11} {'#Ind':>6}")
    print("-" * 58)
    for r in all_results:
        m = r["test_metrics"]
        print(f"{r['affordance_id']:<10} {m['rmse']:>8.4f} {m['mae']:>8.4f} "
              f"{m['pearson_r']:>10.4f} {m['spearman_rho']:>11.4f} {r['n_indicator_features']:>6}")

    print(f"\nTotal training time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
