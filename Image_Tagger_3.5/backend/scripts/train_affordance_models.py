#!/usr/bin/env python3
"""
Train LightGBM affordance models from the Hypersim pilot dataset.

Reads the assembled pilot_dataset.parquet from the research project,
trains one LightGBM regressor per affordance using raw features (310-dim),
and saves the models + metadata into:
  backend/science/data/affordance_models/{aff_id}/lgbm_model.pkl
  backend/science/data/affordance_models/feature_columns.json

Usage:
  python -m backend.scripts.train_affordance_models
  # or
  python backend/scripts/train_affordance_models.py

Requires: lightgbm, pandas, scikit-learn, optuna
"""

import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────

# When run from Image_Tagger_3.5/ root
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
MODEL_OUT_DIR = BACKEND_DIR / "science" / "data" / "affordance_models"

# The research project dataset
PROJECT_ROOT = BACKEND_DIR.parent.parent / "project"
DATASET_PATH = PROJECT_ROOT / "data" / "assembled_dataset" / "pilot_dataset.parquet"

AFFORDANCES = ["L059", "L079", "L091", "L130", "L141"]
N_TRIALS = 50  # Reduced from 100 for faster training
N_FOLDS = 5
SEED = 42


def get_raw_feature_cols(df: pd.DataFrame) -> list:
    cols = []
    for col in df.columns:
        if col.startswith(("presence_", "count_", "dist_", "depth_diff_")):
            cols.append(col)
        elif col in (
            "total_object_count", "total_stuff_count",
            "num_unique_classes", "free_floor_fraction",
            "largest_object_area", "furniture_clutter_index",
            "mean_object_area", "std_object_area",
            "scene_complexity", "vertical_spread",
            "depth_available", "scene_depth_median_m",
            "scene_depth_range_m", "scene_depth_std_m",
        ):
            cols.append(col)
    return cols


def train_with_optuna(X_train, y_train, X_val, y_val):
    """Full Optuna HPO — used when optuna is available."""
    X_search = np.vstack([X_train, X_val])
    y_search = np.concatenate([y_train, y_val])
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    def objective(trial):
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
        for tr_idx, val_idx in kf.split(X_search):
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_search[tr_idx], y_search[tr_idx],
                eval_set=[(X_search[val_idx], y_search[val_idx])],
                callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)],
            )
            preds = model.predict(X_search[val_idx])
            rmses.append(np.sqrt(mean_squared_error(y_search[val_idx], preds)))
        return float(np.mean(rmses))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({"random_state": SEED, "n_jobs": -1, "verbosity": -1})

    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_search, y_search, callbacks=[lgb.log_evaluation(-1)])
    return final_model, best_params, study.best_value


def train_simple(X_train, y_train, X_val, y_val):
    """Fallback training without Optuna — uses reasonable defaults."""
    X_search = np.vstack([X_train, X_val])
    y_search = np.concatenate([y_train, y_val])

    params = {
        "num_leaves": 63,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_samples": 10,
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_search, y_search, callbacks=[lgb.log_evaluation(-1)])
    return model, params, None


def compute_metrics(y_true, y_pred):
    from scipy.stats import pearsonr, spearmanr
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r, _ = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "pearson_r": round(float(r), 4),
        "spearman_rho": round(float(rho), 4),
    }


def main():
    t0 = time.time()
    print("=" * 60)
    print("Affordance Model Training for Image Tagger 3.5")
    print("=" * 60)

    if not DATASET_PATH.exists():
        print(f"\nERROR: Dataset not found at {DATASET_PATH}")
        print("Make sure the research project data is available.")
        return 1

    print(f"\nLoading dataset: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    print(f"  Shape: {df.shape}")

    raw_cols = get_raw_feature_cols(df)
    print(f"  Raw feature columns: {len(raw_cols)}")

    # Save feature column order
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_cols_path = MODEL_OUT_DIR / "feature_columns.json"
    with open(feature_cols_path, "w") as f:
        json.dump(raw_cols, f, indent=2)
    print(f"  Saved feature columns: {feature_cols_path}")

    use_optuna = HAS_OPTUNA
    if use_optuna:
        print(f"\n  Using Optuna HPO ({N_TRIALS} trials, {N_FOLDS}-fold CV)")
    else:
        print("\n  Optuna not available; using default hyperparameters")

    all_results = []
    for aff_id in AFFORDANCES:
        print(f"\n{'─' * 50}")
        print(f"  Training: {aff_id}")

        df_aff = df[df["affordance_id"] == aff_id].copy()
        df_train = df_aff[df_aff["split"] == "train"]
        df_val = df_aff[df_aff["split"] == "val"]
        df_test = df_aff[df_aff["split"] == "test"]

        print(f"  Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

        X_train = df_train[raw_cols].values.astype(np.float32)
        y_train = df_train["vlm_score"].values.astype(np.float32)
        X_val = df_val[raw_cols].values.astype(np.float32)
        y_val = df_val["vlm_score"].values.astype(np.float32)
        X_test = df_test[raw_cols].values.astype(np.float32)
        y_test = df_test["vlm_score"].values.astype(np.float32)

        t_start = time.time()
        if use_optuna:
            model, best_params, cv_rmse = train_with_optuna(X_train, y_train, X_val, y_val)
            print(f"  Best CV RMSE: {cv_rmse:.4f} ({time.time() - t_start:.0f}s)")
        else:
            model, best_params, cv_rmse = train_simple(X_train, y_train, X_val, y_val)
            print(f"  Trained in {time.time() - t_start:.1f}s")

        test_metrics = compute_metrics(y_test, model.predict(X_test))
        print(
            f"  Test RMSE: {test_metrics['rmse']:.4f} | "
            f"MAE: {test_metrics['mae']:.4f} | "
            f"r: {test_metrics['pearson_r']:.4f} | "
            f"rho: {test_metrics['spearman_rho']:.4f}"
        )

        # Save model
        aff_dir = MODEL_OUT_DIR / aff_id
        aff_dir.mkdir(parents=True, exist_ok=True)

        with open(aff_dir / "lgbm_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(aff_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
        with open(aff_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

        all_results.append({
            "affordance_id": aff_id,
            "test_metrics": test_metrics,
            "n_features": len(raw_cols),
        })

    # Summary
    print(f"\n{'=' * 60}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Affordance':<10} {'RMSE':>8} {'MAE':>8} {'r':>8} {'rho':>8}")
    print("-" * 44)
    for r in all_results:
        m = r["test_metrics"]
        print(f"{r['affordance_id']:<10} {m['rmse']:>8.4f} {m['mae']:>8.4f} "
              f"{m['pearson_r']:>8.4f} {m['spearman_rho']:>8.4f}")

    # Save summary
    with open(MODEL_OUT_DIR / "training_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nModels saved to: {MODEL_OUT_DIR}")
    print(f"Total time: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
