#!/usr/bin/env python3
"""
Thresholded afford / does-not-afford analysis.

Defines a binary label as score >= 4 on the 1-7 scale and computes
precision / recall / F1 for the retained regression models. For the CNN
baseline, reruns only the best per-affordance configuration to recover
per-image test predictions.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "project" / "data"
OUT_RESULTS = ROOT / "project" / "outputs" / "results"
OUT_FIGURES = ROOT / "project" / "outputs" / "figures"

DATASET_PATH = DATA / "assembled_dataset" / "pilot_dataset.parquet"
RAW_JSON_DIR = DATA / "vlm_annotations" / "raw"
REVISION_PRED_PATH = OUT_RESULTS / "revision_test_predictions.csv"
IMAGE_DIR = DATA / "hypersim_pilot_420"

AFFORDANCES = ["L059", "L079", "L091", "L130", "L141"]
AFF_NAMES = {
    "L059": "Sleep",
    "L079": "Cook",
    "L091": "Computer Work",
    "L130": "Conversation",
    "L141": "Yoga/Stretching",
}
THRESHOLD = 4.0
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def compute_clf_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = THRESHOLD) -> dict[str, float]:
    y_bin = (y_true >= threshold).astype(int)
    pred_bin = (y_score >= threshold).astype(int)
    return {
        "positive_rate": float(y_bin.mean()),
        "precision": float(precision_score(y_bin, pred_bin, zero_division=0)),
        "recall": float(recall_score(y_bin, pred_bin, zero_division=0)),
        "f1": float(f1_score(y_bin, pred_bin, zero_division=0)),
    }
def make_qualitative_figure(df: pd.DataFrame, pred_df: pd.DataFrame) -> None:
    aff_id = "L091"
    merged = df[df["affordance_id"] == aff_id][["image_id", "file_path", "split", "vlm_score"]].merge(
        pred_df[pred_df["affordance_id"] == aff_id][["image_id", "pred_d"]],
        on="image_id",
        how="inner",
    )
    merged = merged[merged["split"] == "test"].copy()
    merged["abs_err"] = (merged["vlm_score"] - merged["pred_d"]).abs()

    hi = merged[merged["vlm_score"] >= 6].sort_values("abs_err").iloc[0]
    lo = merged[merged["vlm_score"] <= 2].sort_values("abs_err").iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    fig.suptitle("Qualitative L091 Computer Work examples", y=0.98, fontsize=11)

    for row, img_ax, txt_ax, title in [
        (hi, axes[0, 0], axes[0, 1], "High-scoring scene"),
        (lo, axes[1, 0], axes[1, 1], "Low-scoring scene"),
    ]:
        img = Image.open(IMAGE_DIR / row["file_path"]).convert("RGB")
        img_ax.imshow(img)
        img_ax.axis("off")
        img_ax.set_title(f"{title}\nVLM={int(row['vlm_score'])}, Model D={row['pred_d']:.2f}")

        json_path = RAW_JSON_DIR / f"{row['image_id']}_{aff_id}.json"
        with open(json_path) as f:
            data = json.load(f)
        indicators = data.get("indicators", [])[:5]
        bullet_lines = []
        for ind in indicators:
            pol = "+" if ind.get("polarity", "positive") == "positive" else "-"
            bullet_lines.append(f"{pol} {ind['name']}")
        txt_ax.axis("off")
        txt_ax.text(
            0.0, 1.0,
            "VLM indicator checklist\n" + "\n".join(bullet_lines),
            ha="left", va="top", fontsize=8,
        )

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig((OUT_FIGURES / "qualitative_l091_examples").with_suffix(f".{ext}"))
    plt.close(fig)


def main() -> None:
    df = pd.read_parquet(DATASET_PATH)
    pred_df = pd.read_csv(REVISION_PRED_PATH)
    merged = pred_df.copy()

    rows = []
    for aff_id in AFFORDANCES:
        d = merged[merged["affordance_id"] == aff_id]
        for model_col, model_name in [
            ("pred_mean", "Mean"),
            ("pred_linear", "LinearRegression"),
            ("pred_b", "LightGBM"),
            ("pred_d", "Indicator-LGBM"),
        ]:
            m = compute_clf_metrics(d["y_true"].to_numpy(dtype=float), d[model_col].to_numpy(dtype=float))
            rows.append({
                "threshold": THRESHOLD,
                "affordance_id": aff_id,
                "affordance_name": AFF_NAMES[aff_id],
                "model": model_name,
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
                "positive_rate": round(m["positive_rate"], 4),
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_RESULTS / "classification_f1_results.csv", index=False)

    macro_df = out_df.groupby("model")[["precision", "recall", "f1"]].mean().reset_index()
    macro_df.to_csv(OUT_RESULTS / "classification_f1_macro.csv", index=False)

    make_qualitative_figure(df, pred_df)


if __name__ == "__main__":
    main()
