#!/usr/bin/env python3
"""
Model A — CNN Baseline (ResNet-18 / ResNet-50).
Predicts VLM affordance scores from raw images.
40 runs: 5 affordances × 2 architectures × 4 learning rates.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "project" / "data"
IMAGE_DIR = DATA / "hypersim_pilot_420"
DATASET_PATH = DATA / "assembled_dataset" / "pilot_dataset.parquet"
OUT_MODELS = ROOT / "project" / "outputs" / "models" / "cnn"
OUT_RESULTS = ROOT / "project" / "outputs" / "results"
OUT_FIGURES = ROOT / "project" / "outputs" / "figures"

AFFORDANCES = ["L059", "L079", "L091", "L130", "L141"]
ARCHITECTURES = ["resnet18", "resnet50"]
LEARNING_RATES = [1e-3, 5e-4, 1e-4, 5e-5]
BATCH_SIZE = 32
MAX_EPOCHS = 30
PATIENCE = 5
SEED = 42

# Figure styling
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class AffordanceImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.image_dir / row["file_path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        if self.transform:
            img = self.transform(img)
        score = torch.tensor(float(row["vlm_score"]), dtype=torch.float32)
        return img, score


def build_model(arch: str) -> nn.Module:
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model


def train_one_run(
    aff_id: str,
    arch: str,
    lr: float,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    device: torch.device,
    out_dir: Path,
) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_ds = AffordanceImageDataset(df_train, IMAGE_DIR, TRAIN_TRANSFORM)
    val_ds = AffordanceImageDataset(df_val, IMAGE_DIR, EVAL_TRANSFORM)
    test_ds = AffordanceImageDataset(df_test, IMAGE_DIR, EVAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(arch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_val_rmse = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []
    best_epoch = 0

    run_key = f"{aff_id}_{arch}_lr{lr:.0e}"
    checkpoint_path = out_dir / f"{run_key}_best.pt"

    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        epoch_loss = 0.0
        for imgs, scores in train_loader:
            imgs, scores = imgs.to(device), scores.to(device)
            optimizer.zero_grad()
            preds = model(imgs).squeeze(-1)
            loss = criterion(preds, scores)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(imgs)
        train_rmse = (epoch_loss / len(train_ds)) ** 0.5
        train_losses.append(train_rmse)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, scores in val_loader:
                imgs, scores = imgs.to(device), scores.to(device)
                preds = model(imgs).squeeze(-1)
                val_loss += criterion(preds, scores).item() * len(imgs)
        val_rmse = (val_loss / len(val_ds)) ** 0.5
        val_losses.append(val_rmse)
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # Load best checkpoint and evaluate on test
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, scores in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).squeeze(-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(scores.numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)
    test_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    test_mae = float(np.mean(np.abs(y_true - y_pred)))

    from scipy.stats import pearsonr, spearmanr
    r, _ = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)

    return {
        "affordance_id": aff_id,
        "architecture": arch,
        "learning_rate": lr,
        "best_epoch": best_epoch,
        "best_val_rmse": round(best_val_rmse, 4),
        "test_rmse": round(test_rmse, 4),
        "test_mae": round(test_mae, 4),
        "test_pearson_r": round(float(r), 4),
        "test_spearman_rho": round(float(rho), 4),
        "train_losses": [round(x, 4) for x in train_losses],
        "val_losses": [round(x, 4) for x in val_losses],
        "checkpoint_path": str(checkpoint_path),
    }


def plot_training_curves(results: list[dict], out_dir: Path) -> None:
    """Plot training curves for the best and worst run per affordance."""
    for aff_id in AFFORDANCES:
        aff_results = [r for r in results if r["affordance_id"] == aff_id]
        if not aff_results:
            continue
        best_run = min(aff_results, key=lambda r: r["test_rmse"])
        worst_run = max(aff_results, key=lambda r: r["test_rmse"])

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, run, label in zip(axes, [best_run, worst_run], ["Best", "Worst"]):
            epochs = range(1, len(run["train_losses"]) + 1)
            ax.plot(epochs, run["train_losses"], label="Train RMSE", color="#2196F3")
            ax.plot(epochs, run["val_losses"], label="Val RMSE", color="#FF5722", linestyle="--")
            ax.axvline(run["best_epoch"], color="gray", linestyle=":", alpha=0.7, label=f"Best epoch ({run['best_epoch']})")
            ax.set_title(f"{label} run — {aff_id}\n{run['architecture']}, LR={run['learning_rate']:.0e}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("RMSE")
            ax.legend()
            ax.grid(alpha=0.3)
        plt.suptitle(f"CNN Training Curves — {aff_id}", fontsize=11)
        plt.tight_layout()
        for ext in ["png", "pdf"]:
            plt.savefig(out_dir / f"cnn_curves_{aff_id}.{ext}")
        plt.close()


def main() -> None:
    t0 = time.time()
    print("=" * 60)
    print("Model A — CNN Baseline Training (40 runs)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice: {device}")

    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading dataset: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    print(f"  Shape: {df.shape}")

    all_results = []
    run_count = 0
    total_runs = len(AFFORDANCES) * len(ARCHITECTURES) * len(LEARNING_RATES)

    for aff_id in AFFORDANCES:
        df_aff = df[df["affordance_id"] == aff_id].copy()
        df_train = df_aff[df_aff["split"] == "train"]
        df_val = df_aff[df_aff["split"] == "val"]
        df_test = df_aff[df_aff["split"] == "test"]

        aff_dir = OUT_MODELS / aff_id
        aff_dir.mkdir(exist_ok=True)

        for arch in ARCHITECTURES:
            for lr in LEARNING_RATES:
                run_count += 1
                print(f"\n[{run_count}/{total_runs}] {aff_id} | {arch} | LR={lr:.0e}")
                t_run = time.time()
                result = train_one_run(
                    aff_id, arch, lr,
                    df_train, df_val, df_test,
                    device, aff_dir,
                )
                elapsed = time.time() - t_run
                print(f"  Epochs: {result['best_epoch']}/{MAX_EPOCHS} | "
                      f"Val RMSE: {result['best_val_rmse']:.4f} | "
                      f"Test RMSE: {result['test_rmse']:.4f} ({elapsed:.0f}s)")
                all_results.append(result)

    # Save all results
    results_path = OUT_RESULTS / "cnn_training_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {results_path}")

    # Plot curves
    plot_training_curves(all_results, OUT_FIGURES)
    print(f"Saved training curve figures to: {OUT_FIGURES}")

    # Print summary table
    print("\nCNN Test RMSE Summary (best per affordance)")
    print(f"{'Affordance':<10} {'Arch':<12} {'LR':>8} {'Test RMSE':>10} {'Test MAE':>10}")
    print("-" * 56)
    for aff_id in AFFORDANCES:
        aff_results = [r for r in all_results if r["affordance_id"] == aff_id]
        if aff_results:
            best = min(aff_results, key=lambda r: r["test_rmse"])
            print(f"{aff_id:<10} {best['architecture']:<12} {best['learning_rate']:>8.0e} "
                  f"{best['test_rmse']:>10.4f} {best['test_mae']:>10.4f}")

    print(f"\nTotal training time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
