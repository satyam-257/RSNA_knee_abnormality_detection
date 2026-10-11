import io
import sys
import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    TRAIN_SERIES_CSV, PSEUDO_LABELED_TRAIN_CSV, OUTPUT_DIR,
    LABELS, SEED, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    NUM_EPOCHS, NUM_SLICES_PER_SERIES, IMG_SIZE
)
from src.metrics import compute_competition_metric
from src.data.dataset import KneeMRIDataset
from src.models.baseline_model import KneeMultiView25DModel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train Baseline 2.5D Knee Model")
    parser.add_argument("--backbone", type=str, default="resnet34", help="timm backbone name")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (per GPU/device)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num_slices", type=int, default=12, help="Number of slices per plane")
    parser.add_argument("--debug", action="store_true", help="Debug mode with small subset")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pseudo-labeled training metadata
    if not PSEUDO_LABELED_TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Pseudo-labeled file {PSEUDO_LABELED_TRAIN_CSV} not found! "
            "Please run scripts/01_benchmark_and_extract_reports.py first."
        )

    df_train = pd.read_csv(PSEUDO_LABELED_TRAIN_CSV)
    df_series = pd.read_csv(TRAIN_SERIES_CSV)

    if args.debug:
        print("[DEBUG MODE] Using first 20 samples")
        df_train = df_train.head(20).copy()

    # StratifiedGroupKFold by StudyInstanceUID — prevents patient-level data leakage
    # (Key insight from discussion/746124: plain KFold on row indices leaks patients)
    acl_col = "ACL_pseudo" if "ACL_pseudo" in df_train.columns else "ACL"
    strat_labels = (df_train[acl_col].fillna(0.5) >= 0.5).astype(int).values
    group_ids = df_train["StudyInstanceUID"].values

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(sgkf.split(df_train, strat_labels, groups=group_ids))

    train_df = df_train.iloc[train_idx].reset_index(drop=True)
    val_df = df_train.iloc[val_idx].reset_index(drop=True)

    # Sanity: verify zero patient overlap
    assert len(set(train_df["StudyInstanceUID"]) & set(val_df["StudyInstanceUID"])) == 0, \
        "Patient leakage detected between train and val folds!"

    print(f"Train studies: {len(train_df)} | Val studies: {len(val_df)}")

    # Datasets and Loaders
    train_dataset = KneeMRIDataset(
        df_studies=train_df,
        df_series=df_series,
        data_dir=BASE_DIR,
        labels=LABELS,
        num_slices=args.num_slices,
        target_size=(IMG_SIZE, IMG_SIZE),
        is_training=True
    )
    val_dataset = KneeMRIDataset(
        df_studies=val_df,
        df_series=df_series,
        data_dir=BASE_DIR,
        labels=LABELS,
        num_slices=args.num_slices,
        target_size=(IMG_SIZE, IMG_SIZE),
        is_training=False  # No augmentation during validation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Initialize Model
    model = KneeMultiView25DModel(
        backbone_name=args.backbone,
        pretrained=True,
        in_chans=1,
        num_classes=len(LABELS)
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_auc = 0.0

    print("\n" + "=" * 60)
    print("STARTING TRAINING PIPELINE")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            sag = batch["sagittal"].to(device)
            cor = batch["coronal"].to(device)
            ax = batch["axial"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            logits = model(sag, cor, ax)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        train_loss = total_loss / max(1, num_batches)

        # Validation Loop
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                sag = batch["sagittal"].to(device)
                cor = batch["coronal"].to(device)
                ax = batch["axial"].to(device)
                targets = batch["targets"].to(device)

                probs = model.predict_proba(sag, cor, ax)
                val_preds.append(probs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())

        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)

        # Round targets >= 0.5 to evaluate AUC on binary classification
        val_targets_binary = (val_targets >= 0.5).astype(int)

        val_auc, per_class = compute_competition_metric(
            y_true=val_targets_binary,
            y_pred=val_preds,
            label_names=LABELS
        )

        print(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Macro ROC-AUC: {val_auc:.4f}")
        # Per-finding AUC (key signal for diagnosing weak classes, per discussion/746124)
        col_w = max(len(k) for k in per_class)
        for lbl, auc_v in per_class.items():
            marker = " <<" if not (auc_v != auc_v) and auc_v < 0.65 else ""
            print(f"  {lbl:<{col_w}} : {auc_v:.4f}{marker}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_path = OUTPUT_DIR / f"best_{args.backbone}_knee_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best checkpoint to {save_path.name} (Macro AUC: {val_auc:.4f})")

    print("\nTraining completed! Best Validation Macro ROC-AUC:", best_val_auc)

if __name__ == "__main__":
    main()
