"""
05_train_coatnet_kfold.py
=========================
Production-grade 5-fold cross-validation training for SpeedyRaptorCoAtNetModel.

Key practices from RSNA Knee Discussion (discussion/746124):
  1. StratifiedGroupKFold on StudyInstanceUID - prevents ANY patient data leakage
     between train and validation folds (plain KFold on row order is WRONG).
  2. Per-finding AUC is logged every epoch alongside the macro average.
  3. Dual data-path support: automatically detects whether running on Kaggle
     (/kaggle/input/competitions/... OR /kaggle/input/...) vs local.
  4. SWA (Stochastic Weight Averaging) over the last epochs per fold for
     free ensembling inside a single fold.
  5. Label-smoothing BCE for soft pseudo-label targets.
  6. Gradient clipping to prevent NaN divergence on sparse abnormality classes.

Usage (local):
    python scripts/05_train_coatnet_kfold.py --backbone coatnet_0_rw_224 --n_folds 5 --epochs 10

Usage (Kaggle GPU notebook):
    !python scripts/05_train_coatnet_kfold.py --backbone coatnet_0_rw_224 --n_folds 5 --epochs 10
"""

import io
import sys
import os
import argparse
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    TRAIN_SERIES_CSV, PSEUDO_LABELED_TRAIN_CSV, OUTPUT_DIR,
    LABELS, SEED, NUM_SLICES_PER_SERIES, IMG_SIZE
)
from src.metrics import compute_competition_metric
from src.data.dataset import KneeMRISlotDataset
from src.models import (
    SpeedyRaptorCoAtNetModel,
    ConvNeXtV2SlotModel,
    CoAtNetConvNeXtFusionModel
)

# Robust UTF-8 stdout (needed on Windows / Kaggle)
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Kaggle dual data-path detection (tip from Mingquan in discussion/746124)
# ---------------------------------------------------------------------------
def resolve_dicom_data_dir(base_dir: Path) -> Path:
    """
    On Kaggle, DICOMs may be mounted at either:
      - /kaggle/input/competitions/rsna-knee-abnormality-detection/
      - /kaggle/input/rsna-knee-abnormality-detection/
    Locally they live in the project BASE_DIR.
    Returns the first valid directory found.
    """
    candidates = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
        base_dir,
    ]
    for candidate in candidates:
        if candidate.exists():
            print(f"[Data Dir] Resolved DICOM root: {candidate}")
            return candidate
    print(f"[Data Dir] WARNING: No standard Kaggle paths found. Falling back to: {base_dir}")
    return base_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def label_smooth_bce(logits: torch.Tensor, targets: torch.Tensor, smoothing: float = 0.02) -> torch.Tensor:
    """
    Binary Cross-Entropy with label smoothing for pseudo-label targets.
    Replaces hard 0/1 targets with (smoothing/2, 1 - smoothing/2) to prevent
    overconfident predictions on noisy pseudo-labels.
    """
    targets = targets.clamp(smoothing / 2.0, 1.0 - smoothing / 2.0)
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


def print_fold_banner(fold: int, n_folds: int, train_n: int, val_n: int):
    print("\n" + "=" * 65)
    print(f"  FOLD {fold + 1}/{n_folds}  |  Train studies: {train_n}  |  Val studies: {val_n}")
    print("=" * 65)


def print_per_finding_auc(per_label: Dict[str, float], macro: float):
    """Pretty-print per-finding AUC in a compact aligned table."""
    col_w = max(len(k) for k in per_label)
    print(f"\n  {'Finding':<{col_w}}   AUC")
    print(f"  {'-' * col_w}   -----")
    for label in LABELS:
        val = per_label.get(label, float("nan"))
        # Flag low-AUC findings (< 0.65) for investigation
        marker = " <<" if not np.isnan(val) and val < 0.65 else ""
        print(f"  {label:<{col_w}}   {val:.4f}{marker}")
    print(f"  {'MACRO':<{col_w}}   {macro:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="5-Fold CoAtNet Training - RSNA Knee")
    parser.add_argument("--arch",       type=str,   default="coatnet", choices=["coatnet", "convnext", "hybrid_fusion"],
                        help="Model architecture: 'coatnet', 'convnext', or 'hybrid_fusion'")
    parser.add_argument("--backbone",   type=str,   default="coatnet_0_rw_224",
                        help="CoAtNet backbone name (e.g. coatnet_0_rw_224, coatnet_rmlp_2_rw_224)")
    parser.add_argument("--convnext_backbone", type=str, default="convnextv2_tiny.fcmae_ft_in22k_in1k",
                        help="ConvNeXt backbone name (e.g. convnextv2_tiny, convnextv2_base)")
    parser.add_argument("--n_folds",    type=int,   default=5)
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=2)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--num_slices", type=int,   default=16,
                        help="Number of slices sampled per anatomical slot")
    parser.add_argument("--img_size",   type=int,   default=224)
    parser.add_argument("--swa_start",  type=int,   default=8,
                        help="Epoch (1-indexed) from which SWA averaging begins")
    parser.add_argument("--smoothing",  type=float, default=0.02,
                        help="Label smoothing for pseudo-label BCE (0 = off)")
    parser.add_argument("--grad_clip",  type=float, default=5.0,
                        help="Max gradient norm (0 = off)")
    parser.add_argument("--debug",      action="store_true",
                        help="Run on first 24 studies only for quick smoke-test")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[Config] Device         : {device}")
    print(f"[Config] Backbone       : {args.backbone}")
    print(f"[Config] Folds          : {args.n_folds}")
    print(f"[Config] Epochs/fold    : {args.epochs}")
    print(f"[Config] Batch size     : {args.batch_size}")
    print(f"[Config] LR             : {args.lr}")
    print(f"[Config] Slices/slot    : {args.num_slices}")
    print(f"[Config] Image size     : {args.img_size}x{args.img_size}")
    print(f"[Config] SWA start ep.  : {args.swa_start}")
    print(f"[Config] Label smooth.  : {args.smoothing}")
    print(f"[Config] Grad clip norm : {args.grad_clip}")

    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------
    if not PSEUDO_LABELED_TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Pseudo-labeled CSV not found at {PSEUDO_LABELED_TRAIN_CSV}.\n"
            "Please run: python scripts/01_benchmark_and_extract_reports.py"
        )

    df_train = pd.read_csv(PSEUDO_LABELED_TRAIN_CSV)
    df_series = pd.read_csv(TRAIN_SERIES_CSV)

    if args.debug:
        print("\n[DEBUG] Limiting to first 24 studies.")
        df_train = df_train.head(24).copy()

    print(f"\n[Data] Total studies     : {len(df_train)}")
    if "is_expert_labeled" in df_train.columns:
        print(f"[Data] Expert-labeled    : {df_train['is_expert_labeled'].sum()}")

    dicom_root = resolve_dicom_data_dir(BASE_DIR)

    # ------------------------------------------------------------------
    # StratifiedGroupKFold - study-level patient isolation
    # Strategy: stratify on ACL (~41% prevalence, closest to 50/50)
    #           so each fold has a balanced positive/negative ratio.
    # ------------------------------------------------------------------
    acl_col = "ACL_pseudo" if "ACL_pseudo" in df_train.columns else "ACL"
    strat_labels = (df_train[acl_col].fillna(0.5) >= 0.5).astype(int).values
    group_ids    = df_train["StudyInstanceUID"].values

    sgkf  = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(df_train, strat_labels, groups=group_ids))

    # ------------------------------------------------------------------
    # Allocate OOF arrays
    # ------------------------------------------------------------------
    oof_preds   = np.full((len(df_train), len(LABELS)), np.nan, dtype=np.float32)
    oof_targets = np.full((len(df_train), len(LABELS)), np.nan, dtype=np.float32)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_ckpt_dir = OUTPUT_DIR / "fold_checkpoints"
    fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

    fold_best_aucs: List[float] = []
    last_val_preds = None   # will hold val preds from the last epoch of each fold

    for fold, (train_idx, val_idx) in enumerate(folds):
        train_df = df_train.iloc[train_idx].reset_index(drop=True)
        val_df   = df_train.iloc[val_idx].reset_index(drop=True)

        # Sanity: assert zero patient overlap
        train_uids = set(train_df["StudyInstanceUID"].astype(str))
        val_uids   = set(val_df["StudyInstanceUID"].astype(str))
        assert len(train_uids & val_uids) == 0, (
            f"FOLD LEAK in fold {fold + 1}! Shared UIDs: {train_uids & val_uids}"
        )

        print_fold_banner(fold, args.n_folds, len(train_df), len(val_df))

        # Collect val ground-truth targets
        for vi, row_idx in enumerate(val_idx):
            for j, lbl in enumerate(LABELS):
                col_j = f"{lbl}_pseudo" if f"{lbl}_pseudo" in df_train.columns else lbl
                raw = df_train.iloc[row_idx].get(col_j, np.nan)
                oof_targets[row_idx, j] = float(raw) if pd.notnull(raw) else np.nan

        # -- Datasets --
        train_ds = KneeMRISlotDataset(
            df_studies=train_df, df_series=df_series, data_dir=dicom_root,
            labels=LABELS, num_slices=args.num_slices,
            target_size=(args.img_size, args.img_size), is_training=True
        )
        val_ds = KneeMRISlotDataset(
            df_studies=val_df, df_series=df_series, data_dir=dicom_root,
            labels=LABELS, num_slices=args.num_slices,
            target_size=(args.img_size, args.img_size), is_training=True
        )

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available()
        )

        # -- Model --
        if args.arch == "coatnet":
            model = SpeedyRaptorCoAtNetModel(
                backbone_name=args.backbone,
                pretrained=True,
                in_chans=1,
                num_classes=len(LABELS)
            ).to(device)
        elif args.arch == "convnext":
            model = ConvNeXtV2SlotModel(
                backbone_name=args.convnext_backbone,
                pretrained=True,
                in_chans=1,
                num_classes=len(LABELS)
            ).to(device)
        elif args.arch == "hybrid_fusion":
            model = CoAtNetConvNeXtFusionModel(
                coatnet_name=args.backbone,
                convnext_name=args.convnext_backbone,
                pretrained=True,
                in_chans=1,
                num_classes=len(LABELS)
            ).to(device)
        else:
            raise ValueError(f"Unknown arch: {args.arch}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )

        # SWA
        swa_model  = AveragedModel(model)
        swa_sched  = SWALR(optimizer, swa_lr=5e-5, anneal_epochs=3)
        swa_active = False

        best_val_auc  = 0.0
        best_ckpt_path = fold_ckpt_dir / f"fold{fold + 1}_{args.backbone.replace('/', '_')}_best.pth"

        for epoch in range(1, args.epochs + 1):
            # ---- Train ----
            model.train()
            epoch_loss = 0.0
            n_batches  = 0
            t0 = time.time()

            for batch in train_loader:
                slot_tensors = batch["slot_tensors"].to(device)
                slot_mask    = batch["slot_mask"].to(device)
                targets      = batch["targets"].to(device)

                optimizer.zero_grad()
                logits = model(slot_tensors, slot_mask)
                loss   = label_smooth_bce(logits, targets, smoothing=args.smoothing)
                loss.backward()

                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

            # SWA or cosine scheduler step
            if epoch >= args.swa_start:
                swa_model.update_parameters(model)
                swa_sched.step()
                swa_active = True
            else:
                scheduler.step()

            train_loss = epoch_loss / max(1, n_batches)
            elapsed    = time.time() - t0

            # ---- Validate ----
            # Use SWA-averaged weights if active
            infer_model = swa_model if swa_active else model
            infer_model.eval()

            val_preds_list   = []
            val_targets_list = []

            with torch.no_grad():
                for batch in val_loader:
                    slot_tensors = batch["slot_tensors"].to(device)
                    slot_mask    = batch["slot_mask"].to(device)
                    targets      = batch["targets"].to(device)

                    logits = infer_model(slot_tensors, slot_mask)
                    probs  = torch.sigmoid(logits)

                    val_preds_list.append(probs.cpu().numpy())
                    val_targets_list.append(targets.cpu().numpy())

            val_preds   = np.vstack(val_preds_list)
            val_targets = np.vstack(val_targets_list)
            val_targets_bin = (val_targets >= 0.5).astype(int)

            val_auc, per_label_auc = compute_competition_metric(
                val_targets_bin, val_preds, LABELS
            )

            swa_tag = " [SWA]" if swa_active else ""
            print(f"\nEpoch [{epoch:02d}/{args.epochs}]{swa_tag}  "
                  f"Loss: {train_loss:.4f}  "
                  f"Val Macro AUC: {val_auc:.4f}  "
                  f"({elapsed:.0f}s)")
            print_per_finding_auc(per_label_auc, val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                inner_state = (swa_model.module.state_dict()
                               if swa_active else model.state_dict())
                torch.save({
                    "epoch":       epoch,
                    "fold":        fold + 1,
                    "backbone":    args.backbone,
                    "val_auc":     val_auc,
                    "per_label":   per_label_auc,
                    "model_state": inner_state,
                }, best_ckpt_path)
                print(f"  --> New best saved: {best_ckpt_path.name}  "
                      f"(AUC: {val_auc:.4f})")

            last_val_preds = val_preds  # keep the latest for OOF storage

        # Store OOF predictions (last-epoch val preds — SWA preds if active)
        oof_preds[val_idx] = last_val_preds
        fold_best_aucs.append(best_val_auc)
        print(f"\nFold {fold + 1} complete. Best Val Macro AUC: {best_val_auc:.4f}")

    # ------------------------------------------------------------------
    # Final OOF summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("OUT-OF-FOLD SUMMARY")
    print("=" * 65)
    for fi, auc in enumerate(fold_best_aucs):
        print(f"  Fold {fi + 1}: {auc:.4f}")
    print(f"  Mean  : {np.mean(fold_best_aucs):.4f}")
    print(f"  Std   : {np.std(fold_best_aucs):.4f}")

    # Expert-labeled subset OOF (the 58 ground-truth studies)
    if "is_expert_labeled" in df_train.columns:
        expert_mask = df_train["is_expert_labeled"].values == 1
        if expert_mask.sum() >= 2:
            exp_preds   = oof_preds[expert_mask]
            exp_targets = oof_targets[expert_mask]
            exp_bin     = np.where(np.isnan(exp_targets), np.nan,
                                   (exp_targets >= 0.5).astype(float))
            exp_auc, exp_per = compute_competition_metric(exp_bin, exp_preds, LABELS)
            print(f"\nOOF AUC on 58 expert-labeled studies: {exp_auc:.4f}")
            print_per_finding_auc(exp_per, exp_auc)

    # Save OOF predictions CSV
    oof_df = pd.DataFrame(oof_preds, columns=LABELS)
    oof_df.insert(0, "StudyInstanceUID", df_train["StudyInstanceUID"].values)
    oof_path = OUTPUT_DIR / f"oof_{args.backbone.replace('/', '_')}.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"\nOOF predictions saved -> {oof_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
