import io
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import TRAIN_CSV, LABELS, OUTPUT_DIR, PSEUDO_LABELED_TRAIN_CSV
from src.metrics import compute_competition_metric
from src.text.multilingual_extractor import MultilingualKneeReportExtractor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    print("=" * 60)
    print("STEP 1: BENCHMARK REPORT EXTRACTOR ON 58 GROUND-TRUTH STUDIES")
    print("=" * 60)

    df_train = pd.read_csv(TRAIN_CSV)
    labeled_mask = df_train['ACL'].notnull()
    df_labeled = df_train[labeled_mask].copy().reset_index(drop=True)

    print(f"Total train studies: {len(df_train)}")
    print(f"Expert-labeled studies: {len(df_labeled)}")

    extractor = MultilingualKneeReportExtractor()

    # Predict on 58 ground truth cases
    preds_gt = []
    for idx, row in df_labeled.iterrows():
        p = extractor.extract_from_report(row['Report'])
        preds_gt.append(p)

    df_preds_gt = pd.DataFrame(preds_gt)

    macro_auc, per_class_auc = compute_competition_metric(
        y_true=df_labeled[LABELS].values,
        y_pred=df_preds_gt[LABELS].values,
        label_names=LABELS
    )

    print("\n" + "-" * 40)
    print(f"BENCHMARK MACRO ROC-AUC: {macro_auc:.4f}")
    print("-" * 40)
    print(f"{'Label':<20s} | {'ROC-AUC':<8s} | {'Positives':<10s}")
    print("-" * 40)
    for col in LABELS:
        auc_val = per_class_auc.get(col, 0.0)
        pos_cnt = int(df_labeled[col].sum())
        print(f"{col:<20s} | {auc_val:8.4f} | {pos_cnt}/{len(df_labeled)}")

    print("\n" + "=" * 60)
    print("STEP 2: GENERATING WEAK SUPERVISION PSEUDO-LABELS FOR ALL 4,407 STUDIES")
    print("=" * 60)

    all_preds = []
    for idx, row in df_train.iterrows():
        if pd.notnull(row['ACL']):
            # For the 58 ground-truth cases, preserve exact expert labels
            gt_dict = {col: float(row[col]) for col in LABELS}
            gt_dict['is_expert_labeled'] = 1
            all_preds.append(gt_dict)
        else:
            p = extractor.extract_from_report(row['Report'])
            p['is_expert_labeled'] = 0
            all_preds.append(p)

    df_pseudo = pd.DataFrame(all_preds)
    for col in LABELS:
        df_train[col + '_pseudo'] = df_pseudo[col]
    df_train['is_expert_labeled'] = df_pseudo['is_expert_labeled']

    PSEUDO_LABELED_TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(PSEUDO_LABELED_TRAIN_CSV, index=False)
    print(f"\nSaved pseudo-labeled dataset to:\n  {PSEUDO_LABELED_TRAIN_CSV}")
    print("Shape:", df_train.shape)

    print("\nSummary of Pseudo-Labeled Positive Rates (threshold >= 0.5):")
    for col in LABELS:
        pos_rate = (df_train[col + '_pseudo'] >= 0.5).mean()
        print(f"  {col:<20s}: {pos_rate * 100:5.1f}%")

if __name__ == "__main__":
    main()
