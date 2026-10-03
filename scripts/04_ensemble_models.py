"""
04_ensemble_models.py
Orchestrates finding-specific percentile rank ensembling across multiple models
(e.g., CoAtNet, DINOv2, and ViT/Raptors) for the RSNA Knee challenge.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import LABELS
from src.models.pooling import blend_predictions, rank_percentile, COATNET_ENSEMBLE_WEIGHTS

def ensemble_submissions(
    sub_paths: list,
    output_path: Path,
    weights: dict = COATNET_ENSEMBLE_WEIGHTS
):
    print("=" * 60)
    print("RSNA KNEE MULTI-MODEL RANK-PERCENTILE ENSEMBLING")
    print("=" * 60)

    dfs = [pd.read_csv(p) for p in sub_paths]
    print(f"Loaded {len(dfs)} submission files:")
    for p, df in zip(sub_paths, dfs):
        print(f"  - {p.name}: {df.shape[0]} studies")

    # Verify study IDs match across all submissions
    base_uids = dfs[0]['StudyInstanceUID'].astype(str).tolist()
    for i, df in enumerate(dfs[1:], 1):
        uids = df['StudyInstanceUID'].astype(str).tolist()
        if uids != base_uids:
            # Reindex to match base
            dfs[i] = df.set_index('StudyInstanceUID').reindex(base_uids).reset_index()

    # Extract prediction matrices
    preds = [df[LABELS].values for df in dfs]

    print("\nApplying finding-specific percentile rank blend...")
    print("Weights (Model 0 vs Model 1):")
    for lbl, w in weights.items():
        print(f"  {lbl:<18}: {w*100:5.1f}% / {(1-w)*100:5.1f}%")

    blended_matrix = blend_predictions(preds, weights=weights, target_names=LABELS)

    out_df = pd.DataFrame(blended_matrix, columns=LABELS)
    out_df.insert(0, 'StudyInstanceUID', base_uids)

    # Verification: check no NaNs, finite values, proper column format
    assert out_df[LABELS].isna().sum().sum() == 0, "Submission contains NaNs!"
    assert np.isfinite(out_df[LABELS].values).all(), "Submission contains non-finite numbers!"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully wrote verified ensemble submission to: {output_path}")
    print(f"Shape: {out_df.shape}")
    print("\nSample preview:")
    print(out_df.head(3).to_string(index=False))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ensemble multiple knee predictions.")
    parser.add_argument("--inputs", nargs="+", help="Paths to prediction CSV files.")
    parser.add_argument("--output", default="submission.csv", help="Path to write blended submission.")

    args = parser.parse_args()

    if args.inputs and len(args.inputs) >= 2:
        in_paths = [Path(p) for p in args.inputs]
        out_p = Path(args.output)
        ensemble_submissions(in_paths, out_p)
    else:
        print("Usage: python scripts/04_ensemble_models.py --inputs model1.csv model2.csv --output submission.csv")
