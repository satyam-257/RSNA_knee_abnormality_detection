"""
06_four_pillar_ensemble.py
==========================
Orchestrates clinical finding-specific percentile rank ensembling across
all 4 frontier model families:
  1. CoAtNet Consensus (4-Way)
  2. ConvNeXt-V2 Multi-Plane Pool
  3. DINOv2 Self-Supervised ViT
  4. RadImageNet ResNet-50 Specialized Heads
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import LABELS
from src.models.pooling import FOUR_PILLAR_WEIGHTS, blend_four_pillars, rank_percentile


def run_four_pillar_ensemble(
    coat_csv: Path,
    conv_csv: Path,
    dino_csv: Path,
    rad_csv: Path,
    output_path: Path,
    weights: dict = FOUR_PILLAR_WEIGHTS
):
    print("=" * 65)
    print("RSNA KNEE FRONTIER 4-PILLAR RANK ENSEMBLE")
    print("=" * 65)

    files = {
        "CoAtNet Consensus": coat_csv,
        "ConvNeXt-V2 Pool": conv_csv,
        "DINOv2 ViT": dino_csv,
        "RadImageNet ResNet50": rad_csv
    }

    dfs = {}
    for name, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing required prediction file for {name}: {p}")
        df = pd.read_csv(p)
        print(f"  [{name:<20}] {p.name:<25} ({len(df)} studies)")
        dfs[name] = df

    base_uids = dfs["CoAtNet Consensus"]["StudyInstanceUID"].astype(str).tolist()

    # Align study UIDs
    matrices = {}
    for name, df in dfs.items():
        uids = df["StudyInstanceUID"].astype(str).tolist()
        if uids != base_uids:
            df = df.set_index("StudyInstanceUID").reindex(base_uids).reset_index()
        matrices[name] = df[LABELS].values

    print("\nClinical Finding Weights (CoAtNet / ConvNeXt / DINO / RadImageNet):")
    for lbl in LABELS:
        w_c, w_x, w_d, w_r = weights.get(lbl, (0.35, 0.25, 0.25, 0.15))
        print(f"  {lbl:<18}: {w_c*100:4.1f}% | {w_x*100:4.1f}% | {w_d*100:4.1f}% | {w_r*100:4.1f}%")

    blended = blend_four_pillars(
        p_coat=matrices["CoAtNet Consensus"],
        p_conv=matrices["ConvNeXt-V2 Pool"],
        p_dino=matrices["DINOv2 ViT"],
        p_rad=matrices["RadImageNet ResNet50"],
        weights=weights,
        target_names=LABELS
    )

    out_df = pd.DataFrame(blended, columns=LABELS)
    out_df.insert(0, "StudyInstanceUID", base_uids)

    # Sanity verification
    assert out_df[LABELS].isna().sum().sum() == 0, "Submission contains NaN values!"
    assert np.isfinite(out_df[LABELS].values).all(), "Submission contains non-finite values!"
    assert (out_df[LABELS].values >= 0.0).all() and (out_df[LABELS].values <= 1.0).all(), "Values outside [0, 1]!"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully wrote verified 4-pillar submission to: {output_path}")
    print(f"Shape: {out_df.shape}")
    print("\nPreview:")
    print(out_df.head(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble predictions from 4 distinct model families.")
    parser.add_argument("--coat", required=True, help="Path to CoAtNet consensus CSV")
    parser.add_argument("--conv", required=True, help="Path to ConvNeXt-V2 CSV")
    parser.add_argument("--dino", required=True, help="Path to DINOv2 ViT CSV")
    parser.add_argument("--rad",  required=True, help="Path to RadImageNet CSV")
    parser.add_argument("--output", default="submission_4pillar.csv", help="Output submission path")
    args = parser.parse_args()

    run_four_pillar_ensemble(
        coat_csv=Path(args.coat),
        conv_csv=Path(args.conv),
        dino_csv=Path(args.dino),
        rad_csv=Path(args.rad),
        output_path=Path(args.output)
    )
