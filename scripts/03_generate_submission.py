import io
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    TEST_CSV, TEST_SERIES_CSV, SAMPLE_SUBMISSION_CSV,
    LABELS, NUM_SLICES_PER_SERIES, IMG_SIZE, OUTPUT_DIR
)
from src.data.dataset import KneeMRISlotDataset
from src.models.coatnet_model import SpeedyRaptorCoAtNetModel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    print("=" * 60)
    print("GENERATING KAGGLE SUBMISSION (WITH PERCENTILE RANKING)")
    print("=" * 60)

    df_test = pd.read_csv(TEST_CSV)
    df_test_series = pd.read_csv(TEST_SERIES_CSV)
    df_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)

    print(f"Test studies: {len(df_test)}")
    print(f"Sample submission shape: {df_sub.shape}")

    # Prior class positive rates from the 58 ground-truth benchmark
    prior_rates = {
        'Effusion': 0.60,
        'Synovitis': 0.46,
        'Medial Meniscus': 0.45,
        'ACL': 0.41,
        'Lateral Meniscus': 0.40,
        'PF OA': 0.36,
        'Contusion': 0.33,
        'Fracture': 0.31,
        'Medial OA': 0.26,
        "Baker's": 0.21,
        'Lateral OA': 0.19,
        'MCL': 0.16
    }

    checkpoint_path = OUTPUT_DIR / "best_coatnet_knee_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path.exists():
        print(f"Found trained model weights at {checkpoint_path}. Running neural inference...")
        model = SpeedyRaptorCoAtNetModel(
            backbone_name="coatnet_0_rw_224",
            pretrained=False,
            in_chans=1
        ).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        test_dataset = KneeMRISlotDataset(
            df_studies=df_test,
            df_series=df_test_series,
            data_dir=BASE_DIR,
            labels=None,
            num_slices=NUM_SLICES_PER_SERIES,
            target_size=(IMG_SIZE, IMG_SIZE),
            is_training=False
        )

        all_preds = []
        with torch.no_grad():
            for i in range(len(test_dataset)):
                sample = test_dataset[i]
                slot_tensors = sample["slot_tensors"].unsqueeze(0).to(device)
                slot_mask = sample["slot_mask"].unsqueeze(0).to(device)

                probs = model.predict_proba(slot_tensors, slot_mask)
                all_preds.append(probs.cpu().numpy()[0])

        all_preds = np.array(all_preds)
        # Apply Percentile Ranking across test predictions to maximize ROC-AUC
        preds_df = pd.DataFrame(all_preds, columns=LABELS)
        ranked_preds = preds_df.rank(pct=True).values

        for idx, col in enumerate(LABELS):
            df_test[col] = ranked_preds[:, idx]
    else:
        print("Trained model checkpoint not found; generating calibrated prior submission...")
        for col in LABELS:
            df_test[col] = prior_rates.get(col, 0.3)

    # Neutral fill any missing values with 0.5
    for col in LABELS:
        df_test[col] = df_test[col].fillna(0.5)

    submission = df_test[["StudyInstanceUID"] + LABELS].copy()
    sub_path = BASE_DIR / "submission.csv"
    submission.to_csv(sub_path, index=False)

    print("\n" + "-" * 40)
    print("SUBMISSION VERIFICATION")
    print("-" * 40)
    print(f"File created: {sub_path}")
    print(f"Shape: {submission.shape}")
    print(f"NaN count: {submission.isna().sum().sum()}")
    print("\nPreview:")
    print(submission.head())

if __name__ == "__main__":
    main()
