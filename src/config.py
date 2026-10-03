import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_SERIES_CSV = DATA_DIR / "train_series.csv"
TEST_CSV = DATA_DIR / "test.csv"
TEST_SERIES_CSV = DATA_DIR / "test_series.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

PSEUDO_LABELED_TRAIN_CSV = OUTPUT_DIR / "train_pseudo_labeled.csv"

# 12 Target Abnormality Labels
LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture"
]

# 6 Clinical Anatomical Slots (Plane, Fluid_Sensitive)
# Inspired by Speedy Raptors CoAtNet (LB 0.943+)
SLOTS = [
    ("Sagittal", 1),  # Slot 0: Sagittal Fluid-Sensitive (FS/T2/PD)
    ("Coronal",  1),  # Slot 1: Coronal Fluid-Sensitive
    ("Axial",    1),  # Slot 2: Axial Fluid-Sensitive
    ("Sagittal", 0),  # Slot 3: Sagittal Non-Fluid (T1/PD no-FS)
    ("Coronal",  0),  # Slot 4: Coronal Non-Fluid (T1)
    ("Axial",    0),  # Slot 5: Axial Non-Fluid (T1)
]
N_SLOT = len(SLOTS)

# Clinical inductive bias prior table mapping abnormalities to relevant slots
SLOT_PRIOR_TABLE = {
    'ACL': (0, 3, 5),                       # Sagittal fluid, Sagittal T1, Axial T1
    'MCL': (1, 4),                          # Coronal fluid, Coronal T1
    'Medial Meniscus': (0, 1, 3, 4),        # Sagittal & Coronal (fluid + non-fluid)
    'Lateral Meniscus': (0, 1, 3, 4),       # Sagittal & Coronal (fluid + non-fluid)
    'Medial OA': (1, 4, 5),                 # Coronal views + Axial
    'Lateral OA': (1, 4, 5),                # Coronal views + Axial
    'PF OA': (0, 2, 5),                     # Sagittal, Axial fluid, Axial T1
    'Effusion': (0, 2),                     # Sagittal fluid, Axial fluid
    'Synovitis': (0, 2),                    # Sagittal fluid, Axial fluid
    "Baker's": (0,),                        # Sagittal fluid (popliteal fossa)
    'Contusion': (0, 1, 2),                 # Fluid-sensitive across all 3 planes (edema)
    'Fracture': (0, 1, 2, 4, 5)             # Cortical disruption across planes
}
SLOT_PRIOR_STRENGTH = 0.55

# Training Hyperparameters
SEED = 42
NUM_SLICES_PER_SERIES = 16
IMG_SIZE = 224
BATCH_SIZE = 2
NUM_WORKERS = 0
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
