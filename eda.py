import io
import sys
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

labels = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

print("=== LOAD DATA ===")
df_train = pd.read_csv('train.csv')
df_train_series = pd.read_csv('train_series.csv')
df_test = pd.read_csv('test.csv')
df_test_series = pd.read_csv('test_series.csv')
df_sub = pd.read_csv('sample_submission.csv')

print(f"train.csv shape: {df_train.shape}")
print(f"train_series.csv shape: {df_train_series.shape}")
print(f"test.csv shape: {df_test.shape}")
print(f"test_series.csv shape: {df_test_series.shape}")
print(f"sample_submission.csv shape: {df_sub.shape}")

print("\n=== LABELED SUBSET STATS ===")
labeled = df_train[df_train['ACL'].notnull()]
print(f"Number of labeled train studies: {len(labeled)} / {len(df_train)}")
print("Positive label rates in 58 expert-labeled studies:")
pos_rates = labeled[labels].mean().sort_values(ascending=False)
for col, val in pos_rates.items():
    print(f"  {col:18s}: {val*100:5.1f}% ({int(labeled[col].sum())}/{len(labeled)})")

print("\n=== SERIES METADATA (TRAIN) ===")
print("Anatomical Planes:")
print(df_train_series['Anatomical_Plane'].value_counts(dropna=False))
print("\nFluid Sensitive:")
print(df_train_series['Fluid_Sensitive'].value_counts(dropna=False))
print("\nFat Suppression:")
print(df_train_series['Fat_Suppression'].value_counts(dropna=False))

print("\nNumber of series per study in train:")
series_per_study = df_train_series.groupby('StudyInstanceUID').size()
print(series_per_study.describe())

print("\n=== SAMPLE REPORTS ===")
for idx in [0, 29, 63, 100, 200]:
    row = df_train.iloc[idx]
    is_lab = "EXPERT LABELED" if pd.notnull(row['ACL']) else "UNLABELED"
    print(f"\n--- Study {row['StudyInstanceUID'][:16]}... [{is_lab}] ---")
    rep = str(row['Report'])
    print(rep[:250] + ("..." if len(rep) > 250 else ""))
