import io
import sys
import pandas as pd
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

labels = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

df = pd.read_csv('train.csv')
labeled = df[df['ACL'].notnull()].copy()

print(f"Total labeled cases: {len(labeled)}")

# Print 10 detailed examples with positive labels highlighted
for idx, (original_idx, row) in enumerate(labeled.iloc[:10].iterrows()):
    active_labels = [col for col in labels if row[col] == 1.0]
    print(f"\n================ Case {idx+1} (Index {original_idx}) ================")
    print(f"Positive Findings: {active_labels}")
    print("Report:")
    text = str(row['Report']).strip()
    for line in text.split('\n'):
        if line.strip():
            print(f"  {line.strip()}")
