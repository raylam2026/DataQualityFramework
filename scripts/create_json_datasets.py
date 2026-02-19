# Run once to create JSON versions of your ground-truth CSVs:
# Save as: scripts/create_json_datasets.py

import pandas as pd
from pathlib import Path

DATA_DIR = Path('data/labeled')

for csv_file in DATA_DIR.glob('*_ground_truth.csv'):
    df = pd.read_csv(csv_file)
    json_path = csv_file.with_suffix('.json')
    df.to_json(json_path, orient='records', indent=2)
    print(f"✅ {csv_file.name} → {json_path.name} ({len(df)} rows)")
