import pandas as pd
import numpy as np

print("="*70)
print("TITANIC LABELING VERIFICATION")
print("="*70)

# Load labeled data
labels = pd.read_csv('../data/labeled/titanic_ground_truth.csv')

# Check 1: Row count
print("\n[CHECK 1] Row count verification...")
expected_rows = 89
actual_rows = len(labels)
if actual_rows == expected_rows:
    print(f"✅ PASS: {actual_rows} rows (expected {expected_rows})")
else:
    print(f"❌ FAIL: {actual_rows} rows (expected {expected_rows})")
    exit(1)

# Check 2: Required columns
print("\n[CHECK 2] Required columns verification...")
required_cols = ['PassengerId', 'completeness', 'consistency', 'validity', 'accuracy', 'final_label']
for col in required_cols:
    if col in labels.columns:
        print(f"✅ Column present: {col}")
    else:
        print(f"❌ Column missing: {col}")
        exit(1)

# Check 3: No missing values
print("\n[CHECK 3] Missing values verification...")
for dim in ['completeness', 'consistency', 'validity', 'accuracy', 'final_label']:
    missing_count = labels[dim].isnull().sum()
    if missing_count == 0:
        print(f"✅ {dim}: No missing values")
    else:
        print(f"❌ {dim}: {missing_count} missing values")
        exit(1)

# Check 4: All values are 0 or 1
print("\n[CHECK 4] Value range verification...")
for col in ['completeness', 'consistency', 'validity', 'accuracy', 'final_label']:
    unique_vals = set(labels[col].unique())
    if unique_vals.issubset({0, 1}):
        print(f"✅ {col}: All values are 0 or 1")
    else:
        print(f"❌ {col}: Invalid values found: {unique_vals}")
        exit(1)

# Check 5: Final label logic (≥3 dimensions pass = 1)
print("\n[CHECK 5] Final label logic verification...")
logic_errors = []
for idx, row in labels.iterrows():
    dimensions = [row['completeness'], row['consistency'], row['validity'], row['accuracy']]
    passing_count = sum(dimensions)
    expected_label = 1 if passing_count >= 3 else 0
    if row['final_label'] != expected_label:
        logic_errors.append({
            'row': idx,
            'passing_count': passing_count,
            'actual_label': row['final_label'],
            'expected_label': expected_label
        })

if not logic_errors:
    print(f"✅ All {len(labels)} rows have correct label logic")
else:
    print(f"❌ Found {len(logic_errors)} logic errors:")
    for error in logic_errors:
        print(f"   Row {error['row']}: {error['passing_count']} pass → " +
              f"label={error['actual_label']} (expected {error['expected_label']})")
    exit(1)

# Check 6: Label distribution
print("\n[CHECK 6] Label distribution...")
label_counts = labels['final_label'].value_counts().sort_index()
total = len(labels)
for label_val, count in label_counts.items():
    pct = 100 * count / total
    status = "HIGH QUALITY" if label_val == 1 else "LOW QUALITY"
    print(f"  {status} ({label_val}): {count} rows ({pct:.1f}%)")

print("\n" + "="*70)
print("✅ ALL VERIFICATION CHECKS PASSED")
print("="*70)
