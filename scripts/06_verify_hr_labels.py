import pandas as pd

print("="*70)
print("HR ANALYTICS LABELING VERIFICATION")
print("="*70)

labels = pd.read_csv('../data/labeled/hr_ground_truth.csv')

# Check 1: Row count
print(f"\n[CHECK 1] Row count: {len(labels)} (expected 147)")
assert len(labels) == 147

# Check 2: Columns
print(f"\n[CHECK 2] Required columns present")
for col in ['EmpID', 'completeness', 'consistency', 'validity', 'accuracy', 'final_label']:
    assert col in labels.columns
    print(f"  ✅ {col}")

# Check 3: No missing values
print(f"\n[CHECK 3] No missing values in label columns")
for dim in ['completeness', 'consistency', 'validity', 'accuracy', 'final_label']:
    assert labels[dim].isnull().sum() == 0
    print(f"  ✅ {dim}")

# Check 4: Values are 0 or 1
print(f"\n[CHECK 4] Values are 0 or 1")
for col in ['completeness', 'consistency', 'validity', 'accuracy', 'final_label']:
    assert set(labels[col].unique()).issubset({0, 1})

# Check 5: Label logic
print(f"\n[CHECK 5] Final label logic")
for idx, row in labels.iterrows():
    dimensions = [row['completeness'], row['consistency'], row['validity'], row['accuracy']]
    passing = sum(dimensions)
    expected = 1 if passing >= 3 else 0
    assert row['final_label'] == expected

# Check 6: Distribution
print(f"\n[CHECK 6] Label distribution")
dist = labels['final_label'].value_counts().sort_index()
for label, count in dist.items():
    pct = 100 * count / len(labels)
    print(f"  {'HIGH' if label==1 else 'LOW'} ({label}): {count} ({pct:.1f}%)")

print("\n✅ ALL HR ANALYTICS VERIFICATION CHECKS PASSED")
