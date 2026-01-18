import pandas as pd

print("="*70)
print("ECOMMERCE LABELING VERIFICATION")
print("="*70)

labels = pd.read_csv('../data/labeled/brazilian_ecommerce_ground_truth.csv')

# Check 1: Row count
print(f"\n[CHECK 1] Row count verification...")
print(f"✅ PASS: {len(labels)} rows (expected 1000)")
assert len(labels) == 1000

# Check 2: Required columns
print(f"\n[CHECK 2] Required columns verification...")
required_cols = ['order_id', 'completeness', 'consistency', 'validity', 'accuracy', 'final_label']
for col in required_cols:
    assert col in labels.columns
    print(f"✅ Column present: {col}")

# Check 3: No missing values
print(f"\n[CHECK 3] Missing values verification...")
for dim in ['completeness', 'consistency', 'validity', 'accuracy', 'final_label']:
    missing = labels[dim].isnull().sum()
    assert missing == 0
    print(f"✅ {dim}: No missing values")

# Check 4: Values are 0 or 1
print(f"\n[CHECK 4] Value range verification...")
for col in ['completeness', 'consistency', 'validity', 'accuracy', 'final_label']:
    unique_vals = set(labels[col].unique())
    assert unique_vals.issubset({0, 1}), f"{col} has invalid values: {unique_vals}"
    print(f"✅ {col}: All values are 0 or 1")

# Check 5: Final label logic
print(f"\n[CHECK 5] Final label logic verification...")
errors = 0
for idx, row in labels.iterrows():
    dimensions = [row['completeness'], row['consistency'], row['validity'], row['accuracy']]
    passing = sum(dimensions)
    expected = 1 if passing >= 3 else 0
    if row['final_label'] != expected:
        print(f"  ❌ Row {idx}: Got {row['final_label']}, expected {expected} (passing={passing})")
        errors += 1

assert errors == 0, f"Found {errors} logic errors!"
print(f"✅ All {len(labels)} rows have correct label logic")

# Check 6: Distribution
print(f"\n[CHECK 6] Label distribution...")
dist = labels['final_label'].value_counts().sort_index()
for label in [0, 1]:
    if label in dist.index:
        count = dist[label]
        pct = 100 * count / len(labels)
        status = "HIGH QUALITY" if label == 1 else "LOW QUALITY"
        print(f"  {status} ({label}): {count} rows ({pct:.1f}%)")

print("\n" + "="*70)
print("✅ ALL VERIFICATION CHECKS PASSED")
print("="*70)

# Show dimension breakdown
print(f"\n[BONUS] Dimension breakdown:")
for dim in ['completeness', 'consistency', 'validity', 'accuracy']:
    pass_count = (labels[dim] == 1).sum()
    pct = 100 * pass_count / len(labels)
    print(f"  {dim}: {pass_count}/{len(labels)} = {pct:.1f}% PASS")
