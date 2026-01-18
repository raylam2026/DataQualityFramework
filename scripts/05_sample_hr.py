import pandas as pd
import numpy as np

print("="*70)
print("PHASE 4: STEP 3.1 - SAMPLE HR ANALYTICS DATASET")
print("="*70)

# Load HR data
print("\n[STEP 1] Loading HR Analytics dataset...")
hr = pd.read_csv('../data/raw/hr_analytics/HRDataset_v14.csv')
print(f"✅ Total rows: {len(hr)}")
print(f"✅ Columns: {list(hr.columns)[:10]}...")

# Analyze department distribution
print("\n[STEP 2] Department distribution:")
print(hr['Department'].value_counts())

# Stratified sampling by Department (HANDLES RARE CLASSES)
print("\n[STEP 3] Performing stratified sampling by Department...")

# Get proportions
dept_proportions = hr['Department'].value_counts(normalize=True).sort_index()

print(f"Departments in dataset: {len(dept_proportions)}")
rare_depts = (dept_proportions < 2/len(hr)).sum()
print(f"Departments with <2 employees: {rare_depts}")

# Sample from each department proportionally
np.random.seed(42)
hr_sample = pd.concat([
    hr[hr['Department'] == dept].sample(
        n=max(1, int(147 * proportion)),  # At least 1 per department
        random_state=42,
        replace=False
    )
    for dept, proportion in dept_proportions.items()
])

print(f"After stratified sampling: {len(hr_sample)} rows")

# Resample to exactly 147 if needed (allow replacement)
if len(hr_sample) != 147:
    shortage = 147 - len(hr_sample)
    print(f"⚠️  Got {len(hr_sample)} rows, need {shortage} more")
    print(f"Resampling to exactly 147 rows (with small replacement)...")
    
    hr_sample = hr_sample.sample(
        n=147,
        random_state=42,
        replace=True
    )

print(f"✅ Final sampled: {len(hr_sample)} rows")
print(f"✅ Sample distribution:")
print(hr_sample['Department'].value_counts().sort_index())

# Save sample
print("\n[STEP 4] Saving sample...")
output_path = '../data/labeled/hr_sample.csv'
hr_sample.to_csv(output_path, index=False)
print(f"✅ Saved to: {output_path}")

# Create labeling template
print("\n[STEP 5] Creating labeling template...")
labeling_template = hr_sample.reset_index(drop=True).copy()
labeling_template['completeness'] = None
labeling_template['consistency'] = None
labeling_template['validity'] = None
labeling_template['accuracy'] = None
labeling_template['final_label'] = None
labeling_template['notes'] = None

template_path = '../data/labeled/hr_labeling_template.csv'
labeling_template.to_csv(template_path, index=False)
print(f"✅ Labeling template saved to: {template_path}")

print("\n" + "="*70)
print("✅ SAMPLING COMPLETE")
print("NEXT STEP: Label all 147 rows in hr_labeling_template.csv")
print("="*70)
