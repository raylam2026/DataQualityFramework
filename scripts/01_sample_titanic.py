import pandas as pd
import numpy as np

print("="*70)
print("PHASE 4: STEP 1.2 - SAMPLE TITANIC DATASET")
print("="*70)

# Load Titanic data
print("\n[STEP 1] Loading Titanic dataset...")
titanic = pd.read_csv('data/raw/titanic/train.csv')
print(f"✅ Total rows: {len(titanic)}")
print(f"✅ Columns: {list(titanic.columns)}")

# Display basic info
print("\n[STEP 2] Dataset Analysis...")
print(f"Missing values by column:")
print(titanic.isnull().sum())
print(f"\nPassenger class distribution:")
print(titanic['Pclass'].value_counts().sort_index())

# Stratified sampling by Pclass
print("\n[STEP 3] Performing stratified sampling...")
print("  → Maintaining passenger class distribution")
print("  → Using random_state=42 for reproducibility")

titanic_sample = titanic.sample(
    n=90,                              # 10% of 891 rows
    stratify=titanic['Pclass'],        # Stratify by class
    random_state=42                    # Reproducible
)

print(f"✅ Sampled: {len(titanic_sample)} rows")
print(f"✅ Class distribution in sample:")
print(titanic_sample['Pclass'].value_counts().sort_index())

# Save sample
print("\n[STEP 4] Saving sample to CSV...")
output_path = 'data/labeled/titanic_sample.csv'
titanic_sample.to_csv(output_path, index=False)
print(f"✅ Saved to: {output_path}")

# Create labeling template
print("\n[STEP 5] Creating labeling template...")
labeling_template = pd.DataFrame({
    'PassengerId': titanic_sample['PassengerId'],
    'Name': titanic_sample['Name'],
    'Pclass': titanic_sample['Pclass'],
    'Sex': titanic_sample['Sex'],
    'Age': titanic_sample['Age'],
    'Fare': titanic_sample['Fare'],
    'Cabin': titanic_sample['Cabin'],
    'Embarked': titanic_sample['Embarked'],
    'completeness': [None] * 90,
    'consistency': [None] * 90,
    'validity': [None] * 90,
    'accuracy': [None] * 90,
    'final_label': [None] * 90,
    'notes': [None] * 90
})

template_path = 'data/labeled/titanic_labeling_template.csv'
labeling_template.to_csv(template_path, index=False)
print(f"✅ Labeling template saved to: {template_path}")

print("\n" + "="*70)
print("NEXT STEP: Open titanic_labeling_template.csv and manually label")
print("each row with values (0 or 1) for each dimension")
print("="*70)
