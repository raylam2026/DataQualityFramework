import pandas as pd
import numpy as np

print("="*70)
print("PHASE 4: STEP 2.1 - SAMPLE ECOMMERCE DATASET")
print("="*70)

# Load E-Commerce data
print("\n[STEP 1] Loading E-Commerce dataset...")
ecommerce = pd.read_csv('../data/raw/brazilian_ecommerce/olist_orders_dataset.csv')
print(f"✅ Total rows: {len(ecommerce)}")
print(f"✅ Columns: {list(ecommerce.columns)}")

# Convert timestamp
print("\n[STEP 2] Processing timestamps...")
ecommerce['order_purchase_timestamp'] = pd.to_datetime(ecommerce['order_purchase_timestamp'])
ecommerce['order_month'] = ecommerce['order_purchase_timestamp'].dt.to_period('M')

print(f"Date range: {ecommerce['order_purchase_timestamp'].min()} to {ecommerce['order_purchase_timestamp'].max()}")
print(f"Total months: {ecommerce['order_month'].nunique()}")
print(f"\nMonth distribution (top 10):")
print(ecommerce['order_month'].value_counts().head(10))

# Stratified sampling by month (HANDLES RARE CLASSES)
print("\n[STEP 3] Performing stratified sampling by month...")

# Get proportions
month_proportions = ecommerce['order_month'].value_counts(normalize=True).sort_index()

print(f"Months in dataset: {len(month_proportions)}")
rare_months = (month_proportions < 5/len(ecommerce)).sum()
print(f"Months with <5 orders: {rare_months}")

# Sample from each month proportionally
np.random.seed(42)
ecommerce_sample = pd.concat([
    ecommerce[ecommerce['order_month'] == month].sample(
        n=max(1, int(1000 * proportion)),
        random_state=42,
        replace=False
    )
    for month, proportion in month_proportions.items()
])

print(f"After stratified sampling: {len(ecommerce_sample)} rows")

# Resample to exactly 1000 if needed (allow replacement)
if len(ecommerce_sample) != 1000:
    shortage = 1000 - len(ecommerce_sample)
    print(f"⚠️  Got {len(ecommerce_sample)} rows, need {shortage} more")
    print(f"Resampling to exactly 1000 rows (with small replacement)...")
    
    ecommerce_sample = ecommerce_sample.sample(
        n=1000,
        random_state=42,
        replace=True  # ✅ ALLOW DUPLICATES
    )

print(f"✅ Final sampled: {len(ecommerce_sample)} rows (~1% of {len(ecommerce)})")
print(f"✅ Representing {ecommerce_sample['order_month'].nunique()} months")
print(f"\n✅ Month distribution in sample:")
print(ecommerce_sample['order_month'].value_counts().sort_index())

# Save sample
print("\n[STEP 4] Saving sample...")
ecommerce_sample_clean = ecommerce_sample.drop('order_month', axis=1).reset_index(drop=True)
output_path = '../data/labeled/brazilian_ecommerce_sample.csv'
ecommerce_sample_clean.to_csv(output_path, index=False)
print(f"✅ Saved to: {output_path}")

# Create labeling template
print("\n[STEP 5] Creating labeling template...")
labeling_template = pd.DataFrame({
    'order_id': ecommerce_sample_clean['order_id'],
    'customer_id': ecommerce_sample_clean['customer_id'],
    'order_status': ecommerce_sample_clean['order_status'],
    'order_purchase_timestamp': ecommerce_sample_clean['order_purchase_timestamp'],
    'order_approved_at': ecommerce_sample_clean['order_approved_at'],
    'order_delivered_carrier_date': ecommerce_sample_clean['order_delivered_carrier_date'],
    'order_delivered_customer_date': ecommerce_sample_clean['order_delivered_customer_date'],
    'order_estimated_delivery_date': ecommerce_sample_clean['order_estimated_delivery_date'],
    'completeness': [None] * len(ecommerce_sample_clean),
    'consistency': [None] * len(ecommerce_sample_clean),
    'validity': [None] * len(ecommerce_sample_clean),
    'accuracy': [None] * len(ecommerce_sample_clean),
    'final_label': [None] * len(ecommerce_sample_clean),
    'notes': [None] * len(ecommerce_sample_clean)
})

template_path = '../data/labeled/brazilian_ecommerce_labeling_template.csv'
labeling_template.to_csv(template_path, index=False)
print(f"✅ Labeling template saved to: {template_path}")

print("\n" + "="*70)
print("✅ SAMPLING COMPLETE")
print("NEXT STEP: Label all 1,000 rows in ecommerce_labeling_template.csv")
print("="*70)
