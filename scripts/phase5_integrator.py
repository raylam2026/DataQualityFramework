"""
Phase 5: Integration & Cross-Dataset Evaluation (v2)
====================================================
Changes from v1:
  ✅ Uses saved threshold from Phase 4
  ✅ predict_proba + threshold instead of predict
  ✅ Cross-dataset generalization with threshold tuning

Usage:
    python scripts/phase5_integrator.py
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from quality_assessment.feature_engineer import FeatureEngineer, FEATURE_NAMES

DATA_DIR = PROJECT_ROOT / 'data'
LABELED_DIR = DATA_DIR / 'labeled'
MODEL_DIR = DATA_DIR / 'models'


def load_ground_truth(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in ['completeness', 'consistency', 'validity', 'accuracy', 'final_label']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce').fillna(0).astype(int)
    return df


def evaluate(y_true, y_pred, y_proba, label):
    p = float(precision_score(y_true, y_pred, zero_division=0))
    r = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float((y_pred == y_true).mean())
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        auc = 0.0
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except Exception:
        tn = fp = fn = tp = 0
    h1_pass = (p >= 0.80) and (r >= 0.75)
    print(f"  {label}")
    print(f"    P={p:.4f}  R={r:.4f}  F1={f1:.4f}  Acc={acc:.4f}  H1={'✅' if h1_pass else '❌'}")
    return {
        'dataset': label,
        'n_samples': int(len(y_true)),
        'n_positive': int((y_true == 1).sum()),
        'n_negative': int((y_true == 0).sum()),
        'precision': p, 'recall': r, 'f1_score': f1,
        'auc_roc': auc, 'accuracy': acc,
        'h1_pass': h1_pass,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    }


def find_optimal_threshold(model, X, y, min_precision=0.80, min_recall=0.75):
    proba = model.predict_proba(X)[:, 1]
    best_threshold = 0.5
    best_f1 = -1.0
    for t in np.arange(0.10, 0.91, 0.01):
        pred = (proba >= t).astype(int)
        p = precision_score(y, pred, zero_division=0)
        r = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        if p >= min_precision and r >= min_recall and f1 > best_f1:
            best_f1 = f1
            best_threshold = round(t, 2)
    if best_f1 < 0:
        for t in np.arange(0.10, 0.91, 0.01):
            pred = (proba >= t).astype(int)
            r = recall_score(y, pred, zero_division=0)
            f1 = f1_score(y, pred, zero_division=0)
            if r >= min_recall and f1 > best_f1:
                best_f1 = f1
                best_threshold = round(t, 2)
    if best_f1 < 0:
        for t in np.arange(0.10, 0.91, 0.01):
            pred = (proba >= t).astype(int)
            f1 = f1_score(y, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = round(t, 2)
    return best_threshold, best_f1


# ══════════════════════════════════════════════════════════════
# STEP 1: Load Phase 4 Artifacts
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 5 v2: INTEGRATION & CROSS-DATASET EVALUATION")
print("=" * 70)

rf_model = joblib.load(MODEL_DIR / 'rf_classifier.pkl')
feature_engineers = joblib.load(MODEL_DIR / 'feature_engineers.pkl')
thresholds = joblib.load(MODEL_DIR / 'thresholds.pkl')
global_threshold = thresholds['global']
per_dataset_thresholds = thresholds.get('per_dataset', {})

print(f"✅ Loaded RF model, feature engineers, thresholds")
print(f"   Global threshold: {global_threshold}")
print(f"   Per-dataset thresholds: {per_dataset_thresholds}\n")

# ══════════════════════════════════════════════════════════════
# STEP 2: Load Data & Extract Features
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 2: LOADING DATA & EXTRACTING FEATURES (13 features)")
print("=" * 70)

gt_files = {
    'titanic': LABELED_DIR / 'titanic_ground_truth.csv',
    'ecommerce': LABELED_DIR / 'brazilian_ecommerce_ground_truth.csv',
    'hr': LABELED_DIR / 'hr_ground_truth.csv',
}

datasets = {}
X_data = {}
y_data = {}

for name, path in gt_files.items():
    if not path.exists():
        print(f"  ⚠️ {name.upper()}: NOT FOUND — {path}")
        continue
    df = load_ground_truth(path)
    datasets[name] = df
    fe = feature_engineers[name]
    X = fe.transform(df)
    y = df['final_label'].values
    X_data[name] = X
    y_data[name] = y
    print(f"  ✅ {name.upper():15s}: {len(df)} rows, {X.shape[1]} features")

# ══════════════════════════════════════════════════════════════
# STEP 3: Per-Dataset Evaluation (Phase 4 model + threshold)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: PER-DATASET EVALUATION (Phase 4 model + optimized threshold)")
print("=" * 70)

metrics = {}
for name in ['titanic', 'ecommerce', 'hr']:
    if name not in X_data:
        continue
    threshold = per_dataset_thresholds.get(name, global_threshold)
    y_proba = rf_model.predict_proba(X_data[name])[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    metrics[name] = evaluate(y_data[name], y_pred, y_proba, f'{name} (threshold={threshold})')
    metrics[name]['threshold'] = threshold

# Combined
if X_data:
    all_X = np.vstack([X_data[n] for n in X_data])
    all_y = np.concatenate([y_data[n] for n in y_data])
    all_proba = rf_model.predict_proba(all_X)[:, 1]
    all_pred = (all_proba >= global_threshold).astype(int)
    metrics['combined'] = evaluate(all_y, all_pred, all_proba, f'combined (threshold={global_threshold})')

# ══════════════════════════════════════════════════════════════
# STEP 4: Cross-Dataset Generalization (Train on 2, Test on 1)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: CROSS-DATASET GENERALIZATION (Train 2 → Test 1, with threshold)")
print("=" * 70)

dataset_names = [n for n in ['titanic', 'ecommerce', 'hr'] if n in X_data]
cross_results = {}

for test_name in dataset_names:
    train_names = [n for n in dataset_names if n != test_name]
    X_tr = np.vstack([X_data[n] for n in train_names])
    y_tr = np.concatenate([y_data[n] for n in train_names])
    X_te = X_data[test_name]
    y_te = y_data[test_name]

    try:
        sm = SMOTE(random_state=42)
        X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
    except Exception:
        X_tr_sm, y_tr_sm = X_tr, y_tr

    rf_cross = RandomForestClassifier(
        n_estimators=100, max_depth=15,
        class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf_cross.fit(X_tr_sm, y_tr_sm)

    # Optimize threshold on training data
    cross_threshold, _ = find_optimal_threshold(rf_cross, X_tr_sm, y_tr_sm)
    y_proba = rf_cross.predict_proba(X_te)[:, 1]
    y_pred = (y_proba >= cross_threshold).astype(int)

    train_str = '+'.join(t.upper() for t in train_names)
    test_str = test_name.upper()
    cross_results[test_name] = evaluate(
        y_te, y_pred, y_proba,
        f'Train({train_str}) → Test({test_str}) threshold={cross_threshold}'
    )
    cross_results[test_name]['threshold'] = cross_threshold

metrics['cross_dataset'] = cross_results

# ══════════════════════════════════════════════════════════════
# STEP 5: Save Results
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: SAVING RESULTS")
print("=" * 70)

output_path = DATA_DIR / 'phase5_metrics.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print(f"  ✅ Results saved → {output_path}")
if 'combined' in metrics:
    print(f"  Combined F1: {metrics['combined']['f1_score']:.4f}")
for name in dataset_names:
    if name in metrics:
        print(f"  {name.upper()} H1: {'✅' if metrics[name]['h1_pass'] else '❌'}  "
              f"P={metrics[name]['precision']:.4f}  R={metrics[name]['recall']:.4f}")

print("\n" + "=" * 70)
print("✅ PHASE 5 v2 COMPLETE")
print("=" * 70)
