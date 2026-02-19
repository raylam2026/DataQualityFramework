"""
Phase 4: ML Classifier Training & Evaluation (v2)
==================================================
Changes from v1:
  ✅ 13 features (added 2 temporal features)
  ✅ class_weight='balanced' in RF to handle imbalanced data
  ✅ Threshold optimization on validation set
  ✅ Per-dataset threshold tuning for H1

Usage:
    python scripts/phase4_ml_classifier.py
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE

# ── Path Setup ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from quality_assessment.feature_engineer import FeatureEngineer, FEATURE_NAMES

DATA_DIR = PROJECT_ROOT / 'data'
LABELED_DIR = DATA_DIR / 'labeled'
MODEL_DIR = DATA_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def load_ground_truth(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in ['completeness', 'consistency', 'validity', 'accuracy', 'final_label']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce').fillna(0).astype(int)
    return df


def evaluate(model, X, y, label, threshold=0.5):
    """Evaluate with configurable probability threshold."""
    if threshold == 0.5:
        pred = model.predict(X)
    else:
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)
    p = float(precision_score(y, pred, zero_division=0))
    r = float(recall_score(y, pred, zero_division=0))
    f1 = float(f1_score(y, pred, zero_division=0))
    acc = float((pred == y).mean())
    try:
        auc = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))
    except Exception:
        auc = 0.0
    try:
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    except Exception:
        tn = fp = fn = tp = 0
    h1_pass = (p >= 0.80) and (r >= 0.75)
    print(f"  {label}")
    print(f"    Precision: {p:.4f} {'✅' if p >= 0.80 else '⚠️'}  "
          f"Recall: {r:.4f} {'✅' if r >= 0.75 else '⚠️'}  "
          f"F1: {f1:.4f}  Acc: {acc:.4f}  Threshold: {threshold}")
    return {
        'label': label, 'precision': p, 'recall': r, 'f1_score': f1,
        'auc_roc': auc, 'accuracy': acc, 'threshold': threshold,
        'n_samples': int(len(y)), 'n_high': int((y == 1).sum()), 'n_low': int((y == 0).sum()),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'h1_pass': h1_pass,
    }


def find_optimal_threshold(model, X, y, min_precision=0.80, min_recall=0.75):
    """Grid-search threshold on validation set that meets H1 criteria with best F1.
    
    3-tier fallback:
      1. P ≥ 0.80 AND R ≥ 0.75 → best F1
      2. R ≥ 0.75 only → best F1
      3. Any threshold → best F1
    """
    proba = model.predict_proba(X)[:, 1]
    best_threshold = 0.5
    best_f1 = -1.0

    # Tier 1: Both P and R meet H1
    for t in np.arange(0.10, 0.91, 0.01):
        pred = (proba >= t).astype(int)
        p = precision_score(y, pred, zero_division=0)
        r = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        if p >= min_precision and r >= min_recall and f1 > best_f1:
            best_f1 = f1
            best_threshold = round(t, 2)

    # Tier 2: At least recall meets target
    if best_f1 < 0:
        for t in np.arange(0.10, 0.91, 0.01):
            pred = (proba >= t).astype(int)
            r = recall_score(y, pred, zero_division=0)
            f1 = f1_score(y, pred, zero_division=0)
            if r >= min_recall and f1 > best_f1:
                best_f1 = f1
                best_threshold = round(t, 2)

    # Tier 3: Just maximize F1
    if best_f1 < 0:
        for t in np.arange(0.10, 0.91, 0.01):
            pred = (proba >= t).astype(int)
            f1 = f1_score(y, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = round(t, 2)

    return best_threshold, best_f1


# ══════════════════════════════════════════════════════════════
# STEP 1: Load Ground-Truth CSVs
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 4 v2: ML CLASSIFIER WITH TEMPORAL FEATURES + BALANCED WEIGHTS")
print("=" * 70)

datasets = {
    'titanic': load_ground_truth(LABELED_DIR / 'titanic_ground_truth.csv'),
    'ecommerce': load_ground_truth(LABELED_DIR / 'brazilian_ecommerce_ground_truth.csv'),
    'hr': load_ground_truth(LABELED_DIR / 'hr_ground_truth.csv'),
}

for name, df in datasets.items():
    high = (df['final_label'] == 1).sum()
    low = (df['final_label'] == 0).sum()
    ratio = high / low if low > 0 else float('inf')
    print(f"  {name.upper():15s}: {len(df):5d} rows  (HIGH: {high}, LOW: {low}, ratio: {ratio:.1f}:1)")

# ══════════════════════════════════════════════════════════════
# STEP 2: Feature Engineering (13 features including temporal)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: FEATURE ENGINEERING (13 FEATURES: 4 ROW + 4 COL + 3 DATA + 2 TEMPORAL)")
print("=" * 70)

feature_engineers = {}
feature_matrices = {}
label_vectors = {}

for name, df in datasets.items():
    fe = FeatureEngineer()
    X = fe.fit_transform(df)
    y = df['final_label'].values
    feature_engineers[name] = fe
    feature_matrices[name] = X
    label_vectors[name] = y
    ts_count = len(fe.sorted_ts_cols)
    print(f"  {name.upper():15s}: X={X.shape}, timestamp_cols={ts_count}, "
          f"y: HIGH={int(y.sum())}, LOW={int((y==0).sum())}")

X_all = np.vstack([feature_matrices[n] for n in ['titanic', 'ecommerce', 'hr']])
y_all = np.concatenate([label_vectors[n] for n in ['titanic', 'ecommerce', 'hr']])
source_labels = np.concatenate([
    np.full(len(label_vectors[n]), i) for i, n in enumerate(['titanic', 'ecommerce', 'hr'])
])
print(f"\n  COMBINED: X={X_all.shape}, HIGH={int(y_all.sum())}, LOW={int((y_all==0).sum())}")

# ══════════════════════════════════════════════════════════════
# STEP 3: 60/20/20 Split + SMOTE + RF Training (BALANCED)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: TRAINING (class_weight='balanced' + SMOTE)")
print("=" * 70)

X_temp, X_test, y_temp, y_test, src_temp, src_test = train_test_split(
    X_all, y_all, source_labels, test_size=0.20, stratify=y_all, random_state=42
)
X_train, X_val, y_train, y_val, src_train, src_val = train_test_split(
    X_temp, y_temp, src_temp, test_size=0.25, stratify=y_temp, random_state=42
)
print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")

# SMOTE
try:
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    smote_applied = True
    print(f"  SMOTE: {len(X_train)} → {len(X_train_sm)} (balanced)")
except Exception as e:
    X_train_sm, y_train_sm = X_train, y_train
    smote_applied = False
    print(f"  SMOTE skipped: {e}")

# ── FIX 2: class_weight='balanced' ──
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight='balanced',    # ← NEW
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train_sm, y_train_sm)
print("  ✅ Random Forest trained (n_estimators=100, max_depth=15, class_weight='balanced')")

dt = DecisionTreeClassifier(max_depth=15, class_weight='balanced', random_state=42)
dt.fit(X_train_sm, y_train_sm)
print("  ✅ Decision Tree baseline trained (class_weight='balanced')")

# ══════════════════════════════════════════════════════════════
# STEP 4: Threshold Optimization (FIX 3)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: THRESHOLD OPTIMIZATION ON VALIDATION SET")
print("=" * 70)

# Global threshold
global_threshold, global_f1 = find_optimal_threshold(rf, X_val, y_val)
print(f"  Global optimal threshold: {global_threshold:.2f} (F1={global_f1:.4f})")

# Per-dataset threshold (for H1 5-fold CV)
per_dataset_thresholds = {}
for name in ['titanic', 'ecommerce', 'hr']:
    ds_idx = {'titanic': 0, 'ecommerce': 1, 'hr': 2}[name]
    mask = src_val == ds_idx
    if mask.sum() > 0:
        t, f1 = find_optimal_threshold(rf, X_val[mask], y_val[mask])
        per_dataset_thresholds[name] = t
        print(f"  {name.upper():15s} threshold: {t:.2f} (F1={f1:.4f}, n={mask.sum()})")
    else:
        per_dataset_thresholds[name] = global_threshold

# ══════════════════════════════════════════════════════════════
# STEP 5: Evaluation (with optimized threshold)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: EVALUATION (default vs optimized threshold)")
print("=" * 70)

results = {}
print("\n  --- Default threshold (0.5) ---")
results['rf_test_default'] = evaluate(rf, X_test, y_test, 'RF Test (threshold=0.5)', threshold=0.5)

print("\n  --- Optimized threshold ---")
results['rf_validation'] = evaluate(rf, X_val, y_val, 'RF Validation (optimized)', threshold=global_threshold)
results['rf_test'] = evaluate(rf, X_test, y_test, 'RF Test (optimized)', threshold=global_threshold)
results['dt_test'] = evaluate(dt, X_test, y_test, 'DT Baseline', threshold=0.5)

# ══════════════════════════════════════════════════════════════
# STEP 6: H1 Per-Dataset 5-Fold CV (with all 3 fixes)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: H1 HYPOTHESIS — 5-FOLD CV PER DATASET (with all 3 fixes)")
print("=" * 70)
print("  H1 Success: Precision ≥ 0.80 AND Recall ≥ 0.75 on ≥ 2 of 3 datasets\n")

h1_results = {}
h1_pass_count = 0

for name in ['titanic', 'ecommerce', 'hr']:
    X_ds = feature_matrices[name]
    y_ds = label_vectors[name]

    if len(np.unique(y_ds)) < 2:
        print(f"  {name.upper()}: SKIPPED (only one class)")
        continue

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_p, fold_r, fold_f1 = [], [], []
    fold_thresholds = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_ds, y_ds)):
        X_tr, X_te = X_ds[train_idx], X_ds[test_idx]
        y_tr, y_te = y_ds[train_idx], y_ds[test_idx]

        # SMOTE
        try:
            sm = SMOTE(random_state=42)
            X_tr_s, y_tr_s = sm.fit_resample(X_tr, y_tr)
        except Exception:
            X_tr_s, y_tr_s = X_tr, y_tr

        # FIX 2: class_weight='balanced'
        rf_fold = RandomForestClassifier(
            n_estimators=100, max_depth=15,
            class_weight='balanced',
            random_state=42
        )
        rf_fold.fit(X_tr_s, y_tr_s)

        # FIX 3: Threshold optimization per fold
        fold_threshold, _ = find_optimal_threshold(rf_fold, X_tr_s, y_tr_s)
        fold_thresholds.append(fold_threshold)

        # Predict with optimized threshold
        proba = rf_fold.predict_proba(X_te)[:, 1]
        pred = (proba >= fold_threshold).astype(int)

        fold_p.append(precision_score(y_te, pred, zero_division=0))
        fold_r.append(recall_score(y_te, pred, zero_division=0))
        fold_f1.append(f1_score(y_te, pred, zero_division=0))

    mean_p = float(np.mean(fold_p))
    mean_r = float(np.mean(fold_r))
    mean_f1 = float(np.mean(fold_f1))
    h1_pass = (mean_p >= 0.80) and (mean_r >= 0.75)
    if h1_pass:
        h1_pass_count += 1

    h1_results[name] = {
        'mean_precision': mean_p, 'std_precision': float(np.std(fold_p)),
        'mean_recall': mean_r, 'std_recall': float(np.std(fold_r)),
        'mean_f1': mean_f1, 'std_f1': float(np.std(fold_f1)),
        'fold_precisions': [float(x) for x in fold_p],
        'fold_recalls': [float(x) for x in fold_r],
        'fold_f1s': [float(x) for x in fold_f1],
        'fold_thresholds': [float(x) for x in fold_thresholds],
        'h1_pass': h1_pass,
    }
    status = '✅ PASS' if h1_pass else '❌ FAIL'
    avg_t = np.mean(fold_thresholds)
    print(f"  {name.upper():15s}: P={mean_p:.4f}  R={mean_r:.4f}  F1={mean_f1:.4f}  "
          f"avg_threshold={avg_t:.2f} → {status}")

results['h1_per_dataset'] = h1_results
results['h1_overall'] = {
    'datasets_passing': h1_pass_count,
    'threshold': 2,
    'h1_pass': h1_pass_count >= 2,
}
print(f"\n  H1 OVERALL: {h1_pass_count}/3 datasets pass → "
      f"{'✅ H1 SUPPORTED' if h1_pass_count >= 2 else '❌ H1 NOT YET SUPPORTED'}")

# ══════════════════════════════════════════════════════════════
# STEP 7: Feature Importance
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7: FEATURE IMPORTANCE (13 features)")
print("=" * 70)

importance = dict(zip(FEATURE_NAMES, rf.feature_importances_))
results['feature_importance'] = {k: float(v) for k, v in sorted(importance.items(), key=lambda x: -x[1])}
for feat, imp in results['feature_importance'].items():
    bar = '█' * int(imp * 50)
    new = ' ← NEW' if 'timestamp' in feat else ''
    print(f"  {feat:35s} {imp:.4f} {bar}{new}")

# Per-dataset generalization
print("\n  Per-Dataset Generalization (combined model → each dataset):")
gen = {}
for name in ['titanic', 'ecommerce', 'hr']:
    gen[name] = evaluate(
        rf, feature_matrices[name], label_vectors[name],
        f'{name.upper()} (all rows)', threshold=global_threshold
    )
results['generalization'] = gen

results['split'] = {
    'train': int(len(X_train)), 'val': int(len(X_val)), 'test': int(len(X_test)),
    'train_after_smote': int(len(X_train_sm)), 'smote_applied': smote_applied,
    'total': int(len(X_all)),
}

results['thresholds'] = {
    'global': global_threshold,
    'per_dataset': per_dataset_thresholds,
}

cv_scores = cross_val_score(rf, X_all, y_all, cv=5, scoring='accuracy')
results['cv_mean'] = float(cv_scores.mean())
results['cv_std'] = float(cv_scores.std())
results['cv_scores'] = [float(s) for s in cv_scores]

# ══════════════════════════════════════════════════════════════
# STEP 8: Save Artifacts
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 8: SAVING ARTIFACTS")
print("=" * 70)

joblib.dump(rf, MODEL_DIR / 'rf_classifier.pkl')
joblib.dump(dt, MODEL_DIR / 'dt_baseline.pkl')
joblib.dump(feature_engineers, MODEL_DIR / 'feature_engineers.pkl')
joblib.dump({'global': global_threshold, 'per_dataset': per_dataset_thresholds},
            MODEL_DIR / 'thresholds.pkl')
print(f"  ✅ RF model       → {MODEL_DIR / 'rf_classifier.pkl'}")
print(f"  ✅ DT baseline    → {MODEL_DIR / 'dt_baseline.pkl'}")
print(f"  ✅ Feature Eng.   → {MODEL_DIR / 'feature_engineers.pkl'}")
print(f"  ✅ Thresholds     → {MODEL_DIR / 'thresholds.pkl'}")

results_path = DATA_DIR / 'phase4_results.json'
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"  ✅ Results JSON   → {results_path}")

split_info = {
    'source_test': {
        'titanic': [int(i) for i, s in enumerate(src_test) if s == 0],
        'ecommerce': [int(i) for i, s in enumerate(src_test) if s == 1],
        'hr': [int(i) for i, s in enumerate(src_test) if s == 2],
    }
}
joblib.dump(split_info, MODEL_DIR / 'split_info.pkl')

print("\n" + "=" * 70)
print("✅ PHASE 4 v2 COMPLETE — 3 fixes applied")
print("=" * 70)
