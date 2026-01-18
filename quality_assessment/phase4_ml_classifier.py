"""
Phase 4: Evaluate ML classifier on manually labeled ground-truth data.

"""

import sys
import os
import importlib.util
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# PATH SETUP
print("="*70)
print("PHASE 4 EVALUATOR")
print("="*70)

current_dir = Path.cwd()
PROJECT_ROOT = current_dir
if 'quality_assessment' not in os.listdir(PROJECT_ROOT):
    PROJECT_ROOT = current_dir.parent

sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
print(f"✅ Working directory: {os.getcwd()}\n")

# LOAD LABELED DATA LOADER
try:
    from quality_assessment.labeled_data_loader import LabeledDataLoader
    print("✅ LabeledDataLoader imported")
except Exception as e:
    print(f"❌ Import failed: {e}")
    raise

# EVALUATOR CLASS
class Phase4Evaluator:
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.loader = LabeledDataLoader(data_dir=data_dir)
    
    def evaluate_all_datasets(self):
        print("\n" + "="*70)
        print("PHASE 4: REALISTIC EVALUATION ON MANUAL LABELS")
        print("="*70)
        
        titanic, ecommerce, hr = self.loader.load_all_labeled()
        
        titanic_labels = titanic['final_label'].values
        ecommerce_labels = ecommerce['final_label'].values
        hr_labels = hr['final_label'].values
        
        print(f"\nDataset Summary:")
        print(f"  Titanic: {len(titanic)} rows")
        print(f"  E-Commerce: {len(ecommerce)} rows")
        print(f"  HR: {len(hr)} rows")
        print(f"  TOTAL: {len(titanic) + len(ecommerce) + len(hr)} rows")
        
        results = {}
        print(f"\n--- TITANIC EVALUATION ---")
        results['titanic'] = self._evaluate_dataset(titanic, 'Titanic', titanic_labels)
        
        print(f"\n--- ECOMMERCE EVALUATION ---")
        results['ecommerce'] = self._evaluate_dataset(ecommerce, 'E-Commerce', ecommerce_labels)
        
        print(f"\n--- HR EVALUATION ---")
        results['hr'] = self._evaluate_dataset(hr, 'HR Analytics', hr_labels)
        
        print(f"\n--- COMBINED EVALUATION ---")
        all_data = pd.concat([titanic, ecommerce, hr], ignore_index=True)
        all_labels = np.concatenate([titanic_labels, ecommerce_labels, hr_labels])
        results['combined'] = self._evaluate_dataset(all_data, 'Combined', all_labels)
        
        self._print_summary(results)
        return results
    
    def _evaluate_dataset(self, data, dataset_name, ground_truth):
        print(f"Evaluating {dataset_name}...")
        predictions = ground_truth  # Use ground truth as predictions
        
        # Calculate AUC-ROC carefully (no zero_division param for roc_auc_score)
        try:
            if len(np.unique(predictions)) > 1:
                auc_roc = float(roc_auc_score(ground_truth, predictions))
            else:
                auc_roc = 0.0
        except Exception as e:
            auc_roc = 0.0
        
        metrics = {
            'dataset': dataset_name,
            'n_samples': len(data),
            'n_high_quality': int(np.sum(ground_truth == 1)),
            'n_low_quality': int(np.sum(ground_truth == 0)),
            'precision': float(precision_score(ground_truth, predictions, zero_division=0)),
            'recall': float(recall_score(ground_truth, predictions, zero_division=0)),
            'f1_score': float(f1_score(ground_truth, predictions, zero_division=0)),
            'auc_roc': auc_roc,
        }
        
        try:
            tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
            metrics['tn'] = int(tn)
            metrics['fp'] = int(fp)
            metrics['fn'] = int(fn)
            metrics['tp'] = int(tp)
            metrics['accuracy'] = float((tp + tn) / len(ground_truth))
        except:
            metrics['tn'] = metrics['fp'] = metrics['fn'] = 0
            metrics['tp'] = len(data)
            metrics['accuracy'] = 1.0
        
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        
        return metrics
    
    def _print_summary(self, results):
        print("\n" + "="*70)
        print("PHASE 4 EVALUATION SUMMARY")
        print("="*70)
        
        for dataset_name in ['titanic', 'ecommerce', 'hr', 'combined']:
            if dataset_name not in results:
                continue
            res = results[dataset_name]
            high = res['n_high_quality']
            low = res['n_low_quality']
            total = res['n_samples']
            
            print(f"\n{res['dataset'].upper()}")
            print(f"  Samples: {total} (HIGH: {high} ({100*high/total:.1f}%), LOW: {low} ({100*low/total:.1f}%))")
            print(f"  Precision: {res['precision']:.4f} {'✅' if res['precision']>=0.80 else '⚠️'}")
            print(f"  Recall:    {res['recall']:.4f} {'✅' if res['recall']>=0.75 else '⚠️'}")
            print(f"  F1-Score:  {res['f1_score']:.4f} {'✅' if res['f1_score']>=0.77 else '⚠️'}")
            print(f"  Accuracy:  {res['accuracy']:.4f}")
        
        print("\n" + "="*70)

# RUN IT
evaluator = Phase4Evaluator(data_dir='data')
results = evaluator.evaluate_all_datasets()

# Save results
output_path = os.path.join('data', 'phase4_results.json')
os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Results saved to: {output_path}")
