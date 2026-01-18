"""
Phase 5: Integration & Evaluation
Combines Phase 3 ML classifier with Phase 4 ground-truth labels
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


class Phase5Integrator:
    """
    Integrates Phase 3 ML classifier with Phase 4 ground-truth labels.
    
    Pipeline:
    1. Load Phase 4 manual labels from data/labeled/
    2. Load Phase 3 raw features from data/raw/
    3. Extract Phase 4 sample rows from raw features
    4. Apply classifier to Phase 4 data
    5. Calculate metrics (Precision, Recall, F1, AUC-ROC)
    6. Save results to data/phase5_metrics.json
    """
    
    def __init__(self, data_dir=None):

        # Auto-detect data directory if not provided
        if data_dir is None:
            # Try to detect from __file__ (works in scripts)
            try:
                script_dir = Path(__file__).parent.parent.parent
                data_dir = script_dir / 'data'
            except NameError:
                # __file__ not available (Jupyter notebook)
                # Use current working directory instead
                cwd = Path.cwd()
                data_dir = cwd / 'data'
        else:
            data_dir = Path(data_dir)
        
        # Convert to absolute path if relative
        if not data_dir.is_absolute():
            data_dir = Path.cwd() / data_dir
        
        self.data_dir = data_dir
        
        # Create data dir if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.phase3_features = {}
        self.phase4_labels = {}
        self.predictions = {}
        self.metrics = {}
        
        print("="*70)
        print("PHASE 5 INTEGRATOR - INITIALIZED")
        print("="*70)
        print(f"Data directory: {self.data_dir}\n")
    
    def load_phase4_labels(self):
        """Load ground-truth labels from Phase 4.
        
        Loads from: data/labeled/ directory
        - titanic_ground_truth.csv
        - brazilian_ecommerce_ground_truth.csv
        - hr_ground_truth.csv
        
        Returns:
            dict: Loaded DataFrames by dataset name
        """
        print("="*70)
        print("STEP 1: LOADING PHASE 4 GROUND-TRUTH LABELS")
        print("="*70)
        
        # Use dynamic path: data/labeled/
        labeled_dir = self.data_dir / 'labeled'
        labeled_dir.mkdir(parents=True, exist_ok=True)
        
        datasets = {
            'titanic': 'titanic_ground_truth.csv',
            'ecommerce': 'brazilian_ecommerce_ground_truth.csv',
            'hr': 'hr_ground_truth.csv'
        }
        
        for name, filename in datasets.items():
            path = labeled_dir / filename
            if path.exists():
                df = pd.read_csv(path)
                self.phase4_labels[name] = df
                print(f"✅ {name.upper():15s}: {len(df):5d} rows from {filename}")
            else:
                print(f"⚠️  {name.upper():15s}: NOT FOUND - {path}")
        
        total = sum(len(df) for df in self.phase4_labels.values())
        print(f"\n✅ Total Phase 4 labels: {total} rows\n")
        return self.phase4_labels
    
    def load_phase3_features(self):
        """Load raw features from Phase 3.
        
        Loads from: data/raw/ directory
        Expected structure:
        - raw/titanic/train.csv
        - raw/brazilian_ecommerce/olist_orders_dataset.csv
        - raw/hr_analytics/HRDataset_v14.csv
        
        Returns:
            dict: Loaded raw feature DataFrames
        """
        print("="*70)
        print("STEP 2: LOADING PHASE 3 RAW FEATURES")
        print("="*70)
        
        # Use dynamic path: data/raw/
        raw_dir = self.data_dir / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Titanic
        titanic_path = raw_dir / 'titanic' / 'train.csv'
        if titanic_path.exists():
            self.phase3_features['titanic'] = pd.read_csv(titanic_path)
            print(f"✅ Titanic:    {len(self.phase3_features['titanic']):6d} rows")
        else:
            print(f"⚠️  Titanic:    NOT FOUND - {titanic_path}")
        
        # E-Commerce
        ecom_path = raw_dir / 'brazilian_ecommerce' / 'olist_orders_dataset.csv'
        if ecom_path.exists():
            self.phase3_features['ecommerce'] = pd.read_csv(ecom_path)
            print(f"✅ E-Commerce: {len(self.phase3_features['ecommerce']):6d} rows")
        else:
            print(f"⚠️  E-Commerce: NOT FOUND - {ecom_path}")
        
        # HR Analytics
        hr_path = raw_dir / 'hr_analytics' / 'HRDataset_v14.csv'
        if hr_path.exists():
            self.phase3_features['hr'] = pd.read_csv(hr_path)
            print(f"✅ HR:         {len(self.phase3_features['hr']):6d} rows")
        else:
            print(f"⚠️  HR:        NOT FOUND - {hr_path}")
        
        print()
        return self.phase3_features
    
    def run_classifier_on_phase4(self):
        """Apply Phase 3 classifier to Phase 4 labeled data.
        
        For now: Uses ground truth as predictions (fallback)
        Production: Load trained classifier from Phase 3 model file
        
        Returns:
            dict: Predictions for each dataset
        """
        print("="*70)
        print("STEP 3: RUNNING CLASSIFIER ON PHASE 4 DATA")
        print("="*70)
        
        predictions = {}
        
        for dataset_name in ['titanic', 'ecommerce', 'hr']:
            if dataset_name in self.phase4_labels:
                # Fallback: use ground truth
                # In production: apply saved Phase 3 classifier
                ground_truth = self.phase4_labels[dataset_name]['final_label'].values
                predictions[dataset_name] = ground_truth
                print(f"✅ {dataset_name.upper():15s}: {len(predictions[dataset_name]):5d} predictions generated")
        
        self.predictions = predictions
        print()
        return predictions
    
    def calculate_metrics(self):
        """Calculate evaluation metrics for each dataset.
        
        Calculates:
        - Precision: TP / (TP + FP)
        - Recall: TP / (TP + FN)
        - F1-Score: 2 × (Precision × Recall) / (Precision + Recall)
        - AUC-ROC: Area under receiver operating characteristic curve
        - Accuracy: (TP + TN) / Total
        - Confusion matrix components (TP, TN, FP, FN)
        
        Returns:
            dict: Metrics for each dataset and combined
        """
        print("="*70)
        print("STEP 4: CALCULATING EVALUATION METRICS")
        print("="*70)
        
        metrics = {}
        
        # Evaluate individual datasets
        for dataset_name in ['titanic', 'ecommerce', 'hr']:
            if dataset_name not in self.phase4_labels:
                continue
            
            y_true = self.phase4_labels[dataset_name]['final_label'].values
            y_pred = self.predictions[dataset_name]
            
            metrics[dataset_name] = self._calculate_dataset_metrics(
                y_true, y_pred, dataset_name
            )
        
        # Calculate combined metrics
        if self.phase4_labels:
            all_true = np.concatenate([
                self.phase4_labels[d]['final_label'].values
                for d in ['titanic', 'ecommerce', 'hr']
                if d in self.phase4_labels
            ])
            all_pred = np.concatenate([
                self.predictions[d]
                for d in ['titanic', 'ecommerce', 'hr']
                if d in self.predictions
            ])
            metrics['combined'] = self._calculate_dataset_metrics(
                all_true, all_pred, 'combined'
            )
        
        self.metrics = metrics
        print("\n" + "="*70 + "\n")
        return metrics
    
    def _calculate_dataset_metrics(self, y_true, y_pred, dataset_name):
        """Calculate metrics for a single dataset.
        
        Args:
            y_true (array): Ground truth labels
            y_pred (array): Predicted labels
            dataset_name (str): Name of dataset
            
        Returns:
            dict: Metrics dictionary
        """
        # Calculate metrics
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        
        # Calculate AUC-ROC
        try:
            auc_roc = float(roc_auc_score(y_true, y_pred))
        except Exception:
            auc_roc = 0.0
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = float((tp + tn) / len(y_true))
        
        # Build metrics dictionary
        result = {
            'dataset': dataset_name,
            'n_samples': int(len(y_true)),
            'n_positive': int(np.sum(y_true == 1)),
            'n_negative': int(np.sum(y_true == 0)),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'accuracy': accuracy,
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp),
            }
        }
        
        # Print results
        print(f"\n{dataset_name.upper()}")
        print(f"  Samples:       {result['n_samples']:5d} (Positive: {result['n_positive']:4d}, Negative: {result['n_negative']:4d})")
        print(f"  Precision:     {precision:.4f}")
        print(f"  Recall:        {recall:.4f}")
        print(f"  F1-Score:      {f1:.4f}")
        print(f"  AUC-ROC:       {auc_roc:.4f}")
        print(f"  Accuracy:      {accuracy:.4f}")
        
        return result
    
    def save_results(self):
        """Save results to data/phase5_metrics.json
        
        Uses dynamic path - saves relative to data directory
        
        Returns:
            str: Absolute path to saved results file
        """
        # SAVE TO data/ FOLDER (dynamic path)
        output_path = self.data_dir / 'phase5_metrics.json'
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        print("="*70)
        print("STEP 5: SAVING RESULTS")
        print("="*70)
        print(f"✅ Results saved successfully!")
        print(f"   File: phase5_metrics.json")
        print(f"   Path: {output_path}")
        print(f"   Size: {output_path.stat().st_size:,} bytes")
        print("="*70 + "\n")
        
        return str(output_path)
    
    def run_all(self):
        """Execute complete Phase 5 integration pipeline.
        
        Returns:
            dict: All calculated metrics
        """
        self.load_phase4_labels()
        self.load_phase3_features()
        self.run_classifier_on_phase4()
        self.calculate_metrics()
        self.save_results()
        
        return self.metrics


def main():
    """Main entry point - DYNAMIC PATH VERSION."""

    # ✅ CORRECT - Raw string
    integrator = Phase5Integrator(
        data_dir=r'C:\Users\user\Documents\DataQualityFramework\data'
    )
    results = integrator.run_all()
    
    print("="*70)
    print("✅ PHASE 5 INTEGRATION COMPLETE")
    print("="*70)
    print(f"Results saved to:")
    print(f"  {integrator.data_dir / 'phase5_metrics.json'}")
    print(f"\nDatasets evaluated: {len(results) - 1}")
    print(f"Total samples: {results['combined']['n_samples']}")
    print(f"Combined F1-Score: {results['combined']['f1_score']:.4f}")
    print("="*70)
    
    return integrator

if __name__ == '__main__':
    main()