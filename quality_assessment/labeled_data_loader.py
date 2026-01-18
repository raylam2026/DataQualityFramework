"""
Phase 4: Load manually labeled ground-truth data.

This module provides a unified interface to load labeled datasets
across Titanic, E-Commerce, and HR Analytics domains.
"""

import pandas as pd
import os
from typing import Tuple


class LabeledDataLoader:
    """Load manually labeled ground-truth data from CSV files."""
    
    def __init__(self, data_dir: str = '../data/'):
        """Initialize with data directory path."""
        self.data_dir = data_dir
        self.labeled_dir = os.path.join(data_dir, 'labeled')
        self.raw_dir = os.path.join(data_dir, 'raw')
    
    def load_labeled_titanic(self) -> pd.DataFrame:
        """Load Titanic labeled data (90 rows)."""
        print("Loading Titanic labeled data...")
        
        labels_path = os.path.join(self.labeled_dir, 'titanic_ground_truth.csv')
        labels = pd.read_csv(labels_path)
        
        raw_path = os.path.join(self.raw_dir, 'titanic', 'train.csv')
        raw_data = pd.read_csv(raw_path)
        
        merged = raw_data.merge(labels, on='PassengerId', how='inner')
        print(f"✅ Loaded {len(merged)} labeled Titanic rows")
        return merged
    
    def load_labeled_ecommerce(self) -> pd.DataFrame:
        """Load E-Commerce labeled data (1,000 rows)."""
        print("Loading E-Commerce labeled data...")
        
        labels_path = os.path.join(self.labeled_dir, 'brazilian_ecommerce_ground_truth.csv')
        labels = pd.read_csv(labels_path)
        
        raw_path = os.path.join(self.raw_dir, 'brazilian_ecommerce', 'olist_orders_dataset.csv')
        raw_data = pd.read_csv(raw_path)
        
        merged = raw_data.merge(labels, on='order_id', how='inner')
        print(f"✅ Loaded {len(merged)} labeled E-Commerce rows")
        return merged
    
    def load_labeled_hr(self) -> pd.DataFrame:
        """Load HR Analytics labeled data (147 rows)."""
        print("Loading HR Analytics labeled data...")
        
        labels_path = os.path.join(self.labeled_dir, 'hr_ground_truth.csv')
        labels = pd.read_csv(labels_path)
        
        raw_path = os.path.join(self.raw_dir, 'hr_analytics', 'HRDataset_v14.csv')
        raw_data = pd.read_csv(raw_path)
        
        merged = raw_data.merge(labels, on='EmpID', how='inner')
        print(f"✅ Loaded {len(merged)} labeled HR rows")
        return merged
    
    def load_all_labeled(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all three labeled datasets."""
        print("\n" + "="*70)
        print("LOADING ALL LABELED DATASETS")
        print("="*70)
        
        titanic = self.load_labeled_titanic()
        ecommerce = self.load_labeled_ecommerce()
        hr = self.load_labeled_hr()
        
        total = len(titanic) + len(ecommerce) + len(hr)
        print(f"\n✅ Total rows loaded: {total}")
        
        return titanic, ecommerce, hr


if __name__ == "__main__":
    loader = LabeledDataLoader()
    titanic, ecommerce, hr = loader.load_all_labeled()
    
    print("\n--- TITANIC SUMMARY ---")
    print(f"Rows: {len(titanic)}")
    print(f"Label distribution:")
    print(titanic['final_label'].value_counts())
    
    print("\n--- ECOMMERCE SUMMARY ---")
    print(f"Rows: {len(ecommerce)}")
    print(f"Label distribution:")
    print(ecommerce['final_label'].value_counts())
    
    print("\n--- HR SUMMARY ---")
    print(f"Rows: {len(hr)}")
    print(f"Label distribution:")
    print(hr['final_label'].value_counts())
