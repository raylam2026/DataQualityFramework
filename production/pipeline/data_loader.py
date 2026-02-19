# phase6_production/pipeline/data_loader.py

import pandas as pd
import json
import os
from pathlib import Path
from typing import Union, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# PySpark integration — optional import with graceful fallback
try:
    from spark_data_loader import SparkDataLoader, get_spark_session
    SPARK_AVAILABLE = True
    logger.info("✅ PySpark available — SparkDataLoader enabled")
except ImportError:
    SPARK_AVAILABLE = False
    logger.warning("⚠️  PySpark not installed — using pandas LabeledDataLoader only")




class LabeledDataLoader:
    """
    Load Phase 4 ground truth labeled data from the data/labeled/ directory.

    This loader handles:
    - Reading CSV files with proper encoding
    - Loading from relative paths (Windows compatible)
    - Validating label structure
    - Combining datasets
    - Works in both module imports AND standalone execution
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize loader with data directory.
        
        Args:
            data_dir: Path to data directory. 
                     - If None: Auto-detect from DataQualityFramework root
                     - If string: Use provided path
        """
        if data_dir is None:
            # Auto-detect data directory - works in both module and standalone
            self.data_dir = self._find_data_directory()
        else:
            self.data_dir = Path(data_dir)
        
        logger.info(f"Data directory: {self.data_dir}")
        self._validate_directory()
    
    def load_all_labeled(self) -> tuple:
        """
        Load all datasets, using SparkDataLoader if PySpark is available,
        falling back to pandas pd.read_csv() otherwise.
    
        Returns:
            (titanic_df, ecommerce_df, hr_df) as pandas DataFrames
        """
        if SPARK_AVAILABLE:
            logger.info("Loading via SparkDataLoader...")
            spark_loader = SparkDataLoader(data_dir=str(self.data_dir))
            _, titanic   = spark_loader.load_titanic()
            _, ecommerce = spark_loader.load_ecommerce()
            _, hr        = spark_loader.load_hr()
            spark_loader.stop()
        else:
            logger.info("Loading via pandas (PySpark not available)...")
            titanic   = self.load_titanic()
            ecommerce = self.load_ecommerce()
            hr        = self.load_hr()
    
        return titanic, ecommerce, hr

    def _find_data_directory(self) -> Path:
        """
        Find the data/labeled directory from any location.
        
        Search priority:
        1. If running as module: use relative path from __file__
        2. If running in DataQualityFramework: find root and use data/labeled
        3. If running elsewhere: look in current directory and parents
        """
        
        # Try 1: Module execution (from phase6_production/pipeline/)
        try:
            current_file = Path(__file__).resolve()
            potential_dir = current_file.parent.parent.parent / "data" / "labeled"
            if potential_dir.exists():
                logger.info(f"Found data via module path: {potential_dir}")
                return potential_dir
        except (NameError, AttributeError):
            pass
        
        # Try 2: Find DataQualityFramework root from current directory
        cwd = Path.cwd()
        
        # Check if we're inside DataQualityFramework
        if "DataQualityFramework" in str(cwd):
            parts = cwd.parts
            try:
                idx = parts.index("DataQualityFramework")
                framework_root = Path(*parts[:idx+1])
                potential_dir = framework_root / "data" / "labeled"
                if potential_dir.exists():
                    logger.info(f"Found data via DataQualityFramework root: {potential_dir}")
                    return potential_dir
            except ValueError:
                pass
        
        # Try 3: Direct data/labeled from current directory
        potential_dir = cwd / "data" / "labeled"
        if potential_dir.exists():
            logger.info(f"Found data in current directory: {potential_dir}")
            return potential_dir
        
        # Try 4: Parent directory
        potential_dir = cwd.parent / "data" / "labeled"
        if potential_dir.exists():
            logger.info(f"Found data in parent directory: {potential_dir}")
            return potential_dir
        
        # Try 5: Two levels up
        potential_dir = cwd.parent.parent / "data" / "labeled"
        if potential_dir.exists():
            logger.info(f"Found data two levels up: {potential_dir}")
            return potential_dir
        
        # Default: return expected standard location
        return cwd / "data" / "labeled"
    
    def _validate_directory(self):
        """Verify data directory exists and contains required files."""
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"Expected: {self.data_dir}\n"
                f"Current working directory: {Path.cwd()}\n"
                f"Please ensure you're running from C:\\Users\\user\\Documents\\DataQualityFramework or subdirectory"
            )
        
        required_files = [
            "titanic_ground_truth.csv",
            "brazilian_ecommerce_ground_truth.csv",
            "hr_ground_truth.csv"
        ]
        
        found_files = []
        missing_files = []
        
        for fname in required_files:
            fpath = self.data_dir / fname
            if fpath.exists():
                found_files.append(fname)
            else:
                missing_files.append(fname)
                logger.warning(f"Expected file not found: {fpath}")
        
        if missing_files:
            logger.warning(f"Missing {len(missing_files)} files: {missing_files}")
        
        logger.info(f"Found {len(found_files)}/{len(required_files)} required files")
    
    def load_titanic(self) -> pd.DataFrame:
        """Load Titanic labeled dataset (90 rows)."""
        fpath = self.data_dir / "titanic_ground_truth.csv"
        logger.info(f"Loading Titanic from {fpath}")
        df = pd.read_csv(fpath, encoding='utf-8')
        logger.info(f"Loaded {len(df)} Titanic rows")
        return df
    
    def load_ecommerce(self) -> pd.DataFrame:
        """Load E-Commerce labeled dataset (1,000 rows)."""
        fpath = self.data_dir / "brazilian_ecommerce_ground_truth.csv"
        logger.info(f"Loading E-Commerce from {fpath}")
        df = pd.read_csv(fpath, encoding='utf-8')
        logger.info(f"Loaded {len(df)} E-Commerce rows")
        return df
    
    def load_hr(self) -> pd.DataFrame:
        """Load HR Analytics labeled dataset (147 rows)."""
        fpath = self.data_dir / "hr_ground_truth.csv"
        logger.info(f"Loading HR from {fpath}")
        df = pd.read_csv(fpath, encoding='utf-8')
        logger.info(f"Loaded {len(df)} HR rows")
        return df
    
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all three labeled datasets.
        
        Returns:
            Tuple of (titanic_df, ecommerce_df, hr_df)
        """
        titanic = self.load_titanic()
        ecommerce = self.load_ecommerce()
        hr = self.load_hr()
        
        total = len(titanic) + len(ecommerce) + len(hr)
        logger.info(f"Total loaded: {total} rows (Titanic: {len(titanic)}, "
                   f"E-Commerce: {len(ecommerce)}, HR: {len(hr)})")
        
        return titanic, ecommerce, hr
    
    def validate_label_structure(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> bool:
        """
        Validate that dataset has correct label columns and values.
        
        Args:
            df: DataFrame to validate
            dataset_name: Name for logging
            
        Returns:
            True if valid, raises exception otherwise
        """
        required_cols = ['completeness', 'consistency', 'validity', 'accuracy', 'final_label']
        
        # Check columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{dataset_name} missing columns: {missing_cols}")
        
        # Check all values are 0 or 1
        for col in required_cols:
            unique_vals = set(df[col].unique())
            if not unique_vals.issubset({0, 1}):
                raise ValueError(f"{dataset_name}.{col} has invalid values: {unique_vals}")
        
        logger.info(f"✅ {dataset_name} label structure valid")
        return True
    
    def validate_all(self):
        """Validate all three datasets."""
        titanic, ecommerce, hr = self.load_all()
        
        self.validate_label_structure(titanic, "Titanic")
        self.validate_label_structure(ecommerce, "E-Commerce")
        self.validate_label_structure(hr, "HR")
        
        logger.info("✅ All datasets valid")
        return True
    
    def get_dataset_info(self):
        """Get information about available datasets."""
        info = {}
        
        try:
            titanic = self.load_titanic()
            info['titanic'] = {
                'rows': len(titanic),
                'columns': len(titanic.columns),
                'high_quality': int((titanic['final_label'] == 1).sum()),
                'low_quality': int((titanic['final_label'] == 0).sum())
            }
        except Exception as e:
            info['titanic'] = {'error': str(e)}
        
        try:
            ecommerce = self.load_ecommerce()
            info['ecommerce'] = {
                'rows': len(ecommerce),
                'columns': len(ecommerce.columns),
                'high_quality': int((ecommerce['final_label'] == 1).sum()),
                'low_quality': int((ecommerce['final_label'] == 0).sum())
            }
        except Exception as e:
            info['ecommerce'] = {'error': str(e)}
        
        try:
            hr = self.load_hr()
            info['hr'] = {
                'rows': len(hr),
                'columns': len(hr.columns),
                'high_quality': int((hr['final_label'] == 1).sum()),
                'low_quality': int((hr['final_label'] == 0).sum())
            }
        except Exception as e:
            info['hr'] = {'error': str(e)}
        
        return info


# Quick test - WORKS STANDALONE AND IN MODULES
if __name__ == "__main__":
    try:
        print("\n" + "="*70)
        print("PHASE 4 LABELED DATA LOADER - TESTING")
        print("="*70 + "\n")
        
        # Initialize loader
        print("🔄 Initializing LabeledDataLoader...")
        loader = LabeledDataLoader()
        print("✅ Loader initialized\n")
        
        print(f"📁 Data directory: {loader.data_dir}\n")
        
        # Load all datasets
        print("📁 Loading all datasets...")
        titanic, ecommerce, hr = loader.load_all()
        print("✅ All datasets loaded\n")
        
        # Validate
        print("✓ Validating label structure...")
        loader.validate_all()
        print("✅ Validation passed\n")
        
        # Print summary
        print("="*70)
        print("PHASE 4 LABELED DATA SUMMARY")
        print("="*70)
        print(f"Titanic:        {len(titanic):>6} rows | Labels: {len(titanic[titanic['final_label']==1])} HIGH, {len(titanic[titanic['final_label']==0])} LOW")
        print(f"E-Commerce:     {len(ecommerce):>6} rows | Labels: {len(ecommerce[ecommerce['final_label']==1])} HIGH, {len(ecommerce[ecommerce['final_label']==0])} LOW")
        print(f"HR:             {len(hr):>6} rows | Labels: {len(hr[hr['final_label']==1])} HIGH, {len(hr[hr['final_label']==0])} LOW")
        print("-"*70)
        
        total = len(titanic) + len(ecommerce) + len(hr)
        high_quality = len(titanic[titanic['final_label']==1]) + len(ecommerce[ecommerce['final_label']==1]) + len(hr[hr['final_label']==1])
        low_quality = total - high_quality
        
        print(f"TOTAL:          {total:>6} rows | Labels: {high_quality} HIGH ({100*high_quality/total:.1f}%), {low_quality} LOW ({100*low_quality/total:.1f}%)")
        print("="*70 + "\n")
        
        # Show dataset info
        print("📊 Dataset Information:")
        print("-"*70)
        info = loader.get_dataset_info()
        import json
        print(json.dumps(info, indent=2))
        
        print("\n✅ TEST COMPLETE - Data loader working perfectly!")
        print(f"✅ Successfully found and loaded data from: {loader.data_dir}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()