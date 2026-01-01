"""
Setup Kaggle datasets using kaggle-api.
Requires: pip install kaggle

Usage:
    python scripts/setup_kaggle_datasets.py
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.config import RAW_DATA_DIR


class KaggleDatasetSetup:
    """Setup Kaggle datasets."""
    
    DATASETS = {
        'titanic': {
            'type': 'competition',
            'id': 'titanic',
            'files': ['train.csv', 'test.csv'],
        },
        'brazilian_ecommerce': {
            'type': 'dataset',
            'id': 'olistbr/brazilian-ecommerce',
            'files': [
                'olist_customers_dataset.csv',
                'olist_orders_dataset.csv',
                'olist_order_items_dataset.csv',
                'olist_order_payments_dataset.csv',
                'olist_order_reviews_dataset.csv',
                'olist_products_dataset.csv',
                'olist_sellers_dataset.csv',
                'product_category_name_translation.csv',
            ],
        },
        'hr_analytics': {
            'type': 'dataset',
            'id': 'rhuebner/human-resources-data-set',
            'files': ['HR_Employee_Attrition_Data.csv'],
        },
    }
    
    @staticmethod
    def check_kaggle_cli() -> bool:
        """Check if Kaggle CLI is installed and configured."""
        try:
            result = subprocess.run(
                ['kaggle', '--version'],
                capture_output=True,
                text=True,
                check=True
            )
            print(f'✅ {result.stdout.strip()}')
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    @staticmethod
    def check_credentials() -> bool:
        """Check if Kaggle API credentials are configured."""
        credentials_file = Path.home() / '.kaggle' / 'kaggle.json'
        return credentials_file.exists()
    
    @staticmethod
    def download_dataset(dataset_name: str) -> bool:
        """Download single dataset from Kaggle."""
        if dataset_name not in KaggleDatasetSetup.DATASETS:
            print(f'❌ Unknown dataset: {dataset_name}')
            return False
        
        dataset = KaggleDatasetSetup.DATASETS[dataset_name]
        dataset_dir = RAW_DATA_DIR / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        print(f'\n📥 Downloading {dataset_name}...')
        
        try:
            # Prepare download command
            if dataset['type'] == 'competition':
                cmd = ['kaggle', 'competitions', 'download', '-c', dataset['id']]
            else:
                cmd = ['kaggle', 'datasets', 'download', '-d', dataset['id']]
            
            # Add output path
            cmd.extend(['-p', str(dataset_dir)])
            
            # Download
            print(f'  Running: {" ".join(cmd)}')
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Extract and cleanup
            for zip_file in dataset_dir.glob('*.zip'):
                print(f'  📦 Extracting {zip_file.name}...')
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
                zip_file.unlink()
            
            # Verify files
            missing = []
            for filename in dataset['files']:
                if not (dataset_dir / filename).exists():
                    missing.append(filename)
            
            if missing:
                print(f'  ⚠️  Missing files: {", ".join(missing)}')
                return False
            
            print(f'  ✅ {dataset_name} downloaded successfully')
            return True
        
        except subprocess.CalledProcessError as e:
            print(f'  ❌ Download failed: {e}')
            return False
        except Exception as e:
            print(f'  ❌ Error: {e}')
            return False
    
    @staticmethod
    def setup_all() -> bool:
        """Setup all three datasets."""
        print('╔' + '=' * 58 + '╗')
        print('║  KAGGLE DATASET SETUP - Adaptive ML Data Quality        ║')
        print('╚' + '=' * 58 + '╝')
        
        # Check prerequisites
        print('\n🔍 Checking prerequisites...')
        
        if not KaggleDatasetSetup.check_kaggle_cli():
            print('❌ Kaggle CLI not found')
            print('   Install: pip install kaggle')
            return False
        
        if not KaggleDatasetSetup.check_credentials():
            print('❌ Kaggle API credentials not configured')
            print('   Setup: https://www.kaggle.com/settings/account')
            print('   Click "Create New API Token"')
            print('   Save to ~/.kaggle/kaggle.json')
            return False
        
        print('✅ Prerequisites verified\n')
        
        # Download datasets
        print('🚀 Starting downloads...\n')
        results = {}
        for dataset_name in KaggleDatasetSetup.DATASETS.keys():
            results[dataset_name] = KaggleDatasetSetup.download_dataset(dataset_name)
        
        # Summary
        print('\n' + '=' * 60)
        print('DOWNLOAD SUMMARY')
        print('=' * 60)
        
        for dataset_name, success in results.items():
            status = '✅' if success else '❌'
            print(f'{status} {dataset_name}')
        
        all_success = all(results.values())
        if all_success:
            print(f'\n✅ All datasets downloaded successfully!')
            print(f'📁 Location: {RAW_DATA_DIR}\n')
            
            for dataset_name in KaggleDatasetSetup.DATASETS.keys():
                dataset_dir = RAW_DATA_DIR / dataset_name
                file_count = len(list(dataset_dir.glob('*.csv')))
                print(f'   {dataset_name}: {file_count} CSV files')
        else:
            print(f'\n⚠️  Some datasets failed to download')
        
        return all_success


def main():
    """Main entry point."""
    success = KaggleDatasetSetup.setup_all()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
