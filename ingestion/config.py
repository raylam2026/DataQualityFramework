"""
Configuration for data ingestion pipeline.
Includes PySpark settings, data paths, and format options.
"""

from pathlib import Path
from pyspark.sql import SparkSession

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MALFORMED_DATA_DIR = DATA_DIR / 'malformed_records'

# Log directory
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Create data directories
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MALFORMED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SPARK CONFIGURATION
# ============================================================================

SPARK_CONFIG = {
    'appName': 'AdaptiveMLDataQuality',
    'master': 'local[4]',  # 4+ cores as per requirement
    'spark.driver.memory': '2g',
    'spark.executor.memory': '2g',
    'spark.sql.shuffle.partitions': 4,
    'spark.default.parallelism': 4,
    'spark.sql.adaptive.enabled': 'true',  # Adaptive query execution
    'spark.sql.adaptive.skewJoin.enabled': 'true',
}

# ============================================================================
# DATASET SPECIFICATIONS
# ============================================================================

DATASET_SPECS = {
    'titanic': {
        'path': RAW_DATA_DIR / 'titanic' / 'train.csv',
        'format': 'csv',
        'delimiter': ',',
        'encoding': 'utf-8',
        'header': True,
        'inferSchema': False,  # Use custom schema inference
        'description': 'Titanic disaster passenger data',
        'rows_expected': 891,
        'columns': 12,
    },
    'brazilian_ecommerce': {
        'path': RAW_DATA_DIR / 'brazilian_ecommerce',
        'format': 'csv',
        'delimiter': ',',
        'encoding': 'utf-8',
        'header': True,
        'inferSchema': False,
        'description': 'Brazilian E-Commerce multi-table dataset',
        'tables': [
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
        'path': RAW_DATA_DIR / 'hr_analytics' / 'HRDataset_v14.csv',
        'format': 'csv',
        'delimiter': ',',
        'encoding': 'utf-8',
        'header': True,
        'inferSchema': False,
        'description': 'HR employee attrition dataset',
        'rows_expected': 1470,
        'columns': 35,
    },
}

# ============================================================================
# FORMAT-SPECIFIC OPTIONS
# ============================================================================

CSV_OPTIONS = {
    'delimiter': ',',
    'header': 'true',
    'inferSchema': 'false',
    'encoding': 'utf-8',
    'nullValue': '',
    'mode': 'PERMISSIVE',  # or 'FAILFAST', 'DROPMALFORMED'
}

JSON_OPTIONS = {
    'multiline': 'false',
    'encoding': 'utf-8',
    'mode': 'PERMISSIVE',
}

PARQUET_OPTIONS = {
    'mode': 'PERMISSIVE',
}

EXCEL_OPTIONS = {
    'header': 'true',
    'treatEmptyValuesAsNulls': 'true',
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    'log_dir': LOG_DIR,
    'log_level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'malformed_records_file': MALFORMED_DATA_DIR / 'malformed_records.json',
    'error_log_file': LOG_DIR / 'ingestion_errors.log',
}

# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================

VALIDATION_CONFIG = {
    'check_nulls': True,
    'check_duplicates': True,
    'check_schema': True,
    'check_types': True,
    'null_threshold': 0.5,  # Warn if >50% nulls
    'duplicate_threshold': 0.1,  # Warn if >10% duplicates
}

# ============================================================================
# SCHEMA INFERENCE CONFIGURATION
# ============================================================================

SCHEMA_INFERENCE_CONFIG = {
    'sample_size': 1000,  # Sample size for type inference
    'date_formats': [
        'yyyy-MM-dd',
        'yyyy/MM/dd',
        'MM/dd/yyyy',
        'dd-MM-yyyy',
    ],
    'timestamp_formats': [
        'yyyy-MM-dd HH:mm:ss',
        'yyyy-MM-dd\'T\'HH:mm:ss',
    ],
    'boolean_values': {
        'true': ['true', '1', 'yes', 'y', 'on'],
        'false': ['false', '0', 'no', 'n', 'off'],
    },
}
