"""
Feature engineering for data quality assessment.
Extracts row-level, column-level, and dataset-level features.
"""

from typing import Dict, List, Tuple, Any
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, when
import logging

logger = logging.getLogger(__name__)


class RowLevelFeatures:
    """Extract features at row (record) level."""
    
    @staticmethod
    def extract_null_count(df: DataFrame) -> DataFrame:
        """Add column with null count per row."""
        expr = sum([when(col(c).isNull(), 1).otherwise(0) 
                    for c in df.columns])
        
        return df.withColumn('null_count', expr)
    
    @staticmethod
    def extract_data_types_valid(df: DataFrame) -> DataFrame:
        """Add column indicating type validity per row."""
        expr = sum([when(col(c).isNotNull(), 1).otherwise(0) 
                    for c in df.columns])
        
        return df.withColumn('valid_types_count', expr)
    
    @staticmethod
    def extract_all(df: DataFrame) -> DataFrame:
        """Extract all row-level features."""
        logger.info('Extracting row-level features...')
        
        df = RowLevelFeatures.extract_null_count(df)
        df = RowLevelFeatures.extract_data_types_valid(df)
        
        logger.info('Row-level features extracted')
        return df


class ColumnLevelFeatures:
    """Extract features at column (attribute) level."""
    
    @staticmethod
    def extract_null_percentage(df: DataFrame) -> Dict[str, float]:
        """Compute null percentage per column."""
        total_rows = df.count()
        null_percentages = {}
        
        for column in df.columns:
            null_count = df.filter(col(column).isNull()).count()
            null_pct = (null_count / total_rows * 100) \
                if total_rows > 0 else 0
            null_percentages[column] = round(null_pct, 2)
        
        return null_percentages
    
    @staticmethod
    def extract_cardinality(df: DataFrame) -> Dict[str, float]:
        """Compute cardinality (uniqueness) per column."""
        total_rows = df.count()
        cardinality = {}
        
        for column in df.columns:
            unique_count = df.select(column).distinct().count()
            cardinality_ratio = (unique_count / total_rows * 100) \
                if total_rows > 0 else 0
            cardinality[column] = round(cardinality_ratio, 2)
        
        return cardinality
    
    @staticmethod
    def extract_statistics(df: DataFrame) -> Dict[str, Dict]:
        """Extract statistical features per numeric column."""
        stats = {}
        
        for col_name, col_type in df.dtypes:
            if 'double' in col_type or 'integer' in col_type:
                try:
                    col_stats = df.select(col_name).describe().collect()
                    stats[col_name] = {
                        'count': int(col_stats),
                        'mean': float(col_stats),
                        'stddev': float(col_stats),
                        'min': float(col_stats),
                        'max': float(col_stats),
                    }
                except:
                    pass
        
        return stats
    
    @staticmethod
    def extract_all(df: DataFrame) -> Dict[str, Any]:
        """Extract all column-level features."""
        logger.info('Extracting column-level features...')
        
        features = {
            'null_percentages': ColumnLevelFeatures.extract_null_percentage(df),
            'cardinality': ColumnLevelFeatures.extract_cardinality(df),
            'statistics': ColumnLevelFeatures.extract_statistics(df),
        }
        
        logger.info('Column-level features extracted')
        return features


class DatasetLevelFeatures:
    """Extract features at dataset (entire dataset) level."""
    
    @staticmethod
    def extract_row_count(df: DataFrame) -> int:
        """Total number of rows."""
        return df.count()
    
    @staticmethod
    def extract_column_count(df: DataFrame) -> int:
        """Total number of columns."""
        return len(df.columns)
    
    @staticmethod
    def extract_memory_usage(df: DataFrame) -> str:
        """Estimate memory usage."""
        row_count = df.count()
        col_count = len(df.columns)
        estimated_mb = (row_count * col_count * 8) / (1024 * 1024)
        return f'{estimated_mb:.2f} MB'
    
    @staticmethod
    def extract_density(df: DataFrame) -> float:
        """Data density (non-null percentage)."""
        total_cells = df.count() * len(df.columns)
        null_cells = sum([
            df.filter(col(c).isNull()).count() 
            for c in df.columns
        ])
        density = ((total_cells - null_cells) / total_cells * 100) \
            if total_cells > 0 else 0
        return round(density, 2)
    
    @staticmethod
    def extract_all(df: DataFrame) -> Dict[str, Any]:
        """Extract all dataset-level features."""
        logger.info('Extracting dataset-level features...')
        
        features = {
            'row_count': DatasetLevelFeatures.extract_row_count(df),
            'column_count': DatasetLevelFeatures.extract_column_count(df),
            'memory_usage': DatasetLevelFeatures.extract_memory_usage(df),
            'data_density': DatasetLevelFeatures.extract_density(df),
        }
        
        logger.info('Dataset-level features extracted')
        return features


class FeatureEngineer:
    """Main feature engineering orchestrator."""
    
    def __init__(self, spark: SparkSession):
        """Initialize feature engineer."""
        self.spark = spark
        self.logger = logging.getLogger('FeatureEngineer')
    
    def engineer_all_features(
        self, 
        df: DataFrame
    ) -> Tuple[DataFrame, Dict[str, Any], Dict[str, Any]]:
        """Engineer all features (row, column, dataset levels)."""
        self.logger.info('Engineering all features...')
        
        df_with_features = RowLevelFeatures.extract_all(df)
        column_features = ColumnLevelFeatures.extract_all(df)
        dataset_features = DatasetLevelFeatures.extract_all(df)
        
        self.logger.info('All features engineered successfully')
        return df_with_features, column_features, dataset_features
    
    def get_features_summary(
        self,
        column_features: Dict,
        dataset_features: Dict
    ) -> str:
        """Generate feature summary report."""
        report = []
        report.append('\n' + '='*60)
        report.append('FEATURE ENGINEERING SUMMARY')
        report.append('='*60)
        
        report.append('\nDATASET LEVEL FEATURES:')
        report.append('-' * 40)
        for key, value in dataset_features.items():
            report.append(f'{key}: {value}')
        
        report.append('\nCOLUMN LEVEL FEATURES:')
        report.append('-' * 40)
        for key, col_dict in column_features.items():
            report.append(f'\n{key}:')
            for col_name, col_value in col_dict.items():
                report.append(f'  {col_name}: {col_value}')
        
        report.append('\n' + '='*60)
        return '\n'.join(report)
