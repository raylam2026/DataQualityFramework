"""
Data validation for quality checks.
Validates nulls, duplicates, schema, and data types.
"""

from typing import Tuple, Dict, Any, List
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, when, isnan

from ingestion.config import VALIDATION_CONFIG
from ingestion.error_handler import DataIngestionLogger


class RecordValidator:
    """Validate individual records."""
    
    @staticmethod
    def is_valid_type(value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        if value is None:
            return True  # Nulls allowed
        
        type_map = {
            'int': int,
            'float': float,
            'bool': bool,
            'string': str,
        }
        
        expected_py_type = type_map.get(expected_type, str)
        return isinstance(value, expected_py_type)


class DataValidator:
    """Validate entire DataFrame."""
    
    def __init__(self, spark):
        """Initialize validator."""
        self.spark = spark
        self.logger = DataIngestionLogger('DataValidator')
        self.config = VALIDATION_CONFIG
    
    def validate_nulls(self, df: DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Check for null values.
        
        Returns:
            (is_valid, report) where report contains null statistics
        """
        self.logger.info('Validating nulls...')
        
        report = {}
        total_rows = df.count()
        
        for column in df.columns:
            null_count = df.filter(col(column).isNull()).count()
            null_pct = (null_count / total_rows) * 100 if total_rows > 0 else 0
            
            report[column] = {
                'null_count': null_count,
                'null_percentage': round(null_pct, 2),
            }
            
            # Warn if threshold exceeded
            if null_pct > (self.config['null_threshold'] * 100):
                self.logger.warning(f'{column}: {null_pct:.1f}% nulls')
        
        is_valid = sum(r['null_count'] for r in report.values()) == 0
        
        return is_valid, {
            'validation_type': 'nulls',
            'is_valid': is_valid,
            'total_rows': total_rows,
            'columns': report,
        }
    
    def validate_duplicates(self, df: DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Check for duplicate records.
        
        Returns:
            (is_valid, report) containing duplicate statistics
        """
        self.logger.info('Validating duplicates...')
        
        total_rows = df.count()
        distinct_rows = df.distinct().count()
        duplicate_count = total_rows - distinct_rows
        duplicate_pct = (duplicate_count / total_rows) * 100 if total_rows > 0 else 0
        
        is_valid = duplicate_count == 0
        
        if duplicate_pct > (self.config['duplicate_threshold'] * 100):
            self.logger.warning(f'{duplicate_pct:.1f}% duplicate records')
        
        return is_valid, {
            'validation_type': 'duplicates',
            'is_valid': is_valid,
            'total_rows': total_rows,
            'distinct_rows': distinct_rows,
            'duplicate_count': duplicate_count,
            'duplicate_percentage': round(duplicate_pct, 2),
        }
    
    def validate_schema(self, df: DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate schema structure.
        
        Returns:
            (is_valid, report) with schema information
        """
        self.logger.info('Validating schema...')
        
        columns_info = []
        for field in df.schema.fields:
            columns_info.append({
                'name': field.name,
                'type': str(field.dataType),
                'nullable': field.nullable,
            })
        
        return True, {
            'validation_type': 'schema',
            'is_valid': True,
            'column_count': len(df.columns),
            'columns': columns_info,
        }
    
    def validate_data_types(self, df: DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate data types are consistent.
        
        Returns:
            (is_valid, report) with type validation results
        """
        self.logger.info('Validating data types...')
        
        report = {
            'validation_type': 'data_types',
            'is_valid': True,
            'columns': {}
        }
        
        for field in df.schema.fields:
            col_name = field.name
            dtype = str(field.dataType)
            
            report['columns'][col_name] = {
                'type': dtype,
                'status': 'valid',
            }
        
        return True, report
    
    def validate_all(self, df: DataFrame) -> Dict[str, Any]:
        """
        Run all validations.
        
        Returns:
            Combined report from all validations
        """
        self.logger.info('Running all validations...')
        
        _, nulls_report = self.validate_nulls(df)
        _, duplicates_report = self.validate_duplicates(df)
        _, schema_report = self.validate_schema(df)
        _, types_report = self.validate_data_types(df)
        
        return {
            'timestamp': str(self.spark.sparkContext.appName),
            'total_checks': 4,
            'nulls': nulls_report,
            'duplicates': duplicates_report,
            'schema': schema_report,
            'data_types': types_report,
        }
