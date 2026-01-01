"""
Automatic schema inference for CSV/JSON files.
Detects data types and builds Spark schema.
"""

from typing import List, Dict, Optional
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType,
    BooleanType, DateType, TimestampType
)
from datetime import datetime

from ingestion.config import SCHEMA_INFERENCE_CONFIG


class SchemaInferencer:
    """Infer schema from sample data."""
    
    def __init__(self, sample_size: int = 1000):
        """Initialize inferencer."""
        self.sample_size = sample_size
        self.config = SCHEMA_INFERENCE_CONFIG
    
    def infer_type(self, value: str) -> str:
        """
        Infer data type from string value.
        
        Returns: 'int', 'float', 'bool', 'date', 'timestamp', or 'string'
        """
        if value is None or value == '' or value.lower() in ['null', 'none', 'nan']:
            return 'string'  # Default to string for missing values
        
        # Try integer
        try:
            int(value)
            return 'int'
        except ValueError:
            pass
        
        # Try float
        try:
            float(value)
            return 'float'
        except ValueError:
            pass
        
        # Try boolean
        if value.lower() in self.config['boolean_values']['true']:
            return 'bool'
        if value.lower() in self.config['boolean_values']['false']:
            return 'bool'
        
        # Try timestamp
        for fmt in self.config['timestamp_formats']:
            try:
                datetime.strptime(value, fmt)
                return 'timestamp'
            except ValueError:
                pass
        
        # Try date
        for fmt in self.config['date_formats']:
            try:
                datetime.strptime(value, fmt)
                return 'date'
            except ValueError:
                pass
        
        # Default to string
        return 'string'
    
    def infer_column_type(self, column_values: List[str]) -> str:
        """
        Infer type for entire column based on sample.
        
        Uses most common type among non-null values.
        """
        if not column_values:
            return 'string'
        
        type_counts = {}
        for value in column_values:
            inferred_type = self.infer_type(value)
            type_counts[inferred_type] = type_counts.get(inferred_type, 0) + 1
        
        # Return most common type (excluding 'string' if possible)
        non_string_types = {k: v for k, v in type_counts.items() if k != 'string'}
        if non_string_types:
            return max(non_string_types, key=non_string_types.get)
        return 'string'
    
    def build_schema(
        self,
        column_names: List[str],
        sample_rows: List[Dict[str, str]]
    ) -> StructType:
        """
        Build Spark StructType schema from sample data.
        
        Args:
            column_names: List of column names
            sample_rows: Sample of data rows
            
        Returns:
            Spark StructType schema
        """
        spark_type_map = {
            'int': IntegerType(),
            'float': DoubleType(),
            'bool': BooleanType(),
            'date': DateType(),
            'timestamp': TimestampType(),
            'string': StringType(),
        }
        
        fields = []
        for col_name in column_names:
            # Get column values from sample
            column_values = [row.get(col_name, '') for row in sample_rows]
            
            # Infer type
            inferred_type = self.infer_column_type(column_values)
            spark_type = spark_type_map.get(inferred_type, StringType())
            
            # Create field (allow nulls)
            field = StructField(col_name, spark_type, nullable=True)
            fields.append(field)
        
        return StructType(fields)
