"""
Multi-format data parsers for CSV, JSON, Parquet, and Excel.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession, DataFrame

from ingestion.config import CSV_OPTIONS, JSON_OPTIONS, PARQUET_OPTIONS, EXCEL_OPTIONS
from ingestion.schema_inference import SchemaInferencer
from ingestion.error_handler import DataIngestionLogger


class BaseParser(ABC):
    """Abstract base parser."""
    
    def __init__(self, spark: SparkSession):
        """Initialize parser."""
        self.spark = spark
        self.logger = DataIngestionLogger(self.__class__.__name__)
    
    @abstractmethod
    def parse(self, file_path: str, **options) -> DataFrame:
        """
        Parse file and return Spark DataFrame.
        
        Args:
            file_path: Path to file
            **options: Format-specific options
            
        Returns:
            Spark DataFrame
        """
        pass


class CSVParser(BaseParser):
    """CSV file parser."""
    
    def parse(self, file_path: str, **options) -> DataFrame:
        """Parse CSV file."""
        self.logger.info(f'Parsing CSV: {file_path}')
        
        # Merge default options with provided options
        csv_opts = {**CSV_OPTIONS, **options}
        
        df = self.spark.read \
            .option('delimiter', csv_opts['delimiter']) \
            .option('header', csv_opts['header']) \
            .option('inferSchema', csv_opts['inferSchema']) \
            .option('encoding', csv_opts['encoding']) \
            .option('nullValue', csv_opts['nullValue']) \
            .option('mode', csv_opts['mode']) \
            .csv(file_path)
        
        self.logger.info(f'Loaded {df.count()} rows from {file_path}')
        return df


class JSONParser(BaseParser):
    """JSON file parser."""
    
    def parse(self, file_path: str, **options) -> DataFrame:
        """Parse JSON file."""
        self.logger.info(f'Parsing JSON: {file_path}')
        
        # Merge default options
        json_opts = {**JSON_OPTIONS, **options}
        
        df = self.spark.read \
            .option('multiline', json_opts['multiline']) \
            .option('encoding', json_opts['encoding']) \
            .option('mode', json_opts['mode']) \
            .json(file_path)
        
        self.logger.info(f'Loaded {df.count()} rows from {file_path}')
        return df


class ParquetParser(BaseParser):
    """Parquet file parser."""
    
    def parse(self, file_path: str, **options) -> DataFrame:
        """Parse Parquet file."""
        self.logger.info(f'Parsing Parquet: {file_path}')
        
        df = self.spark.read \
            .option('mode', PARQUET_OPTIONS['mode']) \
            .parquet(file_path)
        
        self.logger.info(f'Loaded {df.count()} rows from {file_path}')
        return df


class ExcelParser(BaseParser):
    """Excel file parser."""
    
    def parse(self, file_path: str, **options) -> DataFrame:
        """Parse Excel file."""
        self.logger.info(f'Parsing Excel: {file_path}')
        
        # Excel parsing requires 'com.crealytics:spark-excel' package
        # pip install spark-excel
        
        excel_opts = {**EXCEL_OPTIONS, **options}
        
        try:
            df = self.spark.read \
                .format('com.crealytics.spark.excel') \
                .option('header', excel_opts['header']) \
                .option('treatEmptyValuesAsNulls', excel_opts['treatEmptyValuesAsNulls']) \
                .load(file_path)
            
            self.logger.info(f'Loaded {df.count()} rows from {file_path}')
            return df
        
        except Exception as e:
            self.logger.error(f'Excel parsing failed: {e}')
            raise


class ParserFactory:
    """Factory for creating parsers based on file format."""
    
    @staticmethod
    def get_parser(spark: SparkSession, file_path: str) -> BaseParser:
        """
        Get appropriate parser based on file extension.
        
        Args:
            spark: Spark session
            file_path: Path to file
            
        Returns:
            Appropriate parser instance
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext == '.csv':
            return CSVParser(spark)
        elif ext == '.json':
            return JSONParser(spark)
        elif ext == '.parquet':
            return ParquetParser(spark)
        elif ext in ['.xlsx', '.xls']:
            return ExcelParser(spark)
        else:
            raise ValueError(f'Unsupported file format: {ext}')
