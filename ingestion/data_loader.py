"""
Main data loader orchestrator.
Manages Spark session, loads data, validates, and exports.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pyspark.sql import SparkSession, DataFrame

from ingestion.config import SPARK_CONFIG, DATASET_SPECS, RAW_DATA_DIR
from ingestion.error_handler import DataIngestionLogger, MalformedRecordHandler
from ingestion.schema_inference import SchemaInferencer
from ingestion.parsers import ParserFactory
from ingestion.data_validator import DataValidator


class DataLoader:
    """Main data loader for ingestion pipeline."""
    
    def __init__(self):
        """Initialize DataLoader and create Spark session."""
        self.logger = DataIngestionLogger('DataLoader')
        self.spark = self._create_spark_session()
        self.validator = DataValidator(self.spark)
        self.logger.info('DataLoader initialized')
    
    def _create_spark_session(self) -> SparkSession:
        """Create and configure Spark session."""
        self.logger.info('Creating Spark session...')
        
        spark = SparkSession.builder
        
        for key, value in SPARK_CONFIG.items():
            if key == 'appName':
                spark = spark.appName(value)
            elif key == 'master':
                spark = spark.master(value)
            else:
                spark = spark.config(key, value)
        
        spark = spark.getOrCreate()
        
        self.logger.info(f'Spark version: {spark.version}')
        self.logger.info(f'Master: {spark.sparkContext.master}')
        
        return spark
    
    def load_file(
        self,
        file_path: str,
        format: Optional[str] = None,
        validate: bool = False,
        **options
    ) -> DataFrame:
        """
        Load single file.
        
        Args:
            file_path: Path to file
            format: File format (csv, json, parquet, xlsx)
            validate: Whether to validate after loading
            **options: Format-specific options
            
        Returns:
            Spark DataFrame
        """
        self.logger.info(f'Loading file: {file_path}')
        
        file_path = str(file_path)  # Convert Path to string
        
        # Auto-detect format if not specified
        if format is None:
            format = Path(file_path).suffix.lstrip('.')
        
        # Get appropriate parser
        parser = ParserFactory.get_parser(self.spark, file_path)
        
        # Parse file
        df = parser.parse(file_path, **options)
        
        # Validate if requested
        if validate:
            self.logger.info('Running validation...')
            self.validator.validate_all(df)
        
        return df
    
    def load_directory(
        self,
        dir_path: str,
        pattern: str = '*.csv',
        format: str = 'csv',
        validate: bool = False,
        **options
    ) -> Dict[str, DataFrame]:
        """
        Load all files from directory.
        
        Args:
            dir_path: Directory path
            pattern: File pattern (e.g., '*.csv')
            format: File format
            validate: Whether to validate each file
            **options: Format-specific options
            
        Returns:
            Dictionary of {filename: DataFrame}
        """
        self.logger.info(f'Loading directory: {dir_path}')
        
        dir_path = Path(dir_path)
        dfs = {}
        
        for file_path in sorted(dir_path.glob(pattern)):
            if file_path.is_file():
                try:
                    df = self.load_file(
                        str(file_path),
                        format=format,
                        validate=validate,
                        **options
                    )
                    dfs[file_path.name] = df
                except Exception as e:
                    self.logger.error(f'Failed to load {file_path.name}: {e}')
        
        self.logger.info(f'Loaded {len(dfs)} files from directory')
        return dfs
    
    def load_dataset(self, dataset_name: str, validate: bool = False) -> DataFrame:
        """
        Load predefined dataset.
        
        Args:
            dataset_name: 'titanic', 'brazilian_ecommerce', or 'hr_analytics'
            validate: Whether to validate
            
        Returns:
            Spark DataFrame
        """
        if dataset_name not in DATASET_SPECS:
            raise ValueError(f'Unknown dataset: {dataset_name}')
        
        spec = DATASET_SPECS[dataset_name]
        
        self.logger.info(f'Loading dataset: {dataset_name}')
        
        # Handle multi-file datasets
        if 'tables' in spec:
            return self.load_directory(
                spec['path'],
                pattern='*.csv',
                validate=validate
            )
        
        # Handle single-file datasets
        return self.load_file(
            spec['path'],
            format=spec['format'],
            validate=validate,
            delimiter=spec.get('delimiter', ',')
        )
    
    def save_dataframe(
        self,
        df: DataFrame,
        output_path: str,
        format: str = 'parquet',
        mode: str = 'overwrite'
    ):
        """
        Save DataFrame to file.
        
        Args:
            df: Spark DataFrame
            output_path: Output path
            format: Output format (parquet, csv, json)
            mode: Write mode (overwrite, append, ignore, error)
        """
        self.logger.info(f'Saving to {output_path}...')
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.write \
            .format(format) \
            .mode(mode) \
            .save(str(output_path))
        
        self.logger.info(f'Saved {df.count()} rows to {output_path}')
    
    def remove_nulls(self, df: DataFrame) -> DataFrame:
        """Remove rows with any null values."""
        return df.dropna()
    
    def remove_duplicates(self, df: DataFrame) -> DataFrame:
        """Remove duplicate rows."""
        return df.dropDuplicates()
    
    def stop(self):
        """Stop Spark session."""
        self.spark.stop()
        self.logger.info('Spark session stopped')
