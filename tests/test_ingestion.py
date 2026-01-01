"""
Unit tests for data ingestion pipeline.
"""

import pytest
from pathlib import Path
from pyspark.sql import SparkSession

from ingestion import DataLoader, DataValidator
from ingestion.error_handler import MalformedRecordHandler


@pytest.fixture(scope='session')
def spark():
    """Create Spark session for tests."""
    spark = SparkSession.builder \
        .appName('test') \
        .master('local') \
        .config('spark.sql.shuffle.partitions', 4) \
        .getOrCreate()
    
    yield spark
    
    spark.stop()


@pytest.fixture(scope='session')
def loader(spark):
    """Create DataLoader for tests."""
    return DataLoader()


class TestDataLoader:
    """Tests for DataLoader."""
    
    def test_load_csv_file(self, loader):
        """Test loading CSV file."""
        csv_path = 'data/raw/titanic/train.csv'
        
        if Path(csv_path).exists():
            df = loader.load_file(csv_path, validate=False)
            assert df.count() > 0
            assert len(df.columns) > 0
    
    def test_file_not_found(self, loader):
        """Test error handling for missing file."""
        with pytest.raises(Exception):
            loader.load_file('nonexistent/file.csv')
    
    def test_load_with_validation(self, loader):
        """Test loading with validation."""
        csv_path = 'data/raw/titanic/train.csv'
        
        if Path(csv_path).exists():
            df = loader.load_file(csv_path, validate=True)
            assert df.count() > 0


class TestDataValidator:
    """Tests for DataValidator."""
    
    def test_validate_nulls(self, loader):
        """Test null validation."""
        csv_path = 'data/raw/titanic/train.csv'
        
        if Path(csv_path).exists():
            df = loader.load_file(csv_path, validate=False)
            is_valid, report = loader.validator.validate_nulls(df)
            
            assert 'columns' in report
            assert report['total_rows'] > 0
    
    def test_validate_duplicates(self, loader):
        """Test duplicate validation."""
        csv_path = 'data/raw/titanic/train.csv'
        
        if Path(csv_path).exists():
            df = loader.load_file(csv_path, validate=False)
            is_valid, report = loader.validator.validate_duplicates(df)
            
            assert 'duplicate_count' in report
            assert report['total_rows'] > 0
    
    def test_validate_schema(self, loader):
        """Test schema validation."""
        csv_path = 'data/raw/titanic/train.csv'
        
        if Path(csv_path).exists():
            df = loader.load_file(csv_path, validate=False)
            is_valid, report = loader.validator.validate_schema(df)
            
            assert report['is_valid']
            assert 'columns' in report


class TestMalformedRecordHandler:
    """Tests for MalformedRecordHandler."""
    
    def test_record_malformed(self):
        """Test recording malformed records."""
        handler = MalformedRecordHandler('test_dataset')
        
        record = {'id': 1, 'value': 'invalid'}
        handler.record_malformed(record, 'TYPE_ERROR', 'Invalid type')
        
        summary = handler.get_summary()
        assert summary['total_malformed'] == 1
        assert 'TYPE_ERROR' in summary['error_types']
