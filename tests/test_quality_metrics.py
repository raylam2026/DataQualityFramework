"""
Unit tests for quality assessment pipeline.
Tests quality metrics, feature engineering, and pipeline integration.
"""

import pytest
from pathlib import Path
from pyspark.sql import SparkSession

from quality_assessment import QualityAssessmentPipeline, QualityMetricsComputer


@pytest.fixture(scope='session')
def spark():
    """Create Spark session for tests."""
    spark = SparkSession.builder \
        .appName('test_quality') \
        .master('local') \
        .config('spark.sql.shuffle.partitions', 4) \
        .getOrCreate()
    
    yield spark
    spark.stop()


@pytest.fixture(scope='session')
def pipeline(spark):
    """Create pipeline for tests."""
    return QualityAssessmentPipeline(spark)


class TestQualityMetrics:
    """Test quality metrics computation."""
    
    def test_completeness_metric(self, pipeline):
        """Test completeness metric."""
        csv_path = 'data/raw/titanic/train.csv'
        
        if Path(csv_path).exists():
            df = pipeline.data_loader.load_file(csv_path)
            completeness = pipeline.metrics_computer.compute_all_metrics(df)
            
            assert 'completeness' in completeness
            assert completeness['completeness']['dataset_level'] >= 0
            assert completeness['completeness']['dataset_level'] <= 100
    
    def test_accuracy_metric(self, pipeline):
        """Test accuracy metric."""
        csv_path = 'data/raw/titanic/train.csv'
        
        if Path(csv_path).exists():
            df = pipeline.data_loader.load_file(csv_path)
            accuracy = pipeline.metrics_computer.compute_all_metrics(df)
            
            assert 'accuracy' in accuracy
            assert accuracy['accuracy']['dataset_level'] >= 0
    
    def test_consistency_metric(self, pipeline):
        """Test consistency metric."""
        csv_path = 'data/raw/titanic/train.csv'
        
        if Path(csv_path).exists():
            df = pipeline.data_loader.load_file(csv_path)
            consistency = pipeline.metrics_computer.compute_all_metrics(df)
            
            assert 'consistency' in consistency
            assert consistency['consistency']['dataset_level'] >= 0


class TestFeatureEngineering:
    """Test feature engineering."""
    
    def test_row_level_features(self, pipeline):
        """Test row-level feature extraction."""
        csv_path = 'data/raw/titanic/train.csv'
        
        if Path(csv_path).exists():
            df = pipeline.data_loader.load_file(csv_path)
            df_features, _, _ = pipeline.feature_engineer.engineer_all_features(df)
            
            assert 'null_count' in df_features.columns
            assert df_features.count() > 0
    
    def test_dataset_level_features(self, pipeline):
        """Test dataset-level feature extraction."""
        csv_path = 'data/raw/titanic/train.csv'
        
        if Path(csv_path).exists():
            df = pipeline.data_loader.load_file(csv_path)
            _, _, dataset_features = pipeline.feature_engineer.engineer_all_features(df)
            
            assert 'row_count' in dataset_features
            assert 'column_count' in dataset_features


class TestFullPipeline:
    """Test full pipeline integration."""
    
    def test_full_pipeline_execution(self, pipeline):
        """Test complete pipeline execution."""
        csv_path = 'data/raw/titanic/train.csv'
        
        if Path(csv_path).exists():
            df_result, metrics, features = \
                pipeline.run_full_pipeline(csv_path, validate=False)
            
            assert df_result.count() > 0
            assert 'completeness' in metrics
            assert 'column_features' in features
    
    def test_quick_assessment(self, pipeline):
        """Test quick assessment."""
        csv_path = 'data/raw/titanic/train.csv'
        
        if Path(csv_path).exists():
            metrics = pipeline.run_quick_assessment(csv_path)
            
            assert 'completeness' in metrics
            assert 'accuracy' in metrics
            assert 'consistency' in metrics
