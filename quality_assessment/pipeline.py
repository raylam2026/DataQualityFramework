"""
Quality assessment pipeline integrating Phase 2 & Phase 3.
Orchestrates data loading, metric computation, and feature engineering.
"""

from typing import Dict, Tuple, Any
from pyspark.sql import DataFrame, SparkSession
import logging

from ingestion import DataLoader
from quality_assessment.quality_metrics import QualityMetricsComputer
from quality_assessment.feature_engineer import FeatureEngineer
from quality_assessment.feature_vector import FeatureVectorBuilder


class QualityAssessmentPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, spark: SparkSession = None):
        """Initialize pipeline."""
        self.logger = logging.getLogger('QualityAssessmentPipeline')
        
        if spark is None:
            from ingestion.config import SPARK_CONFIG
            spark = SparkSession.builder
            for key, value in SPARK_CONFIG.items():
                if key == 'appName':
                    spark = spark.appName(value)
                elif key == 'master':
                    spark = spark.master(value)
                else:
                    spark = spark.config(key, value)
            spark = spark.getOrCreate()
        
        self.spark = spark
        self.data_loader = DataLoader()
        self.metrics_computer = QualityMetricsComputer(spark)
        self.feature_engineer = FeatureEngineer(spark)
        self.vector_builder = FeatureVectorBuilder(spark)
        
        self.logger.info('QualityAssessmentPipeline initialized')
    
    def run_full_pipeline(
        self,
        file_path: str,
        validate: bool = True
    ) -> Tuple[DataFrame, Dict[str, Any], Dict[str, Any]]:
        """
        Run complete quality assessment pipeline.
        
        Args:
            file_path: Path to data file
            validate: Whether to validate during loading
        
        Returns:
            (df_with_features, metrics, features_dict)
        """
        self.logger.info(f'Starting full pipeline for {file_path}...')
        
        # Phase 2: Load data
        self.logger.info('PHASE 2: Loading data...')
        df = self.data_loader.load_file(file_path, validate=validate)
        self.logger.info(f'Loaded {df.count()} rows, {len(df.columns)} columns')
        
        # Phase 3: Compute quality metrics
        self.logger.info('PHASE 3: Computing quality metrics...')
        metrics = self.metrics_computer.compute_all_metrics(df)
        df_with_row_metrics = self.metrics_computer.compute_row_level_metrics(df)
        
        # Phase 3: Engineer features
        self.logger.info('PHASE 3: Engineering features...')
        df_with_features, col_features, dataset_features = \
            self.feature_engineer.engineer_all_features(df_with_row_metrics)
        
        # Phase 3: Create feature vectors
        self.logger.info('PHASE 3: Creating feature vectors...')
        row_level_cols = [
            'row_completeness', 'row_accuracy',
            'is_duplicate', 'row_timeliness'
        ]
        
        # CRITICAL FIX: Pass ALL dataset-level metrics as dict (not just float)
        dataset_metrics = {
            'completeness': metrics['completeness']['dataset_level'],
            'accuracy': metrics['accuracy']['dataset_level'],
            'consistency': metrics['consistency']['dataset_level'],
            'timeliness': metrics['timeliness']['dataset_level'],
        }
        
        df_final = self.vector_builder.create_feature_vector(
            df_with_features,
            row_level_cols,
            dataset_metrics
        )
        
        self.logger.info('Full pipeline completed successfully')
        
        return df_final, metrics, {
            'column_features': col_features,
            'dataset_features': dataset_features
        }
    
    def run_quick_assessment(self, file_path: str) -> Dict[str, Any]:
        """
        Run quick quality assessment (metrics only, no features).
        
        Returns:
            Dictionary with all computed metrics
        """
        self.logger.info(f'Running quick assessment for {file_path}...')
        
        df = self.data_loader.load_file(file_path, validate=False)
        metrics = self.metrics_computer.compute_all_metrics(df)
        
        return metrics
    
    def save_results(
        self,
        df: DataFrame,
        output_path: str,
        format: str = 'parquet'
    ):
        """Save pipeline results to file."""
        self.logger.info(f'Saving results to {output_path}...')
        self.data_loader.save_dataframe(df, output_path, format=format)
        self.logger.info('Results saved successfully')
    
    def stop(self):
        """Stop Spark session and cleanup."""
        self.data_loader.stop()
        self.logger.info('Pipeline stopped')
