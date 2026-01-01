"""
Feature vector creation and normalization.
Combines all features into machine learning-ready vectors.
"""

from typing import Dict, List, Any
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit, struct, array, concat_ws
from pyspark.ml.feature import Normalizer, StandardScaler, VectorAssembler
from pyspark.ml.linalg import Vectors
import logging

logger = logging.getLogger(__name__)


class FeatureVectorBuilder:
    """Build feature vectors from computed metrics and features."""
    
    def __init__(self, spark: SparkSession):
        """Initialize feature vector builder."""
        self.spark = spark
        self.logger = logging.getLogger('FeatureVectorBuilder')
    
    def normalize_features(
        self, 
        df: DataFrame, 
        feature_columns: List[str],
        p: float = 2.0
    ) -> DataFrame:
        """
        Normalize features using L2 normalization (default).
        
        Args:
            df: Input DataFrame
            feature_columns: List of column names to normalize
            p: Norm (1=L1, 2=L2, default=2)
        
        Returns:
            DataFrame with normalized features
        """
        self.logger.info(f'Normalizing {len(feature_columns)} features...')
        
        # Create vector column from feature columns
        assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol='raw_features'
        )
        df_with_vectors = assembler.transform(df)
        
        # Normalize using L2 norm
        normalizer = Normalizer(
            inputCol='raw_features',
            outputCol='normalized_features',
            p=p
        )
        df_normalized = normalizer.transform(df_with_vectors)
        
        # Drop intermediate column
        df_normalized = df_normalized.drop('raw_features')
        
        self.logger.info('Features normalized successfully')
        return df_normalized
    
    def standardize_features(
        self,
        df: DataFrame,
        feature_columns: List[str]
    ) -> DataFrame:
        """
        Standardize features (zero mean, unit variance).
        
        Args:
            df: Input DataFrame
            feature_columns: List of columns to standardize
        
        Returns:
            DataFrame with standardized features
        """
        self.logger.info(f'Standardizing {len(feature_columns)} features...')
        
        # Create vector column
        assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol='raw_features'
        )
        df_with_vectors = assembler.transform(df)
        
        # Standardize
        scaler = StandardScaler(
            inputCol='raw_features',
            outputCol='standardized_features',
            withMean=True,
            withStd=True
        )
        scaler_model = scaler.fit(df_with_vectors)
        df_standardized = scaler_model.transform(df_with_vectors)
        
        # Drop intermediate column
        df_standardized = df_standardized.drop('raw_features')
        
        self.logger.info('Features standardized successfully')
        return df_standardized
    
    def create_feature_vector(
        self,
        df: DataFrame,
        row_level_features: List[str],
        metrics_dataset_level: Dict[str, float]
    ) -> DataFrame:
        """
        Create final feature vector combining all levels.
        
        Returns:
            DataFrame with 'quality_feature_vector' column
        """
        self.logger.info('Creating feature vectors...')
        
        # Normalize row-level features to 0-100 scale
        df_normalized = self.normalize_features(df, row_level_features)
        
        # Add dataset-level metrics as additional columns
        for metric_name, metric_value in metrics_dataset_level.items():
            df_normalized = df_normalized.withColumn(
                f'dataset_{metric_name}',
                lit(metric_value).cast('float')
            )
        
        # Create final feature vector
        all_feature_cols = row_level_features + [
            f'dataset_{name}' 
            for name in metrics_dataset_level.keys()
        ]
        
        assembler = VectorAssembler(
            inputCols=all_feature_cols,
            outputCol='quality_feature_vector'
        )
        df_with_vectors = assembler.transform(df_normalized)
        
        self.logger.info('Feature vectors created successfully')
        return df_with_vectors
    
    def get_vector_statistics(self, df: DataFrame) -> Dict[str, Any]:
        """
        Compute statistics on feature vectors.
        
        Returns:
            Dictionary with vector statistics
        """
        self.logger.info('Computing vector statistics...')
        
        stats = df.select('quality_feature_vector').describe().collect()
        
        return {
            'vector_dimension': len(df.first().quality_feature_vector),
            'vector_count': df.count(),
            'statistics': {
                'mean': float(stats) if len(stats) > 1 else 0,
                'stddev': float(stats) if len(stats) > 2 else 0,
                'min': float(stats) if len(stats) > 3 else 0,
                'max': float(stats) if len(stats) > 4 else 0,
            }
        }
