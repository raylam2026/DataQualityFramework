"""
Ground truth label generation for ML training data.
Implements labeling criteria from Spec Design Report.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, sum as spark_sum, count
import logging

logger = logging.getLogger(__name__)


class GroundTruthLabeler:
    """Generate binary quality labels (HIGH/LOW) based on spec criteria."""
    
    @staticmethod
    def compute_dimension_scores(df: DataFrame) -> DataFrame:
        """
        Compute per-record scores for 4 quality dimensions.
        
        Returns DataFrame with columns:
        - completeness_score: 1 if ?10% nulls, 0 otherwise
        - consistency_score: 1 if no format issues, 0 otherwise
        - validity_score: 1 if no outliers, 0 otherwise
        - accuracy_score: 1 if no duplicates, 0 otherwise
        """
        total_cols = len(df.columns)
        
        # Completeness: Count nulls per row
        null_cols = sum([
            when(col(c).isNull(), 1).otherwise(0) 
            for c in df.columns
        ])
        
        completeness_score = when(
            (null_cols / total_cols) <= 0.10, 
            1
        ).otherwise(0)
        
        # Consistency: Check for format inconsistencies (simplified)
        consistency_score = 1  # Placeholder - would check date formats, etc.
        
        # Validity: Outlier detection (3 std deviations from mean)
        # NOTE: For heterogeneous data, simplified to no NaN after cast
        validity_score = 1  # Placeholder - would compute per numeric column
        
        # Accuracy: Duplicate detection
        # NOTE: In practice, mark as duplicate if exact match on all columns
        accuracy_score = 1  # Placeholder
        
        df_with_scores = df.withColumn('completeness_score', completeness_score)
        df_with_scores = df_with_scores.withColumn('consistency_score', consistency_score)
        df_with_scores = df_with_scores.withColumn('validity_score', validity_score)
        df_with_scores = df_with_scores.withColumn('accuracy_score', accuracy_score)
        
        return df_with_scores
    
    @staticmethod
    def assign_ground_truth_labels(df: DataFrame) -> DataFrame:
        """
        Assign binary labels based on dimension scores.
        
        HIGH-QUALITY (1): ?3 of 4 dimensions score = 1
        LOW-QUALITY (0): Fails ?2 dimensions (score ?2 total)
        """
        dimension_sum = (
            col('completeness_score') + 
            col('consistency_score') + 
            col('validity_score') + 
            col('accuracy_score')
        )
        
        ground_truth_label = when(dimension_sum >= 3, 1).otherwise(0)
        
        df_labeled = df.withColumn('ground_truth', ground_truth_label)
        
        return df_labeled
    
    @staticmethod
    def stratified_sample(df: DataFrame, sample_fraction: float = 0.10) -> DataFrame:
        """
        Stratified sampling to maintain label distribution.
        
        Args:
            df: DataFrame with ground_truth column
            sample_fraction: Percentage to sample (e.g., 0.10 = 10%)
        
        Returns:
            Sampled DataFrame stratified by label
        """
        # Stratified sample by label
        fractions = {0: sample_fraction, 1: sample_fraction}
        df_sampled = df.sampleBy('ground_truth', fractions=fractions)
        
        logger.info(f'Stratified sample: {df_sampled.count()} rows from {df.count()}')
        
        return df_sampled


class LabelingStrategy:
    """Manual annotation strategy - researcher-based labeling."""
    
    @staticmethod
    def manual_annotation_template(dataset_name: str, sample_size: int) -> str:
        """
        Generate annotation template for manual labeling.
        
        For your 3 datasets:
        - Titanic: ~90 records (10% of 891)
        - Brazilian E-Commerce: ~1,000 records (stratified from 99K+)
        - HR Analytics: ~147 records (10% of 1,470)
        """
        template = f"""
MANUAL ANNOTATION TEMPLATE: {dataset_name}
Total records to label: {sample_size}

LABELING CRITERIA:
================

Dimension 1 - COMPLETENESS (Score=1 if ?10% nulls):
  - Count null/missing values in record
  - If nulls ? 10% of columns → Score = 1
  - Otherwise → Score = 0

Dimension 2 - CONSISTENCY (Score=1 if no format issues):
  - Check date fields are ISO 8601 format (YYYY-MM-DD)
  - Check categorical values match expected enums
  - Check whitespace consistency
  - If all consistent → Score = 1
  - Otherwise → Score = 0

Dimension 3 - VALIDITY (Score=1 if no extreme outliers):
  - Numeric values within 3 std deviations of column mean
  - No extreme values like 99999, -1, etc.
  - If valid → Score = 1
  - Otherwise → Score = 0

Dimension 4 - ACCURACY (Score=1 if no duplicates):
  - Exact match on all columns (except ID) = duplicate
  - No duplicate records found → Score = 1
  - Duplicate found → Score = 0

FINAL LABEL:
  - Sum scores: if ?3 → HIGH-QUALITY (label=1)
  - Sum scores: if ?2 → LOW-QUALITY (label=0)

ANNOTATION OUTPUT:
Record_ID | Completeness | Consistency | Validity | Accuracy | Sum | Label | Notes
{'-'*100}
...
        """
        return template
