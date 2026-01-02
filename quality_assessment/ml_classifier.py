"""
COMPLETE VERSION OF ml_classifier.py

"""

from typing import Tuple, Dict, Any
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import logging

logger = logging.getLogger(__name__)


class QualityClassifier:
    """Random Forest classifier for data quality assessment."""
    
    def __init__(self, spark: SparkSession):
        """Initialize classifier with Spark session."""
        self.spark = spark
        self.model = None
        self.feature_cols = None
    
    def prepare_features(
        self, 
        df: DataFrame, 
        target_col: str = 'ground_truth'
    ) -> Tuple[DataFrame, list]:
        """
        Prepare feature vectors for ML training.
        
        Includes type casting, NULL handling, vectorization, and scaling.
        """
        # Step 1: Identify feature columns
        exclude_cols = {'row_id', 'ground_truth', target_col, 'dataset', 'split'}
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        logger.info(f'Preparing {len(feature_cols)} features for vectorization')
        
        # Step 2: CAST ALL COLUMNS TO DOUBLEETYPE
        df_cast = df
        for col_name in feature_cols:
            try:
                df_cast = df_cast.withColumn(
                    col_name,
                    col(col_name).cast(DoubleType())
                )
            except Exception as e:
                logger.warning(f'Initial cast failed for {col_name}: {e}')
                df_cast = df_cast.withColumn(
                    col_name,
                    when(col(col_name).isNull(), 0.0).otherwise(
                        col(col_name).cast(DoubleType())
                    )
                )
        
        # Step 3: REPLACE NULLS WITH 0.0
        for col_name in feature_cols:
            df_cast = df_cast.withColumn(
                col_name,
                when(col(col_name).isNull(), 0.0).otherwise(col(col_name))
            )
        
        logger.info(f'Type casting complete: {len(feature_cols)} columns → DoubleType')
        
        # Step 4: Vector assembler
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol='features',
            handleInvalid='skip'
        )
        
        df_with_vectors = assembler.transform(df_cast)
        logger.info(f'VectorAssembly complete: {df_with_vectors.count()} rows vectorized')
        
        # Step 5: Scaling
        scaler = StandardScaler(
            inputCol='features',
            outputCol='scaled_features',
            withMean=True,
            withStd=True
        )
        
        scaler_model = scaler.fit(df_with_vectors)
        df_scaled = scaler_model.transform(df_with_vectors)
        
        self.feature_cols = feature_cols
        
        logger.info(f'Feature preparation complete: {len(feature_cols)} features, {df_scaled.count()} rows')
        
        return df_scaled, feature_cols
    
    def train_random_forest(
        self,
        df_train: DataFrame,
        target_col: str = 'ground_truth',
        num_trees: int = 100,
        max_depth: int = 15,
        seed: int = 42
    ) -> Tuple[RandomForestClassificationModel, Dict[str, float]]:
        """
        Train Random Forest classifier on training data.
        
        Per Specification: numTrees=100, maxDepth=15, seed=42
        """
        logger.info(f'Preparing training data with target column: {target_col}')
        df_scaled, feature_cols = self.prepare_features(df_train, target_col)
        self.feature_cols = feature_cols
        
        logger.info(f'Training Random Forest: {num_trees} trees, max_depth={max_depth}')
        
        # Random Forest with spec parameters
        rf = RandomForestClassifier(
            featuresCol='scaled_features',
            labelCol=target_col,
            numTrees=num_trees,
            maxDepth=max_depth,
            seed=seed,
        )
        
        # Train model
        model = rf.fit(df_scaled)
        logger.info(f'Model training complete')
        
        # Evaluate on training set
        predictions = model.transform(df_scaled)
        
        # AUC-ROC metric
        evaluator_binary = BinaryClassificationEvaluator(
            rawPredictionCol='rawPrediction',
            labelCol=target_col,
            metricName='areaUnderROC'
        )
        auc = evaluator_binary.evaluate(predictions)
        
        # F1-Score metric
        evaluator_multi = MulticlassClassificationEvaluator(
            predictionCol='prediction',
            labelCol=target_col,
            metricName='f1'
        )
        f1 = evaluator_multi.evaluate(predictions)
        
        metrics = {
            'auc_roc': float(auc),
            'f1_score': float(f1),
            'training_samples': df_scaled.count()
        }
        
        logger.info(f'Training Metrics: AUC={auc:.4f}, F1={f1:.4f}, Samples={metrics["training_samples"]}')
        
        self.model = model
        return model, metrics
    
    def evaluate_model(
        self,
        df_test: DataFrame,
        model: RandomForestClassificationModel,
        target_col: str = 'ground_truth'
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of model on test set.
        
        Returns: precision, recall, f1_score, auc_roc, test_samples
        """
        logger.info(f'Evaluating model on test set ({df_test.count()} samples)')
        
        # Step 1: Identify features
        exclude_cols = {'row_id', 'ground_truth', target_col, 'dataset', 'split'}
        feature_cols = [c for c in df_test.columns if c not in exclude_cols]
        
        # Step 2: Cast to numeric (same as training)
        df_cast = df_test
        for col_name in feature_cols:
            try:
                df_cast = df_cast.withColumn(
                    col_name,
                    col(col_name).cast(DoubleType())
                )
            except Exception as e:
                logger.warning(f'Cast failed for {col_name}: {e}')
                df_cast = df_cast.withColumn(
                    col_name,
                    when(col(col_name).isNull(), 0.0).otherwise(
                        col(col_name).cast(DoubleType())
                    )
                )
        
        # Step 3: Replace NULLs
        for col_name in feature_cols:
            df_cast = df_cast.withColumn(
                col_name,
                when(col(col_name).isNull(), 0.0).otherwise(col(col_name))
            )
        
        # Step 4: Vectorize
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol='features',
            handleInvalid='skip'
        )
        df_with_features = assembler.transform(df_cast)
        
        # Step 5: Scale
        scaler = StandardScaler(
            inputCol='features',
            outputCol='scaled_features',
            withMean=True,
            withStd=True
        )
        scaler_model = scaler.fit(df_with_features)
        df_scaled = scaler_model.transform(df_with_features)
        
        # Step 6: Make predictions
        predictions = model.transform(df_scaled)
        
        # Step 7: Evaluate with multiple metrics
        evaluator_precision = MulticlassClassificationEvaluator(
            predictionCol='prediction',
            labelCol=target_col,
            metricName='weightedPrecision'
        )
        evaluator_recall = MulticlassClassificationEvaluator(
            predictionCol='prediction',
            labelCol=target_col,
            metricName='weightedRecall'
        )
        evaluator_f1 = MulticlassClassificationEvaluator(
            predictionCol='prediction',
            labelCol=target_col,
            metricName='f1'
        )
        evaluator_auc = BinaryClassificationEvaluator(
            rawPredictionCol='rawPrediction',
            labelCol=target_col,
            metricName='areaUnderROC'
        )
        
        metrics = {
            'precision': float(evaluator_precision.evaluate(predictions)),
            'recall': float(evaluator_recall.evaluate(predictions)),
            'f1_score': float(evaluator_f1.evaluate(predictions)),
            'auc_roc': float(evaluator_auc.evaluate(predictions)),
            'test_samples': df_scaled.count()
        }
        
        # Check spec criteria (H1)
        spec_met = (
            metrics['precision'] >= 0.80 and
            metrics['recall'] >= 0.75 and
            metrics['f1_score'] >= 0.77
        )
        
        logger.info(f'Test Metrics:')
        logger.info(f'  Precision: {metrics["precision"]:.4f} (need ≥ 0.80)')
        logger.info(f'  Recall: {metrics["recall"]:.4f} (need ≥ 0.75)')
        logger.info(f'  F1-Score: {metrics["f1_score"]:.4f} (need ≥ 0.77)')
        logger.info(f'  AUC-ROC: {metrics["auc_roc"]:.4f}')
        logger.info(f'Spec H1 Criteria Met: {spec_met}')
        
        return metrics
    
    def feature_importance(
        self, 
        model: RandomForestClassificationModel
    ) -> Dict[str, float]:
        """Extract and rank feature importance from trained model."""
        if self.feature_cols is None:
            logger.warning('Feature columns not set. Cannot compute importance.')
            return {}
        
        importances = model.featureImportances.toArray()
        feature_importance_dict = {
            self.feature_cols[i]: float(importances[i])
            for i in range(len(self.feature_cols))
        }
        
        # Sort by importance (descending)
        sorted_features = sorted(
            feature_importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        logger.info('Top 10 Feature Importances:')
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            logger.info(f'  {i}. {feature}: {importance:.4f}')
        
        return dict(sorted_features[:10])
    
    def cross_validate(
        self,
        df: DataFrame,
        num_folds: int = 5,
        target_col: str = 'ground_truth'
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Manual k-fold cross-validation (spec-compliant).
        
        Implements 5-fold cross-validation without Spark MLlib CrossValidator
        to avoid deadlock issues while maintaining statistical rigor.
        """
        from pyspark.sql.window import Window
        from pyspark.sql.functions import row_number, rand, col
        
        logger.info(f'Starting {num_folds}-fold cross-validation (manual k-fold)')
        
        # STEP 1: Add fold numbers to each row
        window = Window.orderBy(rand())
        df_with_fold = df.withColumn(
            'fold_id', 
            (row_number().over(window) % num_folds) + 1
        )
        logger.info(f'Dataset split into {num_folds} folds')
        
        # STEP 2: Run cross-validation for each fold
        all_f1_scores = []
        all_precisions = []
        all_recalls = []
        all_auc_scores = []
        
        for fold_num in range(1, num_folds + 1):
            logger.info(f'Fold {fold_num}/{num_folds}')
            
            # Split data: training = all except this fold, testing = this fold
            train_fold = df_with_fold.filter(col('fold_id') != fold_num)
            test_fold = df_with_fold.filter(col('fold_id') == fold_num)
            
            train_count = train_fold.count()
            test_count = test_fold.count()
            logger.info(f'  Training samples: {train_count}')
            logger.info(f'  Testing samples: {test_count}')
            
            # Train model on this fold's training data
            model, train_metrics = self.train_random_forest(
                train_fold, 
                target_col=target_col
            )
            
            # Evaluate on this fold's test data
            test_metrics = self.evaluate_model(
                test_fold, 
                model, 
                target_col=target_col
            )
            
            # Collect metrics from this fold
            all_f1_scores.append(test_metrics['f1_score'])
            all_precisions.append(test_metrics['precision'])
            all_recalls.append(test_metrics['recall'])
            all_auc_scores.append(test_metrics['auc_roc'])
            
            logger.info(f'  Fold {fold_num} Metrics:')
            logger.info(f'    - F1: {test_metrics["f1_score"]:.4f}')
            logger.info(f'    - Precision: {test_metrics["precision"]:.4f}')
            logger.info(f'    - Recall: {test_metrics["recall"]:.4f}')
        
        # STEP 3: Calculate mean metrics across all folds
        mean_f1 = float(sum(all_f1_scores) / len(all_f1_scores))
        mean_precision = float(sum(all_precisions) / len(all_precisions))
        mean_recall = float(sum(all_recalls) / len(all_recalls))
        mean_auc = float(sum(all_auc_scores) / len(all_auc_scores))
        
        # Log final results
        logger.info(f'{num_folds}-Fold Cross-Validation Complete')
        logger.info(f'  Mean F1-Score: {mean_f1:.4f}')
        logger.info(f'  Mean Precision: {mean_precision:.4f}')
        logger.info(f'  Mean Recall: {mean_recall:.4f}')
        logger.info(f'  Mean AUC-ROC: {mean_auc:.4f}')
        
        # Return results
        mean_metrics = {
            'f1_score': mean_f1,
            'precision': mean_precision,
            'recall': mean_recall,
            'auc_roc': mean_auc
        }
        
        return mean_metrics, {}
