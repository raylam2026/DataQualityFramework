"""
Spark MLlib Classification Pipeline (Layer 3) — Spec Design Report Aligned
==========================================================================
Spark MLlib Random Forest (numTrees=100, maxDepth=15) trained via
PySpark ML Pipeline with VectorAssembler + CrossValidator.

Falls back to sklearn if PySpark/MLlib is unavailable.

Spec: "Random Forest via Spark MLlib (primary)"
Spec: "Decision Tree (optional baseline)"
Spec: "Cross-validation (5-fold) for robust performance estimation"
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ── PySpark ML imports with graceful fallback ──
try:
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.classification import (
        RandomForestClassifier as SparkRF,
        DecisionTreeClassifier as SparkDT,
    )
    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import (
        MulticlassClassificationEvaluator,
        BinaryClassificationEvaluator,
    )
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    SPARK_ML_AVAILABLE = True
except ImportError:
    SPARK_ML_AVAILABLE = False
    logger.warning("PySpark MLlib not available. Using sklearn fallback.")


class SparkMLPipeline:
    """
    Layer 3: Spark MLlib Random Forest classification pipeline.
    Matches spec: numTrees=100, maxDepth=15, 5-fold CV.
    """

    def __init__(self, spark: Optional[Any] = None,
                 num_trees: int = 100, max_depth: int = 15,
                 seed: int = 42):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.seed = seed
        self.spark = spark
        self.model = None
        self.feature_cols = []
        self._use_spark = SPARK_ML_AVAILABLE and (spark is not None)

        if not self._use_spark:
            logger.info("SparkMLPipeline: using sklearn fallback")

    # ──────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray,
              feature_names: list = None) -> Dict[str, Any]:
        """
        Train Random Forest classifier.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Label vector (n_samples,) binary 0/1
            feature_names: Optional list of feature names

        Returns:
            Dict with training metrics and timing
        """
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]
        self.feature_cols = feature_names

        t0 = time.time()

        if self._use_spark:
            result = self._train_spark(X, y, feature_names)
        else:
            result = self._train_sklearn(X, y)

        result['train_seconds'] = round(time.time() - t0, 3)
        return result

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict labels using trained model."""
        if self._use_spark:
            return self._predict_spark(X, threshold)
        else:
            return self._predict_sklearn(X, threshold)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of positive class."""
        if self._use_spark:
            return self._predict_proba_spark(X)
        else:
            return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance dict."""
        if self._use_spark and hasattr(self.model, 'stages'):
            rf_model = self.model.stages[-1]
            importances = rf_model.featureImportances.toArray()
        else:
            importances = self.model.feature_importances_

        return dict(zip(self.feature_cols, importances))

    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                       feature_names: list = None,
                       n_folds: int = 5) -> Dict[str, Any]:
        """Run 5-fold CV matching spec requirements."""
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]

        t0 = time.time()

        if self._use_spark:
            result = self._cv_spark(X, y, feature_names, n_folds)
        else:
            result = self._cv_sklearn(X, y, n_folds)

        result['cv_seconds'] = round(time.time() - t0, 3)
        return result

    # ──────────────────────────────────────────────────
    # Spark MLlib implementation
    # ──────────────────────────────────────────────────

    def _train_spark(self, X, y, feature_names) -> Dict:
        """Train using Spark MLlib Pipeline."""
        sdf = self._numpy_to_spark_df(X, y, feature_names)

        assembler = VectorAssembler(
            inputCols=feature_names,
            outputCol="features",
            handleInvalid="skip"
        )

        rf = SparkRF(
            featuresCol="features",
            labelCol="label",
            numTrees=self.num_trees,
            maxDepth=self.max_depth,
            seed=self.seed,
            # Equivalent to class_weight='balanced' in sklearn:
            # Spark uses weightCol, so we compute weights manually
        )

        pipeline = Pipeline(stages=[assembler, rf])

        # Add sample weights for class imbalance (mirrors sklearn class_weight='balanced')
        pos_count = int(y.sum())
        neg_count = int(len(y) - pos_count)
        total = len(y)
        w_pos = total / (2.0 * pos_count) if pos_count > 0 else 1.0
        w_neg = total / (2.0 * neg_count) if neg_count > 0 else 1.0

        from pyspark.sql.functions import when, col
        sdf = sdf.withColumn(
            "weight",
            when(col("label") == 1, w_pos).otherwise(w_neg)
        )
        rf.setWeightCol("weight")

        self.model = pipeline.fit(sdf)

        # Evaluate on training data
        predictions = self.model.transform(sdf)
        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol="label", metricName="f1"
        )
        evaluator_acc = MulticlassClassificationEvaluator(
            labelCol="label", metricName="accuracy"
        )

        return {
            'engine': 'spark_mllib',
            'train_f1': evaluator_f1.evaluate(predictions),
            'train_accuracy': evaluator_acc.evaluate(predictions),
            'num_trees': self.num_trees,
            'max_depth': self.max_depth,
            'n_samples': len(y),
        }

    def _predict_spark(self, X, threshold) -> np.ndarray:
        """Predict using Spark model with threshold tuning."""
        proba = self._predict_proba_spark(X)
        return (proba >= threshold).astype(int)

    def _predict_proba_spark(self, X) -> np.ndarray:
        """Get probability predictions from Spark model."""
        feature_names = self.feature_cols
        # Create a dummy label column (not used for prediction)
        y_dummy = np.zeros(len(X))
        sdf = self._numpy_to_spark_df(X, y_dummy, feature_names)

        # Add weight column if model expects it
        from pyspark.sql.functions import lit
        sdf = sdf.withColumn("weight", lit(1.0))

        predictions = self.model.transform(sdf)

        # Extract probability of positive class
        from pyspark.sql.functions import udf
        from pyspark.sql.types import FloatType
        extract_prob = udf(lambda v: float(v[1]), FloatType())

        proba_df = predictions.withColumn("prob_pos", extract_prob("probability"))
        proba = np.array(proba_df.select("prob_pos").collect()).flatten()
        return proba

    def _cv_spark(self, X, y, feature_names, n_folds) -> Dict:
        """5-fold CV using Spark ML CrossValidator."""
        sdf = self._numpy_to_spark_df(X, y, feature_names)

        assembler = VectorAssembler(
            inputCols=feature_names,
            outputCol="features",
            handleInvalid="skip"
        )
        rf = SparkRF(
            featuresCol="features",
            labelCol="label",
            numTrees=self.num_trees,
            maxDepth=self.max_depth,
            seed=self.seed,
        )
        pipeline = Pipeline(stages=[assembler, rf])

        paramGrid = ParamGridBuilder().build()  # Using fixed params

        evaluator = BinaryClassificationEvaluator(
            labelCol="label", metricName="areaUnderROC"
        )

        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=paramGrid,
            evaluator=evaluator,
            numFolds=n_folds,
            seed=self.seed,
        )

        cv_model = cv.fit(sdf)
        avg_auc = float(np.mean(cv_model.avgMetrics))

        return {
            'engine': 'spark_mllib',
            'cv_folds': n_folds,
            'cv_avg_auc': avg_auc,
            'cv_metrics': [float(m) for m in cv_model.avgMetrics],
        }

    def _numpy_to_spark_df(self, X, y, feature_names):
        """Convert numpy arrays to Spark DataFrame."""
        data = {fn: X[:, i].tolist() for i, fn in enumerate(feature_names)}
        data['label'] = y.astype(float).tolist()
        pdf = pd.DataFrame(data)
        return self.spark.createDataFrame(pdf)

    # ──────────────────────────────────────────────────
    # sklearn fallback
    # ──────────────────────────────────────────────────

    def _train_sklearn(self, X, y) -> Dict:
        """Fallback: train using sklearn RandomForest."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score, accuracy_score

        self.model = RandomForestClassifier(
            n_estimators=self.num_trees,
            max_depth=self.max_depth,
            class_weight='balanced',
            random_state=self.seed,
            n_jobs=-1,
        )
        self.model.fit(X, y)

        pred = self.model.predict(X)
        return {
            'engine': 'sklearn_fallback',
            'train_f1': float(f1_score(y, pred, zero_division=0)),
            'train_accuracy': float(accuracy_score(y, pred)),
            'num_trees': self.num_trees,
            'max_depth': self.max_depth,
            'n_samples': len(y),
        }

    def _predict_sklearn(self, X, threshold) -> np.ndarray:
        proba = self.model.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def _cv_sklearn(self, X, y, n_folds) -> Dict:
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(
            n_estimators=self.num_trees,
            max_depth=self.max_depth,
            class_weight='balanced',
            random_state=self.seed,
        )
        scores = cross_val_score(rf, X, y, cv=n_folds, scoring='roc_auc')
        return {
            'engine': 'sklearn_fallback',
            'cv_folds': n_folds,
            'cv_avg_auc': float(scores.mean()),
            'cv_metrics': [float(s) for s in scores],
        }
