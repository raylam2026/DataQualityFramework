# production/pipeline/ml_classifier.py
"""
ML Classifier for Data Quality Prediction.
PRIMARY  : Spark MLlib RandomForestClassifier (numTrees=100, maxDepth=15)
FALLBACK : sklearn RandomForestClassifier (when PySpark unavailable)
Spec     : 60/20/20 split · 5-fold CV · Decision Tree baseline · SMOTE
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings
import time
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── PySpark MLlib (primary) ───────────────────────────────────────────
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType
    from pyspark.ml.classification import (
        RandomForestClassifier as SparkRF,
        DecisionTreeClassifier as SparkDT,
        RandomForestClassificationModel
    )
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.evaluation import (
        MulticlassClassificationEvaluator,
        BinaryClassificationEvaluator
    )
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    SPARK_AVAILABLE = True
    print("✅ PySpark MLlib available — using Spark RF as primary classifier")
except ImportError:
    SPARK_AVAILABLE = False
    print("⚠️  PySpark not available — using sklearn fallback")

# ── sklearn (fallback + DT baseline) ─────────────────────────────────
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.tree import DecisionTreeClassifier as SklearnDT
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
import joblib

# ── SMOTE ─────────────────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


# ======================================================================
# PRIMARY: Spark MLlib Classifier
# ======================================================================

class SparkMLlibClassifier:
    """
    Spark MLlib Random Forest — spec-compliant primary classifier.
    Spec: numTrees=100, maxDepth=15, 60/20/20 split, 5-fold CV.
    """

    def __init__(self, spark: 'SparkSession' = None, random_state: int = 42):
        if spark is None and SPARK_AVAILABLE:
            try:
                from spark_data_loader import get_spark_session
                self.spark = get_spark_session()
            except ImportError:
                from pyspark.sql import SparkSession as SS
                self.spark = SS.builder \
                    .appName("QualityClassifier") \
                    .master("local[*]") \
                    .getOrCreate()
        else:
            self.spark = spark
        self.random_state     = random_state
        self.model            = None
        self.dt_model         = None
        self.feature_names    = None
        self.training_history = {}

    def _to_spark(self, df: pd.DataFrame, target_col: str):
        feature_cols = sorted([c for c in df.columns
                                if c.endswith('_score') and c != 'quality_score'])
        if 'quality_score' in df.columns:
            feature_cols.append('quality_score')
        self.feature_names = feature_cols

        subset = df[feature_cols + [target_col]].fillna(0.0)
        sdf = self.spark.createDataFrame(subset)
        for c in feature_cols:
            sdf = sdf.withColumn(c, F.col(c).cast(DoubleType()))
        sdf = sdf.withColumn(target_col, F.col(target_col).cast(DoubleType()))

        assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
        return assembler.transform(sdf)

    def train(self, df: pd.DataFrame, target_col: str = 'final_label') -> dict:
        print("\n" + "=" * 70)
        print("SPARK MLLIB RF — TRAINING")
        print(f"  numTrees=100 | maxDepth=15 | 60/20/20 | 5-fold CV")
        print(f"  Records: {len(df)}")
        print("=" * 70)
        t0 = time.time()

        sdf = self._to_spark(df, target_col)
        train_sdf, val_sdf, test_sdf = sdf.randomSplit(
            [0.6, 0.2, 0.2], seed=self.random_state)
        print(f"\n  Train:{train_sdf.count()}  Val:{val_sdf.count()}  Test:{test_sdf.count()}")

        rf = SparkRF(
            numTrees=100, maxDepth=15,
            labelCol=target_col, featuresCol='features',
            seed=self.random_state
        )
        print("\n[Step 1] 5-fold CV on training set...")
        cv = CrossValidator(
            estimator=rf,
            estimatorParamMaps=ParamGridBuilder().build(),
            evaluator=MulticlassClassificationEvaluator(
                labelCol=target_col, predictionCol='prediction', metricName='f1'),
            numFolds=5, seed=self.random_state
        )
        cv_model    = cv.fit(train_sdf)
        cv_f1       = cv_model.avgMetrics
        self.model  = cv_model.bestModel
        print(f"  CV F1: {np.mean(cv_f1):.4f} ± {np.std(cv_f1):.4f}")

        print("\n[Step 2] Training DT baseline (maxDepth=5)...")
        dt = SparkDT(
            maxDepth=5, labelCol=target_col,
            featuresCol='features', seed=self.random_state
        )
        self.dt_model = dt.fit(train_sdf)

        print("\n[Step 3] Evaluating...")
        val_m  = self._metrics(self.model,    val_sdf,  target_col, "Validation (RF)")
        test_m = self._metrics(self.model,    test_sdf, target_col, "Test (RF)")
        dt_m   = self._metrics(self.dt_model, test_sdf, target_col, "Test (DT Baseline)")

        elapsed = time.time() - t0
        self.training_history = {
            'algorithm':       'Spark MLlib RandomForestClassifier',
            'hyperparameters': {'numTrees': 100, 'maxDepth': 15},
            'split':           '60/20/20',
            'train_size':      train_sdf.count(),
            'val_size':        val_sdf.count(),
            'test_size':       test_sdf.count(),
            'total_records':   len(df),
            'cv_folds':        5,
            'cv_f1_mean':      float(np.mean(cv_f1)),
            'cv_f1_std':       float(np.std(cv_f1)),
            'cv_f1_scores':    [float(s) for s in cv_f1],
            'validation':      val_m,
            'test':            test_m,
            'dt_baseline':     dt_m,
            'training_time_s': elapsed,
            'feature_names':   self.feature_names,
        }
        print(f"\n✅ Training complete in {elapsed:.1f}s")
        return test_m

    def _metrics(self, model, sdf, target_col: str, label: str) -> dict:
        preds = model.transform(sdf)
        ev_mc = MulticlassClassificationEvaluator(
            labelCol=target_col, predictionCol='prediction')
        f1   = ev_mc.setMetricName('f1').evaluate(preds)
        acc  = ev_mc.setMetricName('accuracy').evaluate(preds)
        prec = ev_mc.setMetricName('weightedPrecision').evaluate(preds)
        rec  = ev_mc.setMetricName('weightedRecall').evaluate(preds)
        try:
            auc = BinaryClassificationEvaluator(
                labelCol=target_col, rawPredictionCol='rawPrediction',
                metricName='areaUnderROC').evaluate(preds)
        except Exception:
            auc = 0.5

        pd_preds = preds.select(target_col, 'prediction').toPandas()
        try:
            tn, fp, fn, tp = confusion_matrix(
                pd_preds[target_col], pd_preds['prediction']).ravel()
        except Exception:
            tn = fp = fn = 0; tp = len(pd_preds)

        h1 = prec >= 0.80 and rec >= 0.75
        print(f"\n  [{label}]  n={len(pd_preds)}")
        print(f"    Precision: {prec:.4f} {'✅' if prec >= 0.80 else '❌'}")
        print(f"    Recall   : {rec:.4f}  {'✅' if rec  >= 0.75 else '❌'}")
        print(f"    F1-Score : {f1:.4f}   {'✅' if f1   >= 0.77 else '❌'}")
        print(f"    AUC-ROC  : {auc:.4f}")
        print(f"    Accuracy : {acc:.4f}")
        print(f"    H1       : {'✅ PASS' if h1 else '❌ FAIL'}")
        return {
            'split': label, 'precision': float(prec), 'recall': float(rec),
            'f1_score': float(f1), 'auc_roc': float(auc), 'accuracy': float(acc),
            'n_samples': len(pd_preds),
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp),
                                 'fn': int(fn), 'tp': int(tp)},
            'h1_pass': h1
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained.")
        subset = df[self.feature_names].fillna(0.0)
        sdf    = self.spark.createDataFrame(subset)
        for c in self.feature_names:
            sdf = sdf.withColumn(c, F.col(c).cast(DoubleType()))
        sdf   = VectorAssembler(
            inputCols=self.feature_names, outputCol='features').transform(sdf)
        preds = self.model.transform(sdf).select('prediction').toPandas()
        return np.array(['HIGH' if p == 1.0 else 'LOW'
                         for p in preds['prediction']])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained.")
        subset = df[self.feature_names].fillna(0.0)
        sdf    = self.spark.createDataFrame(subset)
        for c in self.feature_names:
            sdf = sdf.withColumn(c, F.col(c).cast(DoubleType()))
        sdf  = VectorAssembler(
            inputCols=self.feature_names, outputCol='features').transform(sdf)
        prob = self.model.transform(sdf).select('probability').toPandas()
        return np.array([row['probability'][1] for _, row in prob.iterrows()])

    def get_feature_importance(self) -> dict:
        if self.model is None:
            raise ValueError("Model not trained.")
        imp = self.model.featureImportances.toArray()
        return dict(sorted(
            zip(self.feature_names, imp), key=lambda x: x[1], reverse=True))

    def save_model(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model not trained.")
        self.model.write().overwrite().save(path)
        print(f"✅ Spark model saved → {path}")

    def load_model(self, path: str) -> None:
        self.model = RandomForestClassificationModel.load(path)
        print(f"✅ Spark model loaded ← {path}")

    def print_summary(self, metrics: dict) -> None:
        print("\n" + "=" * 70)
        print("SPARK MLLIB — SUMMARY")
        print("=" * 70)
        h = self.training_history
        print(f"  Algorithm  : {h.get('algorithm')}")
        print(f"  numTrees=100  maxDepth=15  Split=60/20/20  CV=5-fold")
        print(f"\n  Train:{h.get('train_size')}  Val:{h.get('val_size')}  "
              f"Test:{h.get('test_size')}")
        print(f"  CV F1: {h.get('cv_f1_mean', 0):.4f} ± {h.get('cv_f1_std', 0):.4f}")
        print(f"\n  Test metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k:<12}: {v:.4f}")
        print(f"\n  Top 5 Features:")
        for i, (f, s) in enumerate(
                list(self.get_feature_importance().items())[:5], 1):
            print(f"    {i}. {f}: {s:.4f}")
        print(f"  Training time: {h.get('training_time_s', 0):.1f}s")
        print("=" * 70)


# ======================================================================
# FALLBACK: sklearn Classifier
# ======================================================================

class QualityMLClassifier:
    """
    sklearn fallback — same interface as SparkMLlibClassifier.
    Spec-compliant: 60/20/20, 5-fold CV, numTrees=100, maxDepth=15,
    DT baseline at maxDepth=5, SMOTE for class imbalance.
    """

    def __init__(self, random_state: int = 42):
        self.model            = None
        self.dt_model         = None
        self.feature_names    = None
        self.label_encoder    = None
        self.random_state     = random_state
        self.training_history = {}

    def _prepare(self, df: pd.DataFrame, target_col: str):
        feature_cols = sorted([c for c in df.columns
                                if c.endswith('_score') and c != 'quality_score'])
        if 'quality_score' in df.columns:
            feature_cols.append('quality_score')
        self.feature_names = feature_cols
        X = df[feature_cols].fillna(0)
        y = df[target_col].copy()
        return X, y

    @staticmethod
    def _apply_smote(X, y, name: str = '') -> tuple:
        if not SMOTE_AVAILABLE:
            return X, y
        minority = int(np.sum(y == 0))
        if minority < 2:
            return X, y
        k = min(5, minority - 1)
        try:
            X_res, y_res = SMOTE(
                random_state=42, k_neighbors=k).fit_resample(X, y)
            print(f"  ✅ SMOTE {name}: {len(y)} → {len(y_res)}")
            return X_res, y_res
        except Exception as e:
            print(f"  ⚠️  SMOTE error: {e}")
            return X, y

    def train(self, df: pd.DataFrame,
              target_col: str = 'final_label') -> dict:
        print("\n" + "=" * 70)
        print("SKLEARN RF — TRAINING  (PySpark fallback)")
        print(f"  n_estimators=100 | max_depth=15 | 60/20/20 | 5-fold CV")
        print("=" * 70)
        t0 = time.time()

        X, y = self._prepare(df, target_col)
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y)

        # 60/20/20 split
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y_enc, test_size=0.40,
            random_state=self.random_state, stratify=y_enc)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50,
            random_state=self.random_state, stratify=y_tmp)
        print(f"\n  Train:{len(X_train)}  Val:{len(X_val)}  Test:{len(X_test)}")

        # SMOTE on training set only
        X_train_res, y_train_res = self._apply_smote(
            X_train.values, y_train, 'train')

        # 5-fold CV
        skf = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=self.random_state)
        rf_cv = SklearnRF(
            n_estimators=100, max_depth=15,
            random_state=self.random_state, n_jobs=-1,
            class_weight=None)           # SMOTE handles balance
        cv_scores = cross_val_score(
            rf_cv, X_train_res, y_train_res,
            cv=skf, scoring='f1_weighted', n_jobs=-1)
        print(f"\n  5-fold CV F1: {[f'{s:.4f}' for s in cv_scores]}")
        print(f"  Mean F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Train RF (primary)
        self.model = SklearnRF(
            n_estimators=100, max_depth=15,
            min_samples_split=5, min_samples_leaf=2,
            random_state=self.random_state, n_jobs=-1,
            class_weight=None)           # SMOTE handles balance
        self.model.fit(X_train_res, y_train_res)

        # Train DT baseline (maxDepth=5 — intentionally shallower)
        self.dt_model = SklearnDT(
            max_depth=5,
            random_state=self.random_state,
            class_weight=None)
        self.dt_model.fit(X_train_res, y_train_res)

        val_m  = self._calc(y_val,  self.model.predict(X_val.values),    "Validation (RF)")
        test_m = self._calc(y_test, self.model.predict(X_test.values),   "Test (RF)")
        dt_m   = self._calc(y_test, self.dt_model.predict(X_test.values),"Test (DT Baseline)")

        elapsed = time.time() - t0
        self.training_history = {
            'algorithm':       'sklearn RandomForestClassifier (PySpark fallback)',
            'hyperparameters': {'n_estimators': 100, 'max_depth': 15},
            'split':           '60/20/20',
            'train_size':      int(len(X_train_res)),
            'val_size':        int(len(X_val)),
            'test_size':       int(len(X_test)),
            'total_records':   int(len(X)),
            'cv_folds':        5,
            'cv_f1_mean':      float(cv_scores.mean()),
            'cv_f1_std':       float(cv_scores.std()),
            'cv_f1_scores':    cv_scores.tolist(),
            'validation':      val_m,
            'test':            test_m,
            'dt_baseline':     dt_m,
            'smote_applied':   SMOTE_AVAILABLE,
            'training_time_s': elapsed,
            'feature_names':   self.feature_names,
        }
        print(f"\n✅ Training complete in {elapsed:.1f}s")
        return test_m

    def _calc(self, y_true, y_pred, label: str) -> dict:
        p   = float(precision_score(y_true, y_pred,
                                    zero_division=0, average='weighted'))
        r   = float(recall_score(y_true, y_pred,
                                  zero_division=0, average='weighted'))
        f1  = float(f1_score(y_true, y_pred,
                              zero_division=0, average='weighted'))
        acc = float(accuracy_score(y_true, y_pred))
        try:
            auc = float(roc_auc_score(y_true, y_pred))
        except Exception:
            auc = 0.5
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        except Exception:
            tn = fp = fn = 0; tp = int(np.sum(y_true == 1))

        h1 = p >= 0.80 and r >= 0.75
        print(f"\n  [{label}]  n={len(y_true)}")
        print(f"    Precision: {p:.4f} {'✅' if p >= 0.80 else '❌'}")
        print(f"    Recall   : {r:.4f}  {'✅' if r >= 0.75 else '❌'}")
        print(f"    F1-Score : {f1:.4f}  {'✅' if f1 >= 0.77 else '❌'}")
        print(f"    AUC-ROC  : {auc:.4f}")
        print(f"    Accuracy : {acc:.4f}")
        print(f"    Confusion: TN={int(tn)} FP={int(fp)} FN={int(fn)} TP={int(tp)}")
        print(f"    H1       : {'✅ PASS' if h1 else '❌ FAIL'}")
        return {
            'split': label, 'precision': p, 'recall': r,
            'f1_score': f1, 'auc_roc': auc, 'accuracy': acc,
            'n_samples': int(len(y_true)),
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp),
                                 'fn': int(fn), 'tp': int(tp)},
            'h1_pass': h1
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained.")
        X = df[self.feature_names].fillna(0)
        preds = self.model.predict(X.values)
        return np.array(['HIGH' if p == 1 else 'LOW' for p in preds])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained.")
        X = df[self.feature_names].fillna(0)
        proba = self.model.predict_proba(X.values)
        classes = list(self.model.classes_)
        idx = classes.index(1) if 1 in classes else 1
        return proba[:, idx]

    def get_feature_importance(self) -> dict:
        if self.model is None:
            raise ValueError("Model not trained.")
        return dict(sorted(
            zip(self.feature_names, self.model.feature_importances_),
            key=lambda x: x[1], reverse=True))

    def save_model(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model not trained.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model':          self.model,
            'dt_model':       self.dt_model,
            'feature_names':  self.feature_names,
            'label_encoder':  self.label_encoder,
            'training_history': self.training_history,
        }, path)
        print(f"✅ sklearn model saved → {path}")

    def load_model(self, path: str) -> None:
        data = joblib.load(path)
        self.model            = data['model']
        self.dt_model         = data.get('dt_model')
        self.feature_names    = data['feature_names']
        self.label_encoder    = data.get('label_encoder')
        self.training_history = data.get('training_history', {})
        print(f"✅ sklearn model loaded ← {path}")

    def print_summary(self, metrics: dict) -> None:
        print("\n" + "=" * 70)
        print("SKLEARN RF — SUMMARY  (PySpark fallback)")
        print("=" * 70)
        h = self.training_history
        print(f"  n_estimators=100  max_depth=15  Split=60/20/20  CV=5-fold")
        print(f"  SMOTE: {'✅ applied' if h.get('smote_applied') else '⚠️ skipped'}")
        print(f"\n  Train:{h.get('train_size')}  Val:{h.get('val_size')}  "
              f"Test:{h.get('test_size')}")
        print(f"  CV F1: {h.get('cv_f1_mean', 0):.4f} ± "
              f"{h.get('cv_f1_std', 0):.4f}")
        print(f"\n  Test metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k:<12}: {v:.4f}")
        print(f"\n  Top 5 Feature Importances:")
        for i, (f, s) in enumerate(
                list(self.get_feature_importance().items())[:5], 1):
            print(f"    {i}. {f}: {s:.4f}")
        print(f"  Training time: {h.get('training_time_s', 0):.1f}s")
        print("=" * 70)


# ======================================================================
# Factory — returns correct classifier based on environment
# ======================================================================

def get_classifier(use_spark: bool = None,
                   spark: 'SparkSession' = None,
                   random_state: int = 42):
    """
    Returns SparkMLlibClassifier if PySpark is available,
    otherwise QualityMLClassifier (sklearn fallback).

    Usage:
        clf = get_classifier()
        metrics = clf.train(df, target_col='final_label')
        clf.print_summary(metrics)
    """
    if use_spark is None:
        use_spark = SPARK_AVAILABLE
    if use_spark and SPARK_AVAILABLE:
        print("🔥 Using Spark MLlib classifier")
        return SparkMLlibClassifier(spark=spark, random_state=random_state)
    print("🐍 Using sklearn classifier (PySpark fallback)")
    return QualityMLClassifier(random_state=random_state)


# ======================================================================
# __main__ — quick smoke test
# ======================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("ML CLASSIFIER — SMOKE TEST")
    print("=" * 70)
    print(f"  PySpark  : {'✅ available' if SPARK_AVAILABLE else '⚠️ not available'}")
    print(f"  SMOTE    : {'✅ available' if SMOTE_AVAILABLE else '⚠️ not available'}")

    # Generate synthetic data to verify the pipeline runs end-to-end
    np.random.seed(42)
    n = 300
    synthetic = pd.DataFrame({
        'completeness_score': np.random.beta(8, 2, n),
        'consistency_score':  np.random.beta(7, 3, n),
        'validity_score':     np.random.beta(9, 1, n),
        'accuracy_score':     np.random.beta(6, 4, n),
        'quality_score':      np.random.beta(7, 3, n),
    })
    synthetic['final_label'] = (
        (synthetic['completeness_score'] > 0.5) &
        (synthetic['consistency_score']  > 0.5)
    ).astype(int)

    pos = int(synthetic['final_label'].sum())
    neg = n - pos
    print(f"\n  Synthetic dataset: {n} rows  (HIGH={pos}  LOW={neg})")

    clf = get_classifier(use_spark=False)   # force sklearn for smoke test
    metrics = clf.train(synthetic, target_col='final_label')
    clf.print_summary(metrics)

    # Test save / load round-trip
    model_path = 'data/models/test_model.joblib'
    clf.save_model(model_path)
    clf2 = QualityMLClassifier()
    clf2.load_model(model_path)
    preds = clf2.predict(synthetic.head(5))
    print(f"\n  Sample predictions: {preds}")
    print("\n✅ Smoke test complete")
