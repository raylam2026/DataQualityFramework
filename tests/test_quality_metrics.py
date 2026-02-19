# tests/test_quality_metrics.py
"""
Unit tests for quality assessment pipeline.
Tests quality metrics, feature engineering, and pipeline integration.
"""

import pytest
from pathlib import Path

from quality_assessment import QualityAssessmentPipeline, QualityMetricsComputer

# ---------------------------------------------------------------------------
TITANIC_CSV = Path("data/raw/titanic/train.csv")
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline(spark):
    """Create QualityAssessmentPipeline using the shared Spark session."""
    p = QualityAssessmentPipeline(spark)
    yield p
    # Do NOT call p.stop() here ¡X the shared spark fixture owns the session


# ======================== Quality Metrics Tests ============================

class TestQualityMetrics:
    """Test quality metrics computation."""

    @pytest.mark.skipif(not TITANIC_CSV.exists(),
                        reason=f"{TITANIC_CSV} not found")
    def test_completeness_metric(self, pipeline):
        """Test completeness metric."""
        df = pipeline.data_loader.load_file(str(TITANIC_CSV))
        metrics = pipeline.metrics_computer.compute_all_metrics(df)
        assert "completeness" in metrics
        assert 0 <= metrics["completeness"]["dataset_level"] <= 100

    @pytest.mark.skipif(not TITANIC_CSV.exists(),
                        reason=f"{TITANIC_CSV} not found")
    def test_accuracy_metric(self, pipeline):
        """Test accuracy metric."""
        df = pipeline.data_loader.load_file(str(TITANIC_CSV))
        metrics = pipeline.metrics_computer.compute_all_metrics(df)
        assert "accuracy" in metrics
        assert metrics["accuracy"]["dataset_level"] >= 0

    @pytest.mark.skipif(not TITANIC_CSV.exists(),
                        reason=f"{TITANIC_CSV} not found")
    def test_consistency_metric(self, pipeline):
        """Test consistency metric."""
        df = pipeline.data_loader.load_file(str(TITANIC_CSV))
        metrics = pipeline.metrics_computer.compute_all_metrics(df)
        assert "consistency" in metrics
        assert metrics["consistency"]["dataset_level"] >= 0

    @pytest.mark.skipif(not TITANIC_CSV.exists(),
                        reason=f"{TITANIC_CSV} not found")
    def test_timeliness_metric(self, pipeline):
        """Test timeliness metric."""
        df = pipeline.data_loader.load_file(str(TITANIC_CSV))
        metrics = pipeline.metrics_computer.compute_all_metrics(df)
        assert "timeliness" in metrics
        assert metrics["timeliness"]["dataset_level"] >= 0


# ====================== Feature Engineering Tests =========================

class TestFeatureEngineering:
    """Test feature engineering."""

    @pytest.mark.skipif(not TITANIC_CSV.exists(),
                        reason=f"{TITANIC_CSV} not found")
    def test_row_level_features(self, pipeline):
        """Test row-level feature extraction."""
        df = pipeline.data_loader.load_file(str(TITANIC_CSV))
        df_features, _, _ = pipeline.feature_engineer.engineer_all_features(df)
        assert "null_count" in df_features.columns
        assert df_features.count() > 0

    @pytest.mark.skipif(not TITANIC_CSV.exists(),
                        reason=f"{TITANIC_CSV} not found")
    def test_dataset_level_features(self, pipeline):
        """Test dataset-level feature extraction."""
        df = pipeline.data_loader.load_file(str(TITANIC_CSV))
        _, _, dataset_features = pipeline.feature_engineer.engineer_all_features(df)
        assert "row_count" in dataset_features
        assert "column_count" in dataset_features


# ======================== Full Pipeline Tests ==============================

class TestFullPipeline:
    """Test full pipeline integration."""

    @pytest.mark.skipif(not TITANIC_CSV.exists(),
                        reason=f"{TITANIC_CSV} not found")
    def test_full_pipeline_execution(self, pipeline):
        """Test complete pipeline execution."""
        df_result, metrics, features = pipeline.run_full_pipeline(
            str(TITANIC_CSV), validate=False
        )
        assert df_result.count() > 0
        assert "completeness" in metrics
        assert "column_features" in features

    @pytest.mark.skipif(not TITANIC_CSV.exists(),
                        reason=f"{TITANIC_CSV} not found")
    def test_quick_assessment(self, pipeline):
        """Test quick assessment."""
        metrics = pipeline.run_quick_assessment(str(TITANIC_CSV))
        assert "completeness" in metrics
        assert "accuracy" in metrics
        assert "consistency" in metrics
        assert "timeliness" in metrics
