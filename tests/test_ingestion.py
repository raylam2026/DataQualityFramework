# tests/test_ingestion.py
"""
Unit tests for data ingestion pipeline.
Tests DataLoader, DataValidator, and MalformedRecordHandler.
"""

import pytest
from pathlib import Path

# conftest.py already added project root to sys.path
from ingestion import DataLoader, DataValidator
from ingestion.error_handler import MalformedRecordHandler

# ---------------------------------------------------------------------------
# Path to test data ¡X adjust if your repo layout differs
# ---------------------------------------------------------------------------
TITANIC_CSV = Path("data/raw/titanic/train.csv")


@pytest.fixture(scope="module")
def loader():
    """Create a DataLoader (it manages its own Spark session internally)."""
    dl = DataLoader()
    yield dl
    dl.stop()


# ========================== DataLoader Tests ===============================

class TestDataLoader:
    """Tests for DataLoader."""

    @pytest.mark.skipif(not TITANIC_CSV.exists(),
                        reason=f"{TITANIC_CSV} not found")
    def test_load_csv_file(self, loader):
        """Test loading CSV file."""
        df = loader.load_file(str(TITANIC_CSV), validate=False)
        assert df.count() > 0
        assert len(df.columns) > 0

    def test_file_not_found(self, loader):
        """Test error handling for missing file."""
        with pytest.raises(Exception):
            loader.load_file("nonexistent/file.csv")

    @pytest.mark.skipif(not TITANIC_CSV.exists(),
                        reason=f"{TITANIC_CSV} not found")
    def test_load_with_validation(self, loader):
        """Test loading with validation enabled."""
        df = loader.load_file(str(TITANIC_CSV), validate=True)
        assert df.count() > 0


# ========================= DataValidator Tests =============================

class TestDataValidator:
    """Tests for DataValidator."""

    @pytest.mark.skipif(not TITANIC_CSV.exists(),
                        reason=f"{TITANIC_CSV} not found")
    def test_validate_nulls(self, loader):
        """Test null validation."""
        df = loader.load_file(str(TITANIC_CSV), validate=False)
        is_valid, report = loader.validator.validate_nulls(df)
        assert "columns" in report
        assert report["total_rows"] > 0

    @pytest.mark.skipif(not TITANIC_CSV.exists(),
                        reason=f"{TITANIC_CSV} not found")
    def test_validate_duplicates(self, loader):
        """Test duplicate validation."""
        df = loader.load_file(str(TITANIC_CSV), validate=False)
        is_valid, report = loader.validator.validate_duplicates(df)
        assert "duplicate_count" in report
        assert report["total_rows"] > 0

    @pytest.mark.skipif(not TITANIC_CSV.exists(),
                        reason=f"{TITANIC_CSV} not found")
    def test_validate_schema(self, loader):
        """Test schema validation."""
        df = loader.load_file(str(TITANIC_CSV), validate=False)
        is_valid, report = loader.validator.validate_schema(df)
        assert report["is_valid"]
        assert "columns" in report


# ===================== MalformedRecordHandler Tests ========================

class TestMalformedRecordHandler:
    """Tests for MalformedRecordHandler."""

    def test_record_malformed(self):
        """Test recording malformed records."""
        handler = MalformedRecordHandler("test_dataset")
        record = {"id": 1, "value": "invalid"}
        handler.record_malformed(record, "TYPE_ERROR", "Invalid type")
        summary = handler.get_summary()
        assert summary["total_malformed"] == 1
        assert "TYPE_ERROR" in summary["error_types"]

    def test_multiple_malformed(self):
        """Test recording multiple malformed records of different types."""
        handler = MalformedRecordHandler("test_dataset")
        handler.record_malformed({"id": 1}, "TYPE_ERROR", "Bad type")
        handler.record_malformed({"id": 2}, "MISSING_VALUE", "Null field")
        handler.record_malformed({"id": 3}, "TYPE_ERROR", "Bad type again")
        summary = handler.get_summary()
        assert summary["total_malformed"] == 3
        assert summary["error_types"]["TYPE_ERROR"] == 2
        assert summary["error_types"]["MISSING_VALUE"] == 1
