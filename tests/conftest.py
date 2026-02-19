# tests/conftest.py
"""
Pytest configuration and shared fixtures.
Adds project root to sys.path so that 'ingestion' and 'quality_assessment'
packages are importable from the tests/ directory.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# PATH SETUP ¡X ensures 'import ingestion' and 'import quality_assessment' work
# regardless of where pytest is invoked from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """Create a single Spark session shared across the entire test run."""
    spark = (
        SparkSession.builder
        .appName("DataQualityFramework-Tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", 2)
        .config("spark.ui.enabled", "false")       # faster in CI
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )
    yield spark
    spark.stop()
