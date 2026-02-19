"""
Data Ingestion Pipeline (Layer 1) — Spec Design Report Aligned
===============================================================
PySpark reads CSV/JSON from multiple sources into distributed
DataFrames with schema inference and error handling.

Spec: "PySpark session initialization with 4+ cores, distributed local mode"
Spec: "Multi-source loader supporting CSV/JSON with schema inference"
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── PySpark import with graceful fallback ──
try:
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
    from pyspark.sql.types import StructType
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    logger.warning("PySpark not installed. Using pandas fallback. "
                   "Install via: pip install pyspark>=3.5.0")


class DataIngestionPipeline:
    """
    Layer 1: PySpark-based data ingestion for CSV and JSON files.
    Falls back to pandas if PySpark is unavailable.
    """

    def __init__(self, app_name: str = "DataQualityFramework",
                 cores: str = "local[4]",
                 driver_memory: str = "4g"):
        """
        Initialize PySpark session.

        Args:
            app_name: Spark application name
            cores: Spark master URL. "local[4]" = 4 cores distributed local mode
            driver_memory: JVM driver memory allocation
        """
        self.app_name = app_name
        self.cores = cores
        self.driver_memory = driver_memory
        self.spark: Optional[Any] = None
        self._stats: Dict[str, Any] = {}

        if PYSPARK_AVAILABLE:
            self._init_spark()
        else:
            logger.info("Running in pandas-only mode (PySpark fallback)")

    def _init_spark(self):
        """Initialize SparkSession with spec-aligned configuration."""
        self.spark = (
            SparkSession.builder
            .appName(self.app_name)
            .master(self.cores)
            .config("spark.driver.memory", self.driver_memory)
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.session.timeZone", "UTC")
            # Handle malformed records gracefully
            .config("spark.sql.columnNameOfCorruptRecord", "_corrupt_record")
            .getOrCreate()
        )
        self.spark.sparkContext.setLogLevel("WARN")
        sc = self.spark.sparkContext
        logger.info(f"SparkSession initialized: {sc.master}, "
                    f"cores={sc.defaultParallelism}, "
                    f"version={sc.version}")

    # ──────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────

    def ingest(self, path: str, file_format: str = "auto",
               schema: Optional[Any] = None) -> pd.DataFrame:
        """
        Ingest a file (CSV or JSON) and return a pandas DataFrame.

        Args:
            path: Path to the data file
            file_format: "csv", "json", or "auto" (detect from extension)
            schema: Optional PySpark StructType for explicit schema

        Returns:
            pandas DataFrame with ingested data
        """
        path = str(Path(path).resolve())
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

        if file_format == "auto":
            file_format = self._detect_format(path)

        logger.info(f"Ingesting {file_format.upper()}: {path}")

        if self.spark is not None:
            return self._ingest_spark(path, file_format, schema)
        else:
            return self._ingest_pandas(path, file_format)

    def ingest_multi(self, paths: list, file_format: str = "auto") -> Dict[str, pd.DataFrame]:
        """Ingest multiple files. Returns dict of {filename: DataFrame}."""
        results = {}
        for p in paths:
            name = Path(p).stem
            results[name] = self.ingest(p, file_format)
            logger.info(f"  ✅ {name}: {results[name].shape}")
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Return ingestion statistics for the last operation."""
        return self._stats.copy()

    def stop(self):
        """Stop SparkSession gracefully."""
        if self.spark is not None:
            self.spark.stop()
            self.spark = None
            logger.info("SparkSession stopped")

    # ──────────────────────────────────────────────────
    # PySpark ingestion
    # ──────────────────────────────────────────────────

    def _ingest_spark(self, path: str, fmt: str,
                      schema: Optional[StructType]) -> pd.DataFrame:
        """Read via PySpark, log stats, convert to pandas."""
        import time
        t0 = time.time()

        if fmt == "csv":
            reader = self.spark.read.option("header", "true") \
                                     .option("inferSchema", "true") \
                                     .option("mode", "PERMISSIVE") \
                                     .option("columnNameOfCorruptRecord", "_corrupt_record")
            if schema:
                reader = reader.schema(schema)
            sdf = reader.csv(path)

        elif fmt == "json":
            reader = self.spark.read.option("mode", "PERMISSIVE")
            if schema:
                reader = reader.schema(schema)
            # Try JSON Lines first (one object per line), fall back to multi-line JSON array
            sdf = reader.json(path)
            # If Spark couldn't parse any rows, the only column will be _corrupt_record
            # This means the file is likely a multi-line JSON array, not JSON Lines
            parsed_cols = [c for c in sdf.columns if c != "_corrupt_record"]
            if len(parsed_cols) == 0:
                logger.info("  JSON Lines parse failed, retrying with multiLine=true")
                sdf = self.spark.read.option("mode", "PERMISSIVE") \
                                      .option("multiLine", "true") \
                                      .json(path)
        else:
            raise ValueError(f"Unsupported format: {fmt}. Use 'csv' or 'json'.")

        # Handle corrupt records (CSV only — JSON PERMISSIVE mode
        # silently drops malformed rows without _corrupt_record)
        if fmt == "csv" and "_corrupt_record" in sdf.columns:
            sdf = sdf.cache()
            corrupt_count = sdf.filter(sdf["_corrupt_record"].isNotNull()).count()
            if corrupt_count > 0:
                logger.warning(f"  ⚠️ {corrupt_count} malformed records found and logged")
            sdf = sdf.drop("_corrupt_record")

        # Collect stats
        row_count = sdf.count()
        col_count = len(sdf.columns)
        elapsed = time.time() - t0

        self._stats = {
            'engine': 'pyspark',
            'format': fmt,
            'path': path,
            'rows': row_count,
            'columns': col_count,
            'schema': {f.name: str(f.dataType) for f in sdf.schema.fields},
            'ingest_seconds': round(elapsed, 3),
        }


        logger.info(f"  PySpark: {row_count} rows × {col_count} cols "
                    f"in {elapsed:.2f}s ({sdf.rdd.getNumPartitions()} partitions)")

        # Convert to pandas for downstream feature engineering
        pdf = sdf.toPandas()
        return pdf

    # ──────────────────────────────────────────────────
    # Pandas fallback
    # ──────────────────────────────────────────────────

    def _ingest_pandas(self, path: str, fmt: str) -> pd.DataFrame:
        """Fallback ingestion using pandas."""
        import time
        t0 = time.time()

        if fmt == "csv":
            df = pd.read_csv(path)
        elif fmt == "json":
            # Try records-oriented first, then default
            try:
                df = pd.read_json(path, orient='records', lines=True)
            except ValueError:
                try:
                    df = pd.read_json(path, orient='records')
                except ValueError:
                    df = pd.read_json(path)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        elapsed = time.time() - t0
        self._stats = {
            'engine': 'pandas_fallback',
            'format': fmt,
            'path': path,
            'rows': len(df),
            'columns': len(df.columns),
            'ingest_seconds': round(elapsed, 3),
        }

        logger.info(f"  Pandas fallback: {len(df)} rows × {len(df.columns)} cols in {elapsed:.2f}s")
        return df

    # ──────────────────────────────────────────────────
    @staticmethod
    def _detect_format(path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext in ('.csv', '.tsv'):
            return 'csv'
        elif ext in ('.json', '.jsonl', '.ndjson'):
            return 'json'
        else:
            raise ValueError(f"Cannot detect format for extension '{ext}'. "
                             f"Specify file_format='csv' or 'json'.")
