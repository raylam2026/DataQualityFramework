# phase6_production/pipeline/spark_data_loader.py
# PySpark-powered data loader for Phase 6 production pipeline.
# Replaces pandas pd.read_csv() with spark.read.csv() for scalable ingestion.

from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql import functions as F
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


import os

def get_spark_session(app_name: str = "DataQualityFramework") -> SparkSession:
    """
    Create or retrieve existing SparkSession.
    - spark.local.dir: fixes Windows ShutdownHookManager NoSuchFileException
    - spark.sql.execution.arrow.pyspark.enabled: fast toPandas() via Arrow
    """
    # Create a dedicated temp dir to avoid Windows cleanup race condition
    spark_tmp = os.path.join(os.environ.get("TEMP", "C:/tmp"), "spark_tmp")
    os.makedirs(spark_tmp, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.local.dir", spark_tmp)                        # ✅ FIX 3: Windows temp dir
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.debug.maxToStringFields", "100")           # suppress truncation WARN
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

class SparkDataLoader:
    """
    PySpark-powered data loader for Phase 4 labeled ground-truth datasets.

    Replaces LabeledDataLoader (pandas) with a fully distributed ingestion layer.

    Pipeline:
        CSV on disk
            └─► spark.read.csv()     [Spark: scalable I/O]
                └─► validate()       [Spark: groupBy, filter, count]
                    └─► toPandas()   [Bridge: Arrow-optimised]
                        └─► sklearn  [pandas: RandomForest training]

    For production at petabyte scale: swap CSV paths for HDFS / S3 URIs.
    The rest of the pipeline is unchanged.
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.spark = get_spark_session()
        self.data_dir = self._resolve_data_dir(data_dir)
        print("=" * 60)
        print("SPARK DATA LOADER — INITIALISED")
        print("=" * 60)
        print(f"  Spark version : {self.spark.version}")
        print(f"  Master        : {self.spark.sparkContext.master}")
        print(f"  Data dir      : {self.data_dir}")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve_data_dir(self, data_dir: Optional[str]) -> Path:
        if data_dir:
            return Path(data_dir)
        try:
            # Running as module inside phase6_production/pipeline/
            return Path(__file__).resolve().parent.parent.parent / "data" / "labeled"
        except NameError:
            return Path.cwd() / "data" / "labeled"

    # ------------------------------------------------------------------
    # Core Spark loader
    # ------------------------------------------------------------------

    def _load_spark_df(self, filename: str) -> SparkDataFrame:
        """
        Read CSV into a Spark DataFrame.
        Fixes applied:
          - FIX 1: Cast all label columns to IntegerType (prevents float 0.0/1.0)
          - FIX 2: Drop phantom _c* columns caused by trailing commas in CSV header
        """
        from pyspark.sql.types import IntegerType
    
        path = str(self.data_dir / filename)
        sdf = (
            self.spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .option("encoding", "UTF-8")
            .option("nullValue", "")
            .csv(path)
        )
    
        # FIX 2: Drop any auto-named columns from trailing commas (e.g. _c42)
        phantom_cols = [c for c in sdf.columns if c.startswith("_c")]
        if phantom_cols:
            logger.warning(f"{filename}: Dropping phantom columns {phantom_cols} (trailing comma in CSV header)")
            sdf = sdf.drop(*phantom_cols)
    
        # FIX 1: Cast all label columns to IntegerType
        label_cols = ['completeness', 'consistency', 'validity', 'accuracy', 'final_label']
        for col in label_cols:
            if col in sdf.columns:
                sdf = sdf.withColumn(col, F.col(col).cast(IntegerType()))
    
        logger.info(f"Loaded Spark DF: {filename} ({sdf.count()} rows, {len(sdf.columns)} cols)")
        return sdf



    # ------------------------------------------------------------------
    # Spark-native validation (no pandas)
    # ------------------------------------------------------------------

    def _validate_spark_df(self, sdf: SparkDataFrame, name: str) -> None:
        """
        Run data quality checks using Spark DataFrame operations.
        All operations run as distributed Spark jobs.
        """
        row_count = sdf.count()
        print(f"\n[SPARK VALIDATION] {name}")
        print(f"  Rows    : {row_count}")
        print(f"  Columns : {len(sdf.columns)}")

        # Null check on label columns — uses Spark filter
        label_cols = ['completeness', 'consistency', 'validity', 'accuracy', 'final_label']
        for col in label_cols:
            if col in sdf.columns:
                null_count = sdf.filter(F.col(col).isNull()).count()
                status = "✅ no nulls" if null_count == 0 else f"⚠️  {null_count} nulls"
                print(f"  {col:<20}: {status}")

        # Label distribution — uses Spark groupBy aggregation
        if 'final_label' in sdf.columns:
            dist = (
                sdf.groupBy('final_label')
                .count()
                .orderBy('final_label')
                .collect()
            )
            print(f"  Label distribution:")
            for row in dist:
                label_str = "HIGH" if row['final_label'] == 1 else "LOW"
                pct = 100 * row['count'] / row_count
                print(f"    {label_str} ({row['final_label']}): {row['count']} rows ({pct:.1f}%)")

    # ------------------------------------------------------------------
    # Public loaders (return both Spark DF + pandas bridge)
    # ------------------------------------------------------------------

    def load_titanic(self) -> Tuple[SparkDataFrame, pd.DataFrame]:
        """Load Titanic ground-truth. Returns (SparkDataFrame, pandas DataFrame)."""
        sdf = self._load_spark_df("titanic_ground_truth.csv")
        self._validate_spark_df(sdf, "Titanic")
        return sdf, sdf.toPandas()

    def load_ecommerce(self) -> Tuple[SparkDataFrame, pd.DataFrame]:
        """Load E-Commerce ground-truth. Returns (SparkDataFrame, pandas DataFrame)."""
        sdf = self._load_spark_df("brazilian_ecommerce_ground_truth.csv")
        self._validate_spark_df(sdf, "E-Commerce")
        return sdf, sdf.toPandas()

    def load_hr(self) -> Tuple[SparkDataFrame, pd.DataFrame]:
        """Load HR Analytics ground-truth. Returns (SparkDataFrame, pandas DataFrame)."""
        sdf = self._load_spark_df("hr_ground_truth.csv")
        self._validate_spark_df(sdf, "HR Analytics")
        return sdf, sdf.toPandas()

    def load_all(self) -> Tuple[SparkDataFrame, pd.DataFrame]:
        """
        Load and UNION all three datasets using Spark.

        Spark UNION is used for distributed merging — at scale this avoids
        pulling all data to a single node (unlike pandas pd.concat).

        Returns:
            (combined_spark_df, combined_pandas_df)
        """
        print("\n" + "=" * 60)
        print("SPARK DATA LOADER: LOADING ALL DATASETS")
        print("=" * 60)

        titanic_sdf,   titanic_pd   = self.load_titanic()
        ecommerce_sdf, ecommerce_pd = self.load_ecommerce()
        hr_sdf,        hr_pd        = self.load_hr()

        # Select shared label columns for UNION
        label_cols = ['completeness', 'consistency', 'validity', 'accuracy', 'final_label']

        def select_labels(sdf):
            available = [c for c in label_cols if c in sdf.columns]
            return sdf.select([F.col(c) for c in available])

        combined_sdf = (
            select_labels(titanic_sdf)
            .union(select_labels(ecommerce_sdf))
            .union(select_labels(hr_sdf))
        )
        combined_pd = pd.concat([titanic_pd, ecommerce_pd, hr_pd], ignore_index=True)

        total = combined_sdf.count()
        print(f"\n✅ Spark UNION complete : {total} rows")
        print(f"✅ Pandas bridge ready  : {len(combined_pd)} rows for sklearn")
        print("=" * 60 + "\n")

        return combined_sdf, combined_pd

    # ------------------------------------------------------------------
    # Spark-native quality statistics (no pandas)
    # ------------------------------------------------------------------

    def compute_spark_quality_stats(self, sdf: SparkDataFrame, name: str = "Dataset") -> dict:
        """
        Compute dataset-level quality statistics using pure Spark aggregations.
        Used by the dashboard summary panel.
        """
        label_cols = ['completeness', 'consistency', 'validity', 'accuracy']
        available  = [c for c in label_cols if c in sdf.columns]
        total_rows = sdf.count()

        stats = {"dataset": name, "total_rows": total_rows}

        if available:
            agg_exprs = []
            for col in available:
                agg_exprs.append(F.mean(F.col(col).cast("float")).alias(f"{col}_mean"))
                agg_exprs.append(F.sum(F.col(col).cast("int")).alias(f"{col}_pass"))
            result = sdf.agg(*agg_exprs).collect()[0]
            for col in available:
                stats[f"{col}_pass_rate"] = float(result[f"{col}_mean"] or 0.0)
                stats[f"{col}_pass_count"] = int(result[f"{col}_pass"] or 0)

        if 'final_label' in sdf.columns:
            for row in sdf.groupBy('final_label').count().collect():
                key = 'high_quality_count' if row['final_label'] == 1 else 'low_quality_count'
                stats[key] = row['count']

        return stats

    def stop(self):
        """Stop the SparkSession. Call once all processing is complete."""
        self.spark.stop()
        print("✅ SparkSession stopped.")


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    loader = SparkDataLoader()
    combined_sdf, combined_pd = loader.load_all()

    stats = loader.compute_spark_quality_stats(combined_sdf, "Combined")
    print("\nSpark Quality Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print(f"\n✅ Ready for sklearn: {len(combined_pd)} rows")
    loader.stop()
