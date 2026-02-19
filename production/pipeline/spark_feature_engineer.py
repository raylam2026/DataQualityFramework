# phase6_production/pipeline/spark_feature_engineer.py
# PySpark-based quality feature extraction.
# Computes 11 quality dimensions using Spark DataFrame operations.
# Replaces pandas row-by-row iteration with distributed Spark jobs.

from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from spark_data_loader import get_spark_session
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SparkFeatureEngineer:
    """
    Distributed quality feature extraction using PySpark.

    Replaces QualityFeatureEngineer (pandas) with fully Spark-native operations.
    Key difference: no df.iterrows() — all computations parallelised across partitions.

    Feature computation approach:
    ┌─────────────────────────┬──────────────────────────────────────────┐
    │ Feature                 │ Spark Implementation                     │
    ├─────────────────────────┼──────────────────────────────────────────┤
    │ completeness_score      │ F.when(isNotNull, 1) per row             │
    │ consistency_score       │ Cast-based type validation per column    │
    │ uniqueness_score        │ groupBy().count() + window join          │
    │ validity_score          │ Aggregated non-empty ratio               │
    │ accuracy_score          │ Mean column completeness (broadcast)     │
    │ conformity_score        │ approx_count_distinct / total            │
    │ timeliness_score        │ max(datetime) → days delta               │
    │ integrity_score         │ Schema StringType ratio check            │
    │ schema_match_score      │ Mirrors integrity_score                  │
    │ format_compliance_score │ Mirrors validity_score                   │
    │ outlier_score           │ percentile_approx IQR detection          │
    └─────────────────────────┴──────────────────────────────────────────┘
    """

    def __init__(self, spark: SparkSession = None):
        self.spark = spark or get_spark_session()

    # ------------------------------------------------------------------
    # ROW-LEVEL: computed per row in parallel
    # ------------------------------------------------------------------

    def compute_completeness(self, sdf: SparkDataFrame) -> SparkDataFrame:
        """
        Row completeness = non-null columns / total columns.
        Spark equivalent of: df.notna().sum(axis=1) / len(df.columns)
        """
        n_cols = len(sdf.columns)
        non_null_expr = sum(
            F.when(F.col(c).isNotNull(), 1).otherwise(0)
            for c in sdf.columns
        )
        return sdf.withColumn(
            "completeness_score",
            (non_null_expr / n_cols).cast(FloatType())
        )

    def compute_consistency(self, sdf: SparkDataFrame) -> SparkDataFrame:
        """
        Row consistency = proportion of columns where value matches inferred type.
        Uses TRY_CAST logic: if a column is numeric, check it can cast to double.
        """
        n_cols = len(sdf.columns)
        consistent_expr = F.lit(0)

        for field in sdf.schema.fields:
            c = field.name
            dtype = str(field.dataType)
            if "IntegerType" in dtype or "LongType" in dtype or "DoubleType" in dtype or "FloatType" in dtype:
                # Numeric column: consistent if not null
                consistent_expr = consistent_expr + F.when(
                    F.col(c).isNotNull() & F.col(c).cast("double").isNotNull(),
                    1
                ).otherwise(0)
            else:
                # String column: always consistent (treat as type-neutral)
                consistent_expr = consistent_expr + F.when(F.col(c).isNotNull(), 1).otherwise(1)

        return sdf.withColumn(
            "consistency_score",
            (consistent_expr / n_cols).cast(FloatType())
        )

    def compute_uniqueness(self, sdf: SparkDataFrame) -> SparkDataFrame:
        """
        Row uniqueness = 1.0 if row is unique, 0.0 if exact duplicate.
        Uses Spark groupBy + join (distributed equivalent of df.duplicated()).
        """
        cols = sdf.columns

        # Count occurrences of each distinct row combination
        counts_df = (
            sdf.groupBy(cols)
            .count()
            .withColumnRenamed("count", "_row_count")
        )
        result = sdf.join(counts_df, on=cols, how="left")
        return result.withColumn(
            "uniqueness_score",
            F.when(F.col("_row_count") == 1, 1.0).otherwise(0.0).cast(FloatType())
        ).drop("_row_count")

    # ------------------------------------------------------------------
    # DATASET-LEVEL: computed once, broadcast as constant to all rows
    # ------------------------------------------------------------------

    def compute_validity(self, sdf: SparkDataFrame) -> SparkDataFrame:
        """
        Validity = mean(non-empty ratio per column), broadcast to all rows.
        """
        total = sdf.count()
        if total == 0:
            return sdf.withColumn("validity_score", F.lit(0.0).cast(FloatType()))

        col_validity = []
        for c in sdf.columns:
            valid = sdf.filter(
                F.col(c).isNotNull() & (F.trim(F.col(c).cast("string")) != "")
            ).count()
            col_validity.append(valid / total)

        mean_validity = float(np.mean(col_validity)) if col_validity else 0.0
        return sdf.withColumn("validity_score", F.lit(mean_validity).cast(FloatType()))

    def compute_accuracy(self, sdf: SparkDataFrame) -> SparkDataFrame:
        """
        Accuracy (column completeness proxy) = mean(non-null ratio per column).
        """
        total = sdf.count()
        if total == 0:
            return sdf.withColumn("accuracy_score", F.lit(0.0).cast(FloatType()))

        col_completeness = [
            sdf.filter(F.col(c).isNotNull()).count() / total
            for c in sdf.columns
        ]
        mean_acc = float(np.mean(col_completeness)) if col_completeness else 0.0
        return sdf.withColumn("accuracy_score", F.lit(mean_acc).cast(FloatType()))

    def compute_conformity(self, sdf: SparkDataFrame) -> SparkDataFrame:
        """
        Conformity = approx_count_distinct / total.
        Uses Spark's approx_count_distinct (HyperLogLog) — efficient at scale.
        """
        total = sdf.count()
        if total == 0:
            return sdf.withColumn("conformity_score", F.lit(0.0).cast(FloatType()))

        row_key = F.concat_ws("|", *[F.col(c).cast("string") for c in sdf.columns])
        distinct_approx = sdf.agg(
            F.approx_count_distinct(row_key).alias("distinct")
        ).collect()[0]["distinct"]

        conformity = min(1.0, float(distinct_approx) / total)
        return sdf.withColumn("conformity_score", F.lit(conformity).cast(FloatType()))

    def compute_timeliness(self, sdf: SparkDataFrame) -> SparkDataFrame:
        """
        Timeliness = recency of most recent datetime column.
        Returns 1.0 for static datasets with no timestamp columns.
        """
        date_cols = [
            f.name for f in sdf.schema.fields
            if "TimestampType" in str(f.dataType) or "DateType" in str(f.dataType)
        ]

        if not date_cols:
            timeliness = 1.0
        else:
            latest = sdf.agg(F.max(F.col(date_cols[0]))).collect()[0][0]
            if latest is None:
                timeliness = 1.0
            else:
                days_old = (datetime.now() - latest).days if hasattr(latest, 'year') else 0
                timeliness = max(0.0, 1.0 - days_old / 365.0)

        return sdf.withColumn("timeliness_score", F.lit(timeliness).cast(FloatType()))

    def compute_integrity(self, sdf: SparkDataFrame) -> SparkDataFrame:
        """
        Integrity = schema consistency (penalises datasets where >50% of
        columns are StringType, suggesting un-typed/raw ingestion).
        """
        string_cols = sum(1 for f in sdf.schema.fields if "StringType" in str(f.dataType))
        integrity = 0.7 if string_cols > len(sdf.columns) / 2 else 1.0
        return sdf.withColumn("integrity_score", F.lit(integrity).cast(FloatType()))

    def compute_schema_match(self, sdf: SparkDataFrame) -> SparkDataFrame:
        """Schema match mirrors integrity score."""
        val = float(sdf.select("integrity_score").first()[0]) if "integrity_score" in sdf.columns else 1.0
        return sdf.withColumn("schema_match_score", F.lit(val).cast(FloatType()))

    def compute_format_compliance(self, sdf: SparkDataFrame) -> SparkDataFrame:
        """Format compliance mirrors validity score."""
        val = float(sdf.select("validity_score").first()[0]) if "validity_score" in sdf.columns else 1.0
        return sdf.withColumn("format_compliance_score", F.lit(val).cast(FloatType()))

    def compute_outlier_score(self, sdf: SparkDataFrame) -> SparkDataFrame:
        """
        IQR-based outlier detection using Spark's percentile_approx.
        Spark equivalent of pandas quantile() — runs distributed.
        """
        numeric_cols = [
            f.name for f in sdf.schema.fields
            if any(t in str(f.dataType) for t in ("IntegerType", "LongType", "FloatType", "DoubleType"))
            and not f.name.endswith("_score")
        ]

        if not numeric_cols:
            return sdf.withColumn("outlier_score", F.lit(1.0).cast(FloatType()))

        total_cells = sdf.count() * len(numeric_cols)
        outlier_count = 0

        for col in numeric_cols:
            q1, q3 = sdf.agg(
                F.percentile_approx(F.col(col).cast("double"), 0.25),
                F.percentile_approx(F.col(col).cast("double"), 0.75)
            ).collect()[0]

            if q1 is not None and q3 is not None:
                iqr   = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_count += sdf.filter(
                    (F.col(col).cast("double") < lower) |
                    (F.col(col).cast("double") > upper)
                ).count()

        outlier_score = max(0.0, 1.0 - outlier_count / total_cells) if total_cells > 0 else 1.0
        return sdf.withColumn("outlier_score", F.lit(outlier_score).cast(FloatType()))

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def extract_all_features(self, sdf: SparkDataFrame) -> SparkDataFrame:
        """
        Full distributed feature extraction pipeline.
        Applies all 11 quality dimensions sequentially via Spark lazy evaluation.
        Spark builds a DAG and optimises execution automatically.

        Returns:
            SparkDataFrame with original columns + 11 _score columns appended.
        """
        row_count = sdf.count()
        print(f"[SPARK FEATURES] Processing {row_count} rows...")

        result = sdf
        result = self.compute_completeness(result)
        result = self.compute_consistency(result)
        result = self.compute_uniqueness(result)
        result = self.compute_validity(result)
        result = self.compute_accuracy(result)
        result = self.compute_conformity(result)
        result = self.compute_timeliness(result)
        result = self.compute_integrity(result)
        result = self.compute_schema_match(result)
        result = self.compute_format_compliance(result)
        result = self.compute_outlier_score(result)

        n_features = sum(1 for f in result.schema.fields if f.name.endswith("_score"))
        print(f"[SPARK FEATURES] ✅ {n_features} feature columns extracted")
        return result

    def to_pandas(self, sdf: SparkDataFrame) -> pd.DataFrame:
        """
        Convert enriched SparkDataFrame to pandas for sklearn.
        Uses Arrow-optimised conversion (zero-copy when possible).
        """
        return sdf.toPandas()
