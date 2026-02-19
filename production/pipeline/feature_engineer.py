# phase6_production/pipeline/feature_engineer.py
# -*- coding: utf-8 -*-
# Quality Feature Extraction Engine

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class QualityFeatureEngineer:
    """
    Extract quality metrics from dataframes.

    Computes 11 quality dimensions across 3 levels:

    ROW-LEVEL (per-record):
    - completeness_score: % non-null values in the row
    - consistency_score: data type uniformity per row
    - uniqueness_score: 1.0 if row is unique, 0.0 if exact duplicate
    - outlier_score: 1.0 - (outlier_cells / numeric_cells) per row  [FIXED v2.1]

    COLUMN-LEVEL (aggregated to row via broadcast):
    - validity_score: mean column validity (valid format ratio)
    - accuracy_score: mean column completeness (used as accuracy proxy)

    DATASET-LEVEL (constant per dataset, broadcast to all rows):
    - conformity_score: dataset-level row uniqueness ratio
    - timeliness_score: data currency (datetime-based; 1.0 for static data)
    - integrity_score: schema consistency score
    - schema_match_score: duplicate of integrity check for schema structure
    - format_compliance_score: average format validity across all columns
    """

    def __init__(self):
        """Initialise feature engineer."""
        logger.info("Initialising QualityFeatureEngineer")

    # ==================== ROW-LEVEL METRICS ====================

    def compute_row_completeness(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute completeness for each row.
        Score = (non-null values) / (total columns)
        """
        non_null_counts = df.notna().sum(axis=1)
        total_cols = len(df.columns)
        return non_null_counts / total_cols

    def compute_row_type_consistency(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute type consistency for each row.
        Score = proportion of values matching their column expected dtype.
        """
        scores = []
        expected_dtypes = df.dtypes
        for idx, row in df.iterrows():
            matches = 0
            for col, expected_dtype in expected_dtypes.items():
                val = row[col]
                if pd.isna(val):
                    matches += 1  # NaN treated as type-neutral
                elif expected_dtype == 'object':
                    matches += 1
                elif expected_dtype in ['int64', 'int32']:
                    try:
                        int(val)
                        matches += 1
                    except (ValueError, TypeError):
                        pass
                elif expected_dtype in ['float64', 'float32']:
                    try:
                        float(val)
                        matches += 1
                    except (ValueError, TypeError):
                        pass
            score = matches / len(expected_dtypes)
            scores.append(score)
        return pd.Series(scores, index=df.index)

    def compute_row_uniqueness(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute uniqueness for each row.
        Score = 1.0 if row is unique, 0.0 if exact duplicate exists.
        """
        duplicates = df.duplicated(keep=False)
        return (~duplicates).astype(float)

    def compute_row_outlier_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute per-row outlier score using IQR method.

        For each row, counts how many of its numeric cell values fall outside
        the [Q1 - 1.5*IQR, Q3 + 1.5*IQR] bounds of that column.

        Score = 1.0 - (outlier_cell_count / numeric_column_count)
        A perfect row (no outliers) scores 1.0.
        A row where every numeric value is an outlier scores 0.0.

        Spec reference: "outlier_flags: Count of numeric values exceeding IQR bounds"
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) == 0 or len(df) == 0:
            return pd.Series(1.0, index=df.index)

        # Pre-compute IQR bounds per numeric column
        bounds = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            bounds[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

        # Vectorised: build a boolean DataFrame of outlier flags
        outlier_flags = pd.DataFrame(index=df.index)
        for col in numeric_cols:
            lower, upper = bounds[col]
            outlier_flags[col] = (df[col] < lower) | (df[col] > upper)

        outlier_count_per_row = outlier_flags.sum(axis=1)
        score = 1.0 - (outlier_count_per_row / len(numeric_cols))
        return score.clip(0.0, 1.0)

    # ==================== COLUMN-LEVEL METRICS ====================

    def compute_column_completeness(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute completeness for each column.
        Score = (non-null values) / (total rows)
        """
        return {col: df[col].notna().sum() / len(df) for col in df.columns}

    def compute_column_uniqueness(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute uniqueness for each column.
        Score = (unique values) / (total non-null values)
        """
        scores = {}
        for col in df.columns:
            non_null = df[col].notna().sum()
            if non_null > 0:
                unique = df[col].nunique()
                scores[col] = unique / non_null
            else:
                scores[col] = 0.0
        return scores

    def compute_column_validity(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute validity for each column.
        Score = (valid format values) / (total non-null values)
        Valid = non-empty strings, finite numeric values.
        """
        scores = {}
        for col in df.columns:
            non_null = df[col].notna().sum()
            if non_null == 0:
                scores[col] = 0.0
                continue
            valid_count = 0
            for val in df[col].dropna():
                if isinstance(val, str):
                    if len(str(val).strip()) > 0:
                        valid_count += 1
                elif isinstance(val, (int, float)):
                    if not np.isnan(float(val)) and not np.isinf(float(val)):
                        valid_count += 1
                else:
                    valid_count += 1
            scores[col] = valid_count / non_null
        return scores

    # ==================== DATASET-LEVEL METRICS ====================

    def compute_dataset_completeness(self, df: pd.DataFrame) -> float:
        """
        Compute overall dataset completeness.
        Score = (total non-null values) / (total cells)
        """
        total_cells = len(df) * len(df.columns)
        non_null_cells = df.notna().sum().sum()
        return non_null_cells / total_cells if total_cells > 0 else 0.0

    def compute_dataset_consistency(self, df: pd.DataFrame) -> float:
        """
        Compute overall dataset consistency.
        Score = mean type consistency across all rows.
        """
        consistency_scores = self.compute_row_type_consistency(df)
        return consistency_scores.mean()

    def compute_dataset_uniqueness(self, df: pd.DataFrame) -> float:
        """
        Compute overall dataset uniqueness.
        Score = (unique rows) / (total rows)
        """
        unique_rows = len(df) - df.duplicated().sum()
        return unique_rows / len(df) if len(df) > 0 else 0.0

    def compute_dataset_validity(self, df: pd.DataFrame) -> float:
        """
        Compute overall dataset validity.
        Score = mean validity across all columns.
        """
        col_validity = self.compute_column_validity(df)
        if len(col_validity) == 0:
            return 0.0
        return np.mean(list(col_validity.values()))

    def compute_schema_match(self, df: pd.DataFrame) -> float:
        """
        Compute schema match score.
        Score = 1.0 if all columns have consistent internal types, else 0.7.
        """
        for col in df.columns:
            unique_types = set()
            for val in df[col].dropna():
                unique_types.add(type(val).__name__)
            if len(unique_types) > 1:
                return 0.7
        return 1.0

    def compute_timeliness_score(self, df: pd.DataFrame) -> float:
        """
        Compute timeliness score based on data currency.

        Logic:
        - If datetime columns exist: score based on recency of latest date
          (1.0 = within 30 days, decreasing linearly to 0.0 at 365+ days)
        - If no datetime columns: returns 1.0 (static datasets treated as current)
        """
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if len(date_cols) == 0:
            logger.debug("No datetime columns found; timeliness_score set to 1.0 (static dataset)")
            return 1.0

        try:
            latest_date = df[date_cols[0]].dropna().max()
            if pd.isna(latest_date):
                return 1.0
            today = pd.Timestamp.now()
            days_old = (today - latest_date).days
            timeliness = max(0.0, 1.0 - (days_old / 365.0))
            logger.debug(f"Timeliness: {days_old} days old -> score {timeliness:.3f}")
            return float(timeliness)
        except Exception as e:
            logger.warning(f"Could not compute timeliness from datetime column: {e}")
            return 1.0

    # ==================== EXTRACTION PIPELINE ====================

    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all 11 quality features for a dataset.

        Feature levels:
        ROW-level  (4): completeness, consistency, uniqueness, outlier  [outlier FIXED v2.1]
        COLUMN-level(2): validity, accuracy (broadcast from column stats)
        DATASET-level(5): conformity, timeliness, integrity, schema_match,
                          format_compliance

        Returns:
            DataFrame with original columns + 11 quality metric columns.
        """
        result = df.copy()
        logger.info(f"Extracting features from {len(df)} rows, {len(df.columns)} columns")

        # --- ROW-LEVEL (per row) ---
        result['completeness_score'] = self.compute_row_completeness(df)
        result['consistency_score'] = self.compute_row_type_consistency(df)
        result['uniqueness_score'] = self.compute_row_uniqueness(df)
        result['outlier_score'] = self.compute_row_outlier_score(df)  # ✅ FIXED v2.1

        # --- COLUMN-LEVEL (broadcast column stats to each row) ---
        col_validity = self.compute_column_validity(df)
        col_completeness = self.compute_column_completeness(df)
        result['validity_score'] = np.mean(list(col_validity.values()))
        result['accuracy_score'] = np.mean(list(col_completeness.values()))

        # --- DATASET-LEVEL (constant per dataset, broadcast to all rows) ---
        result['conformity_score'] = self.compute_dataset_uniqueness(df)
        result['timeliness_score'] = self.compute_timeliness_score(df)
        result['integrity_score'] = self.compute_schema_match(df)
        result['schema_match_score'] = self.compute_schema_match(df)
        result['format_compliance_score'] = self.compute_dataset_validity(df)

        n_features = len([c for c in result.columns if c.endswith('_score')])
        logger.info(f"Extracted {n_features} feature columns")
        return result

    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for all quality feature columns.
        """
        feature_cols = [col for col in df.columns if col.endswith('_score')]
        summary = {}
        for col in feature_cols:
            summary[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median(),
            }
        return summary
