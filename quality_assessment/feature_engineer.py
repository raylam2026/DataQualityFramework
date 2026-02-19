"""
Feature Engineering Engine — Spec Design Report Aligned (v2)
=============================================================
13 features across 3 levels:

ROW-Level (4):     null_count, null_ratio, data_type_mismatch, outlier_flags
COLUMN-Level (4):  column_completeness, column_consistency, column_data_type_validity, column_duplicate_ratio
DATASET-Level (3): schema_inference_confidence, referential_integrity_score, overall_density
TEMPORAL (2):      max_timestamp_gap_hours, timestamp_order_violations   ← NEW

Usage:
    fe = FeatureEngineer()
    X = fe.fit_transform(ground_truth_df)   # fit + extract 13 features
    y = ground_truth_df['final_label'].values
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict

ANNOTATION_COLS = {'completeness', 'consistency', 'validity', 'accuracy', 'final_label', 'notes'}

FEATURE_NAMES = [
    # ROW-Level (4)
    'null_count', 'null_ratio', 'data_type_mismatch', 'outlier_flags',
    # COLUMN-Level (4)
    'column_completeness', 'column_consistency', 'column_data_type_validity', 'column_duplicate_ratio',
    # DATASET-Level (3)
    'schema_inference_confidence', 'referential_integrity_score', 'overall_density',
    # TEMPORAL (2) — NEW: captures time-gap quality issues
    'max_timestamp_gap_hours', 'timestamp_order_violations',
]


class FeatureEngineer:
    """Spec-aligned feature engineering across 3 levels + temporal (ROW / COLUMN / DATASET / TEMPORAL)."""

    def __init__(self):
        self.inferred_types: Dict[str, str] = {}
        self.iqr_bounds: Dict[str, tuple] = {}
        self.sorted_ts_cols: List[str] = []         # NEW: chronologically sorted timestamp columns
        self.fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        raw_cols = self._raw_cols(df)
        raw_df = df[raw_cols]

        for col in raw_cols:
            series = raw_df[col]
            non_null = series.dropna()
            if len(non_null) == 0:
                self.inferred_types[col] = 'string'
                continue
            if self._is_date_column(non_null):
                self.inferred_types[col] = 'date'
                continue
            numeric_parsed = pd.to_numeric(non_null, errors='coerce')
            if numeric_parsed.notna().sum() / len(non_null) > 0.7:
                self.inferred_types[col] = 'numeric'
                valid = numeric_parsed.dropna()
                if len(valid) >= 4:
                    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
                    iqr = q3 - q1
                    self.iqr_bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            else:
                self.inferred_types[col] = 'string'

        # NEW: Identify and sort timestamp columns by median for temporal features
        ts_cols = [c for c in raw_cols if self.inferred_types.get(c) == 'date']
        if len(ts_cols) >= 2:
            medians = {}
            for col in ts_cols:
                parsed = pd.to_datetime(raw_df[col], errors='coerce', format='mixed', dayfirst=False).dropna()
                if len(parsed) > 0:
                    medians[col] = parsed.median()
            self.sorted_ts_cols = sorted(medians.keys(), key=lambda c: medians[c])
        else:
            self.sorted_ts_cols = ts_cols

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self.fitted, "Call fit() before transform()"
        raw_cols = self._raw_cols(df)
        raw_df = df[raw_cols].copy()
        n, m = len(raw_df), len(raw_cols)
        F = np.zeros((n, 13))  # ← Changed from 11 to 13

        # ═══════════ ROW-Level (4) ═══════════
        F[:, 0] = raw_df.isnull().sum(axis=1).values
        F[:, 1] = raw_df.isnull().mean(axis=1).values

        mismatch = np.zeros(n)
        for col in raw_cols:
            itype = self.inferred_types.get(col, 'string')
            not_null = raw_df[col].notna()
            if itype == 'numeric':
                parsed = pd.to_numeric(raw_df[col], errors='coerce')
                mismatch += (not_null & parsed.isna()).astype(int).values
            elif itype == 'date':
                parsed = pd.to_datetime(raw_df[col], errors='coerce', format='mixed', dayfirst=False)
                mismatch += (not_null & parsed.isna()).astype(int).values
        F[:, 2] = mismatch

        outliers = np.zeros(n)
        for col, bounds in self.iqr_bounds.items():
            if col in raw_df.columns:
                vals = pd.to_numeric(raw_df[col], errors='coerce')
                is_out = ((vals < bounds[0]) | (vals > bounds[1])) & vals.notna()
                outliers += is_out.astype(int).values
        F[:, 3] = outliers

        # ═══════════ COLUMN-Level (4) ═══════════
        comp_scores, cons_scores, val_scores = [], [], []
        for col in raw_cols:
            series = raw_df[col]
            non_null = series.dropna()
            total = len(series)
            comp_scores.append(len(non_null) / total if total > 0 else 0.0)
            if len(non_null) > 0:
                str_vals = non_null.astype(str)
                lengths = str_vals.str.len()
                cv = lengths.std() / (lengths.mean() + 1e-10)
                cons_scores.append(1.0 / (1.0 + cv))
            else:
                cons_scores.append(1.0)
            itype = self.inferred_types.get(col, 'string')
            if len(non_null) > 0:
                if itype == 'numeric':
                    valid = pd.to_numeric(non_null, errors='coerce').notna().sum()
                    val_scores.append(valid / len(non_null))
                elif itype == 'date':
                    valid = pd.to_datetime(non_null, errors='coerce', format='mixed', dayfirst=False).notna().sum()
                    val_scores.append(valid / len(non_null))
                else:
                    val_scores.append(1.0)
            else:
                val_scores.append(1.0)

        F[:, 4] = np.mean(comp_scores)
        F[:, 5] = np.mean(cons_scores)
        F[:, 6] = np.mean(val_scores)

        dup_score = np.zeros(n)
        for col in raw_cols:
            is_dup = raw_df[col].duplicated(keep=False) & raw_df[col].notna()
            dup_score += is_dup.astype(float).values
        F[:, 7] = dup_score / m

        # ═══════════ DATASET-Level (3) ═══════════
        schema_ok = np.ones(n, dtype=float)
        for col in raw_cols:
            itype = self.inferred_types.get(col, 'string')
            not_null = raw_df[col].notna()
            if itype == 'numeric':
                parsed = pd.to_numeric(raw_df[col], errors='coerce')
                schema_ok *= (~(not_null & parsed.isna())).astype(float).values
            elif itype == 'date':
                parsed = pd.to_datetime(raw_df[col], errors='coerce', format='mixed', dayfirst=False)
                schema_ok *= (~(not_null & parsed.isna())).astype(float).values
        F[:, 8] = schema_ok

        id_cols = [c for c in raw_cols if 'id' in c.lower() or c.lower().endswith('id')]
        if id_cols:
            integrity = np.ones(n, dtype=float)
            for col in id_cols:
                is_dup = raw_df[col].duplicated(keep=False) & raw_df[col].notna()
                integrity *= (~is_dup).astype(float).values
            F[:, 9] = integrity
        else:
            F[:, 9] = 1.0

        total_cells = n * m
        non_null_cells = int(raw_df.notna().sum().sum())
        F[:, 10] = non_null_cells / total_cells if total_cells > 0 else 0.0

        # ═══════════ TEMPORAL (2) — NEW ═══════════
        # Auto-detects date columns sorted by median, computes time gaps
        # between consecutive pairs. Captures "too long approval time" and
        # chronological order violations generically.
        ts_cols = [c for c in self.sorted_ts_cols if c in raw_df.columns]

        if len(ts_cols) >= 2:
            parsed_dates = {}
            for col in ts_cols:
                parsed_dates[col] = pd.to_datetime(raw_df[col], errors='coerce', format='mixed', dayfirst=False)

            max_gaps = np.zeros(n)
            violations = np.zeros(n)

            for i in range(len(ts_cols) - 1):
                col_a = ts_cols[i]
                col_b = ts_cols[i + 1]
                both_valid = parsed_dates[col_a].notna() & parsed_dates[col_b].notna()

                # Time difference in hours (col_b - col_a)
                diff_hours = (parsed_dates[col_b] - parsed_dates[col_a]).dt.total_seconds() / 3600.0

                # Feature 11: max absolute gap across all consecutive date pairs
                gap_abs = diff_hours.abs().fillna(0).values
                max_gaps = np.maximum(max_gaps, gap_abs)

                # Feature 12: chronological order violation (col_b < col_a)
                is_violation = (diff_hours < 0) & both_valid
                violations += is_violation.astype(float).values

            F[:, 11] = max_gaps
            F[:, 12] = violations
        else:
            F[:, 11] = 0.0
            F[:, 12] = 0.0

        return F

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    @staticmethod
    def _raw_cols(df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c.strip().lower() not in ANNOTATION_COLS]

    @staticmethod
    def _is_date_column(series: pd.Series, sample_size: int = 50) -> bool:
        sample = series.dropna().astype(str).head(sample_size)
        if len(sample) == 0:
            return False
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{4}-\d{2}-\d{2}',
            r'\d{1,2}-\d{1,2}-\d{2,4}',
        ]
        date_count = sum(
            1 for v in sample
            if any(re.search(p, str(v).strip()) for p in date_patterns)
        )
        return (date_count / len(sample)) > 0.5
