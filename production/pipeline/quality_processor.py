# phase6_production/pipeline/quality_processor.py
# Quality Score Processing & Reporting Engine

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class QualityProcessor:
    """
    Process quality features and generate comprehensive quality reports.

    Handles:
    - Aggregating 11 individual quality scores into a weighted composite
    - Classifying records as HIGH (≥0.7) or LOW (<0.7) quality
    - Generating quality reports with issue identification
    - Providing remediation recommendations
    - Computing quality trends by group

    Feature Weights (sum = 1.0):
        completeness_score:       0.20  (most critical — missing data)
        consistency_score:        0.15  (type uniformity)
        validity_score:           0.15  (format correctness)
        uniqueness_score:         0.10  (duplicate detection)
        accuracy_score:           0.10  (column completeness proxy)
        conformity_score:         0.10  (schema conformance)
        integrity_score:          0.10  (referential integrity)
        timeliness_score:         0.05  (data currency)
        schema_match_score:       0.03  (structure validation)
        format_compliance_score:  0.02  (format validation)
        Total:                    1.00
    """

    def __init__(self):
        """Initialise quality processor with feature weights."""
        logger.info("Initialising QualityProcessor")

        # Weighted scoring scheme — weights sum to 1.0
        self.feature_weights = {
            'completeness_score':      0.20,
            'consistency_score':       0.15,
            'validity_score':          0.15,
            'uniqueness_score':        0.10,
            'accuracy_score':          0.10,
            'conformity_score':        0.10,
            'integrity_score':         0.10,
            'timeliness_score':        0.05,
            'schema_match_score':      0.03,
            'format_compliance_score': 0.02,
        }
        assert abs(sum(self.feature_weights.values()) - 1.0) < 1e-9,             "Feature weights must sum to 1.0"

    def compute_quality_score(self, row: pd.Series) -> float:
        """
        Compute weighted quality score for a single row.

        Score = sum(feature_weight * feature_value) / sum(applicable_weights)
        Uses only features present in the row (gracefully handles missing features).
        Returns: float in [0, 1]
        """
        score = 0.0
        total_weight = 0.0
        for feature, weight in self.feature_weights.items():
            if feature in row.index and not pd.isna(row[feature]):
                score += row[feature] * weight
                total_weight += weight
        return score / total_weight if total_weight > 0 else 0.0

    def compute_quality_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute overall quality scores for all rows in the dataset.

        Returns:
            DataFrame with original data + 'quality_score' column (0–1 scale).
        """
        result = df.copy()
        logger.info(f"Computing quality scores for {len(df)} rows")
        result['quality_score'] = result.apply(self.compute_quality_score, axis=1)
        logger.info(
            f"Quality scores: mean={result['quality_score'].mean():.3f}, "
            f"std={result['quality_score'].std():.3f}"
        )
        return result

    def classify_quality(self, quality_score: float, threshold: float = 0.7) -> str:
        """
        Classify a single record as HIGH or LOW quality.

        HIGH: quality_score >= threshold (default 0.7)
        LOW:  quality_score <  threshold
        """
        return "HIGH" if quality_score >= threshold else "LOW"

    def classify_records(self, df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
        """
        Classify all records as HIGH or LOW quality based on quality_score.

        Args:
            df:        DataFrame with 'quality_score' column
            threshold: Boundary between HIGH and LOW (default 0.7)
        Returns:
            DataFrame with added 'quality_class' column ('HIGH'/'LOW')
        """
        result = df.copy()
        result['quality_class'] = result['quality_score'].apply(
            lambda x: self.classify_quality(x, threshold)
        )
        high_count = (result['quality_class'] == 'HIGH').sum()
        low_count  = (result['quality_class'] == 'LOW').sum()
        logger.info(
            f"Classification: {high_count} HIGH ({100*high_count/len(result):.1f}%), "
            f"{low_count} LOW ({100*low_count/len(result):.1f}%)"
        )
        return result

    def compute_dataset_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Compute comprehensive dataset-level summary metrics.
        """
        quality_scores = df['quality_score'] if 'quality_score' in df.columns else pd.Series([0.0])
        metrics = {
            'total_records':   len(df),
            'mean_quality':    float(quality_scores.mean()),
            'median_quality':  float(quality_scores.median()),
            'std_quality':     float(quality_scores.std()),
            'min_quality':     float(quality_scores.min()),
            'max_quality':     float(quality_scores.max()),
            'q25_quality':     float(quality_scores.quantile(0.25)),
            'q75_quality':     float(quality_scores.quantile(0.75)),
        }
        if 'quality_class' in df.columns:
            metrics['high_quality_count'] = int((df['quality_class'] == 'HIGH').sum())
            metrics['low_quality_count']  = int((df['quality_class'] == 'LOW').sum())
            metrics['high_quality_pct']   = 100 * metrics['high_quality_count'] / len(df)
            metrics['low_quality_pct']    = 100 * metrics['low_quality_count']  / len(df)
        return metrics

    def identify_issues(self, df: pd.DataFrame) -> Dict[str, dict]:
        """
        Identify common quality issues by scanning for low feature scores.
        Threshold for flagging: score < 0.5 (below acceptable quality).
        """
        issues = {
            'completeness': {},
            'consistency':  {},
            'uniqueness':   {},
            'validity':     {},
            'accuracy':     {},
            'integrity':    {},
        }
        issue_threshold = 0.5
        feature_issue_map = {
            'completeness_score': 'completeness',
            'consistency_score':  'consistency',
            'uniqueness_score':   'uniqueness',
            'validity_score':     'validity',
            'accuracy_score':     'accuracy',
            'integrity_score':    'integrity',
        }
        for feature, issue_type in feature_issue_map.items():
            if feature in df.columns:
                low_scoring_rows = df[df[feature] < issue_threshold].index.tolist()
                if len(low_scoring_rows) > 0:
                    issues[issue_type] = {
                        'affected_rows': len(low_scoring_rows),
                        'percentage':    100 * len(low_scoring_rows) / len(df),
                        'sample_indices': low_scoring_rows[:5],
                    }
        return issues

    def generate_recommendations(self, issues: Dict) -> List[str]:
        """
        Generate actionable improvement recommendations based on identified issues.
        """
        recommendations = []
        issue_messages = {
            'completeness': lambda pct: (
                f"Missing Data: {pct:.1f}% of records have missing values. "
                f"Review imputation strategy or upstream data collection process."
            ),
            'consistency': lambda pct: (
                f"Type Inconsistency: {pct:.1f}% of records have mixed data types. "
                f"Implement schema validation and type enforcement at ingestion."
            ),
            'uniqueness': lambda pct: (
                f"Duplicates: {pct:.1f}% of records are duplicated. "
                f"Run deduplication pipeline before model training."
            ),
            'validity': lambda pct: (
                f"Invalid Format: {pct:.1f}% of records contain invalid values. "
                f"Apply format validation rules and data cleansing transformations."
            ),
            'accuracy': lambda pct: (
                f"Accuracy Issues: {pct:.1f}% of records have accuracy concerns. "
                f"Verify against source systems and apply reconciliation checks."
            ),
            'integrity': lambda pct: (
                f"Referential Integrity: {pct:.1f}% of records have integrity problems. "
                f"Enforce foreign key constraints and schema conformance rules."
            ),
        }
        for issue_type, message_fn in issue_messages.items():
            issue_detail = issues.get(issue_type, {})
            if isinstance(issue_detail, dict) and 'percentage' in issue_detail:
                recommendations.append(message_fn(issue_detail['percentage']))
        return recommendations

    def generate_quality_report(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict:
        """
        Generate a comprehensive quality report for a dataset.

        Returns:
            dict with keys: dataset_name, timestamp, metrics, issues,
                            recommendations, overall_percentage, status,
                            quality_distribution
        """
        if 'quality_score' not in df.columns:
            df = self.compute_quality_scores(df)
        if 'quality_class' not in df.columns:
            df = self.classify_records(df)

        metrics         = self.compute_dataset_metrics(df)
        issues          = self.identify_issues(df)
        recommendations = self.generate_recommendations(issues)

        mean_q = metrics['mean_quality']
        status = (
            'EXCELLENT' if mean_q >= 0.9 else
            'GOOD'      if mean_q >= 0.8 else
            'FAIR'      if mean_q >= 0.7 else
            'POOR'
        )
        report = {
            'dataset_name':         dataset_name,
            'timestamp':            datetime.now().isoformat(),
            'metrics':              metrics,
            'issues':               issues,
            'recommendations':      recommendations,
            'overall_percentage':   mean_q * 100,
            'status':               status,
            'quality_distribution': {
                'high': metrics.get('high_quality_pct', 0),
                'low':  metrics.get('low_quality_pct',  0),
            },
        }
        return report

    def print_report(self, report: Dict) -> None:
        """Print a formatted quality report to console."""
        print("\n" + "="*80)
        print(f"QUALITY REPORT: {report['dataset_name']}")
        print("="*80)

        metrics = report['metrics']
        print(f"\nDataset Size: {metrics['total_records']} records")
        print(f"\nQuality Score Statistics:")
        print(f"  Mean:   {metrics['mean_quality']:.3f}  ({metrics['mean_quality']*100:.1f}%)")
        print(f"  Median: {metrics['median_quality']:.3f}")
        print(f"  Std:    {metrics['std_quality']:.3f}")
        print(f"  Min:    {metrics['min_quality']:.3f}")
        print(f"  Max:    {metrics['max_quality']:.3f}")
        print(f"  Q25:    {metrics['q25_quality']:.3f}")
        print(f"  Q75:    {metrics['q75_quality']:.3f}")

        if 'high_quality_count' in metrics:
            print(f"\nQuality Classification (threshold=0.7):")
            print(f"  HIGH: {metrics['high_quality_count']:>5} records  ({metrics['high_quality_pct']:>5.1f}%)")
            print(f"  LOW:  {metrics['low_quality_count']:>5} records  ({metrics['low_quality_pct']:>5.1f}%)")

        print(f"\nOverall Status: {report['status']} ({report['overall_percentage']:.1f}%)")

        issues = report['issues']
        has_issues = any(isinstance(v, dict) and 'percentage' in v for v in issues.values())
        if has_issues:
            print(f"\nIdentified Issues:")
            for issue_type, details in issues.items():
                if isinstance(details, dict) and 'percentage' in details:
                    print(f"  • {issue_type.upper()}: {details['percentage']:.1f}% affected")
        else:
            print(f"\n  No significant quality issues detected.")

        if report['recommendations']:
            print(f"\nRecommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")

        print("="*80 + "\n")

    def export_report_json(self, report: Dict, filepath: str) -> None:
        """Export quality report as JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report exported to {filepath}")

    def export_scores_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Export quality scores to CSV file."""
        score_cols = [col for col in df.columns if col.endswith('_score')]
        if 'quality_class' in df.columns:
            score_cols.append('quality_class')
        df[score_cols].to_csv(filepath, index=True)
        logger.info(f"Scores exported to {filepath}")

    def compute_quality_trend(self, df: pd.DataFrame, group_by: str = None) -> Dict:
        """
        Compute quality metrics by group (optional).
        If group_by is None or not found, returns overall metrics.
        """
        if group_by is None or group_by not in df.columns:
            return {'overall': self.compute_dataset_metrics(df)}
        trends = {}
        for group_name, group_df in df.groupby(group_by):
            trends[str(group_name)] = self.compute_dataset_metrics(group_df)
        return trends

    def extract_labels(self, df: pd.DataFrame, source: str = 'final_label') -> pd.Series:
        """
        Extract binary quality labels from a ground-truth dataset.

        Args:
            df:     DataFrame containing ground-truth labels
            source: Column name with labels ('final_label', 'quality_label', etc.)
        Returns:
            pd.Series of int labels (1 = HIGH quality, 0 = LOW quality)
        Raises:
            ValueError if source column is not found
        """
        if source not in df.columns:
            available = [c for c in df.columns if 'label' in c.lower() or 'quality' in c.lower()]
            raise ValueError(
                f"Column '{source}' not found. Available label-related columns: {available}"
            )
        labels = df[source].copy()
        if labels.dtype in ['int64', 'int32', 'float64', int, float]:
            return labels.astype(int)
        # Handle string labels ('HIGH'/'LOW', '1'/'0', etc.)
        labels_str = labels.astype(str).str.strip().str.upper()
        return (labels_str == 'HIGH').astype(int)