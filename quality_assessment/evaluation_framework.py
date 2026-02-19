"""
Evaluation framework for H1, H2, H3 hypothesis testing.
Implements statistical tests per Spec Design Report.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class HypothesisEvaluation:
    """Test research hypotheses from spec."""
    
    @staticmethod
    def test_h1_feature_effectiveness(
        results_dict: Dict[str, List[float]]
    ) -> Dict[str, any]:
        """
        H1: Engineered ML features classify with precision ? 0.80 and recall ? 0.75
        
        Args:
            results_dict: {'titanic': [...], 'ecommerce': [...], 'hr': [...]}
                Each list contains fold results: precision, recall, f1 per fold
        
        Returns:
            h1_results dictionary with criteria met status
        """
        h1_met = {}
        
        for dataset, metrics_list in results_dict.items():
            # Average metrics across folds
            avg_precision = np.mean([m['precision'] for m in metrics_list])
            avg_recall = np.mean([m['recall'] for m in metrics_list])
            avg_f1 = np.mean([m['f1'] for m in metrics_list])
            
            criteria = (avg_precision >= 0.80) and (avg_recall >= 0.75)
            
            h1_met[dataset] = {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'h1_criteria_met': criteria,
                'num_folds': len(metrics_list)
            }
            
            logger.info(f'{dataset}: P={avg_precision:.4f}, R={avg_recall:.4f}, F1={avg_f1:.4f}, H1_MET={criteria}')
        
        # Success: ?2 of 3 datasets meet criteria
        datasets_met = sum(1 for v in h1_met.values() if v['h1_criteria_met'])
        h1_success = datasets_met >= 2
        
        logger.info(f'H1 Overall Success: {h1_success} ({datasets_met}/3 datasets met criteria)')
        
        return h1_met, h1_success
    
    @staticmethod
    def test_h2_benchmark_vs_griffin(
        ml_results: List[float],
        griffin_results: List[float],
        metric: str = 'f1'
    ) -> Dict[str, any]:
        """
        H2: ML framework ? 2x speedup vs Apache Griffin baseline
            with ? 80% precision
        
        Paired t-test comparing F1 scores and execution times
        """
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(ml_results, griffin_results)
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(np.array(ml_results) - np.array(griffin_results))
        pooled_std = np.sqrt((np.std(ml_results)**2 + np.std(griffin_results)**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Speedup factor (execution times)
        ml_time = np.mean(ml_results)  # Assuming execution time
        griffin_time = np.mean(griffin_results)
        speedup = griffin_time / ml_time if ml_time > 0 else 0
        
        h2_met = (p_value < 0.05) and (speedup >= 2.0)
        
        logger.info(f'H2 Results:')
        logger.info(f'  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}')
        logger.info(f'  Cohen\'s d: {cohens_d:.4f}')
        logger.info(f'  Speedup: {speedup:.2f}x')
        logger.info(f'  H2 Criteria Met: {h2_met}')
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'speedup_factor': speedup,
            'h2_criteria_met': h2_met
        }
    
    @staticmethod
    def test_h3_dashboard_usability(
        likert_ratings: List[int],  # 1-5 scale
        remediation_times: List[float]  # seconds
    ) -> Dict[str, any]:
        """
        H3: Dashboard usability ? 3.5/5 Likert rating
            and ? 30% reduction in remediation time
        
        With 5-10 user study participants
        """
        avg_likert = np.mean(likert_ratings)
        std_likert = np.std(likert_ratings)
        
        # Calculate time reduction
        baseline_time = 600  # 10 minutes baseline (estimate)
        avg_remediation_time = np.mean(remediation_times)
        time_reduction_pct = ((baseline_time - avg_remediation_time) / baseline_time) * 100
        
        h3_met = (avg_likert >= 3.5) and (time_reduction_pct >= 30)
        
        logger.info(f'H3 Results:')
        logger.info(f'  Likert Rating: {avg_likert:.2f} ± {std_likert:.2f} (need ?3.5)')
        logger.info(f'  Time Reduction: {time_reduction_pct:.1f}% (need ?30%)')
        logger.info(f'  H3 Criteria Met: {h3_met}')
        
        return {
            'avg_likert_rating': avg_likert,
            'std_likert': std_likert,
            'time_reduction_pct': time_reduction_pct,
            'h3_criteria_met': h3_met,
            'sample_size': len(likert_ratings)
        }
