"""
Data validation module for Interview Training Pipeline.

This module provides additional validation beyond the basic loader validation,
focusing on data quality and consistency checks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from configs.config import Config


class DataValidator:
    """Validates data quality and consistency."""
    
    def __init__(self, config: Config):
        """
        Initialize the data validator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_feature_consistency(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate feature consistency and relationships.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': []
        }
        
        # Check 1: Correctness vs Missing Concepts should be negatively correlated
        if "Correctness Score" in df.columns and "Missing Concepts" in df.columns:
            high_correctness = df[df["Correctness Score"] > 90]
            if not high_correctness.empty:
                if (high_correctness["Missing Concepts"] > 3).any():
                    validation_results['issues'].append(
                        "High correctness (>90) with many missing concepts (>3) found"
                    )
        
        # Check 2: Streak constraint (both streaks should not be positive)
        if "Correct Streak" in df.columns and "Wrong Streak" in df.columns:
            both_positive = (df["Correct Streak"] > 0) & (df["Wrong Streak"] > 0)
            if both_positive.any():
                count = both_positive.sum()
                validation_results['issues'].append(
                    f"Found {count} rows with both positive streaks"
                )
        
        # Check 3: Difficulty-Difficulty policy constraints
        if "Difficulty" in df.columns and self.config.data.TARGET_COLUMN in df.columns:
            # Easy difficulty should not have Reduce Difficulty policy
            easy_reduce = df[(df["Difficulty"] == 0) & (df[self.config.data.TARGET_COLUMN] == 2)]
            if not easy_reduce.empty:
                validation_results['issues'].append(
                    f"Found {len(easy_reduce)} Easy difficulty samples with Reduce Difficulty policy"
                )
            
            # Hard difficulty should not have Increase Difficulty policy
            hard_increase = df[(df["Difficulty"] == 2) & (df[self.config.data.TARGET_COLUMN] == 0)]
            if not hard_increase.empty:
                validation_results['issues'].append(
                    f"Found {len(hard_increase)} Hard difficulty samples with Increase Difficulty policy"
                )
        
        # Check 4: Behavioral feature consistency
        behavioral_features = self.config.data.BEHAVIORAL_FEATURES
        for feature in behavioral_features:
            if feature in df.columns:
                if not df[feature].between(0, 1).all():
                    validation_results['issues'].append(
                        f"{feature} has values outside [0, 1] range"
                    )
        
        # Update validation status
        if validation_results['issues']:
            validation_results['is_valid'] = False
        
        return validation_results
    
    def validate_class_balance(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate class distribution balance.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary containing class balance information
        """
        target_col = self.config.data.TARGET_COLUMN
        
        if target_col not in df.columns:
            return {'error': f"Target column {target_col} not found"}
        
        class_counts = df[target_col].value_counts()
        total_samples = len(df)
        
        class_balance = {
            'total_samples': total_samples,
            'num_classes': len(class_counts),
            'class_counts': class_counts.to_dict(),
            'class_percentages': (class_counts / total_samples * 100).to_dict(),
            'min_class': class_counts.min(),
            'max_class': class_counts.max(),
            'class_imbalance_ratio': class_counts.max() / class_counts.min()
        }
        
        return class_balance
    
    def check_data_leakage(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Check for potential data leakage issues.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary containing data leakage check results
        """
        leakage_results = {
            'has_leakage': False,
            'issues': []
        }
        
        # Check for perfect correlations (>0.99)
        numerical_features = self.config.data.SEMANTIC_FEATURES + self.config.data.BEHAVIORAL_FEATURES
        numerical_df = df[numical_features]
        
        if len(numerical_df.columns) > 1:
            corr_matrix = numerical_df.corr()
            perfect_correlations = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.99:
                        perfect_correlations.append(
                            f"{corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}"
                        )
            
            if perfect_correlations:
                leakage_results['has_leakage'] = True
                leakage_results['issues'].extend(perfect_correlations)
        
        return leakage_results
