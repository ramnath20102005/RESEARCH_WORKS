"""
Semantic feature generation module for AIPD-100K Dataset Generator.

This module handles the generation of semantic features (Correctness, Coverage, Reasoning, Missing Concepts)
with realistic correlations and distributions based on interview rules.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats

from config import get_config, get_distribution_params


class SemanticFeatureGenerator:
    """Generates semantic features with realistic correlations."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize the semantic feature generator."""
        self.random_state = np.random.RandomState(random_seed)
        self.config = get_config()
        self.dist_params = get_distribution_params()
    
    def generate_correlated_semantic_features(
        self,
        correctness_mean: float,
        correctness_std: float,
        coverage_mean: float,
        coverage_std: float,
        reasoning_mean: float,
        reasoning_std: float,
        missing_mean: float,
        missing_std: float,
        size: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate correlated semantic features.
        
        Semantic features should have positive correlations:
        - Correctness ↔ Coverage
        - Correctness ↔ Reasoning
        - Coverage ↔ Reasoning
        
        And negative correlations:
        - Correctness ↔ Missing Concepts
        - Coverage ↔ Missing Concepts
        - Reasoning ↔ Missing Concepts
        
        Args:
            correctness_mean: Target mean for correctness
            correctness_std: Target std for correctness
            coverage_mean: Target mean for coverage
            coverage_std: Target std for coverage
            reasoning_mean: Target mean for reasoning
            reasoning_std: Target std for reasoning
            missing_mean: Target mean for missing concepts
            missing_std: Target std for missing concepts
            size: Number of samples to generate
            
        Returns:
            Tuple of (correctness, coverage, reasoning, missing) arrays
        """
        # Define correlation matrix (positive correlations between semantic features)
        # Correctness, Coverage, Reasoning are positively correlated
        # Missing Concepts is negatively correlated with all three
        corr_matrix = np.array([
            [1.0, 0.7, 0.7, -0.6],  # Correctness
            [0.7, 1.0, 0.7, -0.6],  # Coverage
            [0.7, 0.7, 1.0, -0.6],  # Reasoning
            [-0.6, -0.6, -0.6, 1.0]  # Missing Concepts
        ])
        
        # Convert to covariance matrix
        stds = np.array([correctness_std, coverage_std, reasoning_std, missing_std])
        cov_matrix = np.outer(stds, stds) * corr_matrix
        
        # Generate multivariate normal samples
        means = np.array([correctness_mean, coverage_mean, reasoning_mean, missing_mean])
        samples = self.random_state.multivariate_normal(means, cov_matrix, size=size)
        
        # Extract individual features
        correctness = samples[:, 0]
        coverage = samples[:, 1]
        reasoning = samples[:, 2]
        missing = samples[:, 3]
        
        # Clip to valid ranges
        correctness = np.clip(correctness, self.config.CORRECTNESS_RANGE[0], self.config.CORRECTNESS_RANGE[1])
        coverage = np.clip(coverage, self.config.COVERAGE_RANGE[0], self.config.COVERAGE_RANGE[1])
        reasoning = np.clip(reasoning, self.config.REASONING_RANGE[0], self.config.REASONING_RANGE[1])
        missing = np.clip(missing, self.config.MISSING_CONCEPTS_RANGE[0], self.config.MISSING_CONCEPTS_RANGE[1])
        
        # Round Missing Concepts to integer
        missing = np.round(missing).astype(int)
        
        # Round semantic features to integers
        correctness = np.round(correctness).astype(int)
        coverage = np.round(coverage).astype(int)
        reasoning = np.round(reasoning).astype(int)
        
        if size == 1:
            return correctness[0], coverage[0], reasoning[0], missing[0]
        
        return correctness, coverage, reasoning, missing
    
    def generate_single_sample(
        self,
        correctness_mean: float,
        correctness_std: float,
        coverage_mean: float,
        coverage_std: float,
        reasoning_mean: float,
        reasoning_std: float,
        missing_mean: float,
        missing_std: float
    ) -> Dict[str, int]:
        """
        Generate a single sample of semantic features.
        
        Args:
            correctness_mean: Target mean for correctness
            correctness_std: Target std for correctness
            coverage_mean: Target mean for coverage
            coverage_std: Target std for coverage
            reasoning_mean: Target mean for reasoning
            reasoning_std: Target std for reasoning
            missing_mean: Target mean for missing concepts
            missing_std: Target std for missing concepts
            
        Returns:
            Dictionary with semantic feature values
        """
        correctness, coverage, reasoning, missing = self.generate_correlated_semantic_features(
            correctness_mean, correctness_std,
            coverage_mean, coverage_std,
            reasoning_mean, reasoning_std,
            missing_mean, missing_std,
            size=1
        )
        
        return {
            'Correctness Score': correctness,
            'Concept Coverage': coverage,
            'Reasoning Score': reasoning,
            'Missing Concepts': missing
        }
    
    def generate_truncated_normal(
        self,
        mean: float,
        std: float,
        min_val: float,
        max_val: float,
        size: int = 1
    ) -> np.ndarray:
        """
        Generate samples from a truncated normal distribution.
        
        Args:
            mean: Mean of the distribution
            std: Standard deviation
            min_val: Minimum value (truncation point)
            max_val: Maximum value (truncation point)
            size: Number of samples
            
        Returns:
            Array of samples
        """
        a, b = (min_val - mean) / std, (max_val - mean) / std
        samples = stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=size, random_state=self.random_state)
        samples = np.clip(samples, min_val, max_val)
        
        if size == 1:
            return samples[0]
        return samples
    
    def adjust_for_correlation(
        self,
        base_value: float,
        target_value: float,
        correlation_strength: float
    ) -> float:
        """
        Adjust a value based on correlation with another feature.
        
        Args:
            base_value: The base feature value
            target_value: The target correlated feature's value
            correlation_strength: Strength of correlation (0-1)
            
        Returns:
            Adjusted value
        """
        # Move base_value towards target_value based on correlation strength
        return base_value + correlation_strength * (target_value - base_value)
    
    def calculate_performance_index(
        self,
        correctness: int,
        coverage: int,
        reasoning: int
    ) -> float:
        """
        Calculate Performance Index from semantic features.
        
        Performance Index = 0.45 × Correctness + 0.30 × Coverage + 0.25 × Reasoning
        
        Args:
            correctness: Correctness score (0-100)
            coverage: Concept coverage (0-100)
            reasoning: Reasoning score (0-100)
            
        Returns:
            Performance Index (0-100)
        """
        return 0.45 * correctness + 0.30 * coverage + 0.25 * reasoning
    
    def validate_semantic_consistency(
        self,
        correctness: int,
        coverage: int,
        reasoning: int,
        missing: int
    ) -> bool:
        """
        Validate that semantic features are logically consistent.
        
        Args:
            correctness: Correctness score
            coverage: Concept coverage
            reasoning: Reasoning score
            missing: Missing concepts count
            
        Returns:
            True if features are consistent, False otherwise
        """
        # High correctness should not have many missing concepts
        if correctness > self.config.MAX_CORRECTNESS_FOR_MISSING and missing > self.config.MAX_MISSING_FOR_HIGH_CORRECTNESS:
            return False
        
        # Coverage and reasoning should be reasonably close to correctness
        # Allow some variation but not extreme inconsistencies
        if abs(coverage - correctness) > 30:
            return False
        if abs(reasoning - correctness) > 30:
            return False
        
        return True
