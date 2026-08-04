"""
Behavioral feature generation module for AIPD-100K Dataset Generator.

This module handles the generation of behavioral features (Engagement, Confidence, Hesitation, Eye Contact)
based on the candidate's semantic performance level. Behavioral features are conditionally generated
to maintain realistic correlations with semantic performance.
"""

import numpy as np
from typing import Dict, Tuple
from scipy import stats

from config import get_config, get_behavioral_constraints


class BehavioralFeatureGenerator:
    """Generates behavioral features based on candidate performance level."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize the behavioral feature generator."""
        self.random_state = np.random.RandomState(random_seed)
        self.config = get_config()
        self.constraints = get_behavioral_constraints()
    
    def determine_performance_level(self, performance_index: float) -> str:
        """
        Determine candidate performance level based on Performance Index.
        
        Args:
            performance_index: Performance Index (0-100)
            
        Returns:
            Performance level: 'excellent', 'good', 'average', or 'poor'
        """
        if performance_index > 90:
            return 'excellent'
        elif performance_index >= 70:
            return 'good'
        elif performance_index >= 50:
            return 'average'
        else:
            return 'poor'
    
    def get_behavioral_ranges(self, performance_level: str) -> Dict[str, Tuple[float, float]]:
        """
        Get the behavioral feature ranges for a given performance level.
        
        Args:
            performance_level: The candidate's performance level
            
        Returns:
            Dictionary mapping feature names to (min, max) ranges
        """
        if performance_level == 'excellent':
            return {
                'Engagement Score': self.constraints.EXCELLENT_ENGAGEMENT_RANGE,
                'Confidence Score': self.constraints.EXCELLENT_CONFIDENCE_RANGE,
                'Hesitation Score': self.constraints.EXCELLENT_HESITATION_RANGE,
                'Eye Contact Score': self.constraints.EXCELLENT_EYE_CONTACT_RANGE
            }
        elif performance_level == 'good':
            return {
                'Engagement Score': self.constraints.GOOD_ENGAGEMENT_RANGE,
                'Confidence Score': self.constraints.GOOD_CONFIDENCE_RANGE,
                'Hesitation Score': self.constraints.GOOD_HESITATION_RANGE,
                'Eye Contact Score': self.constraints.GOOD_EYE_CONTACT_RANGE
            }
        elif performance_level == 'average':
            return {
                'Engagement Score': self.constraints.AVERAGE_ENGAGEMENT_RANGE,
                'Confidence Score': self.constraints.AVERAGE_CONFIDENCE_RANGE,
                'Hesitation Score': self.constraints.AVERAGE_HESITATION_RANGE,
                'Eye Contact Score': self.constraints.AVERAGE_EYE_CONTACT_RANGE
            }
        else:  # poor
            return {
                'Engagement Score': self.constraints.POOR_ENGAGEMENT_RANGE,
                'Confidence Score': self.constraints.POOR_CONFIDENCE_RANGE,
                'Hesitation Score': self.constraints.POOR_HESITATION_RANGE,
                'Eye Contact Score': self.constraints.POOR_EYE_CONTACT_RANGE
            }
    
    def generate_correlated_behavioral_features(
        self,
        engagement_mean: float,
        engagement_std: float,
        confidence_mean: float,
        confidence_std: float,
        hesitation_mean: float,
        hesitation_std: float,
        eye_contact_mean: float,
        eye_contact_std: float,
        performance_index: float,
        size: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate correlated behavioral features.
        
        Behavioral features should have:
        - Positive correlations:
          * Engagement ↔ Confidence
          * Engagement ↔ Eye Contact
          * Confidence ↔ Eye Contact
        - Negative correlations:
          * Engagement ↔ Hesitation
          * Confidence ↔ Hesitation
          * Eye Contact ↔ Hesitation
          
        Args:
            engagement_mean: Target mean for engagement
            engagement_std: Target std for engagement
            confidence_mean: Target mean for confidence
            confidence_std: Target std for confidence
            hesitation_mean: Target mean for hesitation
            hesitation_std: Target std for hesitation
            eye_contact_mean: Target mean for eye contact
            eye_contact_std: Target std for eye contact
            performance_index: Performance Index (for additional correlation)
            size: Number of samples to generate
            
        Returns:
            Tuple of (engagement, confidence, hesitation, eye_contact) arrays
        """
        # Define correlation matrix for behavioral features
        # Engagement, Confidence, Eye Contact are positively correlated
        # Hesitation is negatively correlated with all three
        corr_matrix = np.array([
            [1.0, 0.8, -0.7, 0.7],  # Engagement
            [0.8, 1.0, -0.75, 0.75],  # Confidence
            [-0.7, -0.75, 1.0, -0.7],  # Hesitation
            [0.7, 0.75, -0.7, 1.0]  # Eye Contact
        ])
        
        # Convert to covariance matrix
        stds = np.array([engagement_std, confidence_std, hesitation_std, eye_contact_std])
        cov_matrix = np.outer(stds, stds) * corr_matrix
        
        # Generate multivariate normal samples
        means = np.array([engagement_mean, confidence_mean, hesitation_mean, eye_contact_mean])
        samples = self.random_state.multivariate_normal(means, cov_matrix, size=size)
        
        # Extract individual features
        engagement = samples[:, 0]
        confidence = samples[:, 1]
        hesitation = samples[:, 2]
        eye_contact = samples[:, 3]
        
        # Clip to valid ranges
        engagement = np.clip(engagement, self.config.ENGAGEMENT_RANGE[0], self.config.ENGAGEMENT_RANGE[1])
        confidence = np.clip(confidence, self.config.CONFIDENCE_RANGE[0], self.config.CONFIDENCE_RANGE[1])
        hesitation = np.clip(hesitation, self.config.HESITATION_RANGE[0], self.config.HESITATION_RANGE[1])
        eye_contact = np.clip(eye_contact, self.config.EYE_CONTACT_RANGE[0], self.config.EYE_CONTACT_RANGE[1])
        
        if size == 1:
            return engagement[0], confidence[0], hesitation[0], eye_contact[0]
        
        return engagement, confidence, hesitation, eye_contact
    
    def generate_single_sample(
        self,
        performance_index: float,
        engagement_mean: float = None,
        engagement_std: float = None,
        confidence_mean: float = None,
        confidence_std: float = None,
        hesitation_mean: float = None,
        hesitation_std: float = None,
        eye_contact_mean: float = None,
        eye_contact_std: float = None
    ) -> Dict[str, float]:
        """
        Generate a single sample of behavioral features based on Performance Index.
        
        Args:
            performance_index: Performance Index (determines performance level)
            engagement_mean: Optional override for engagement mean
            engagement_std: Optional override for engagement std
            confidence_mean: Optional override for confidence mean
            confidence_std: Optional override for confidence std
            hesitation_mean: Optional override for hesitation mean
            hesitation_std: Optional override for hesitation std
            eye_contact_mean: Optional override for eye contact mean
            eye_contact_std: Optional override for eye contact std
            
        Returns:
            Dictionary with behavioral feature values
        """
        performance_level = self.determine_performance_level(performance_index)
        ranges = self.get_behavioral_ranges(performance_level)
        
        # Use provided means/stds or calculate from ranges
        if engagement_mean is None:
            engagement_mean = (ranges['Engagement Score'][0] + ranges['Engagement Score'][1]) / 2
        if engagement_std is None:
            engagement_std = (ranges['Engagement Score'][1] - ranges['Engagement Score'][0]) / 4
        
        if confidence_mean is None:
            confidence_mean = (ranges['Confidence Score'][0] + ranges['Confidence Score'][1]) / 2
        if confidence_std is None:
            confidence_std = (ranges['Confidence Score'][1] - ranges['Confidence Score'][0]) / 4
        
        if hesitation_mean is None:
            hesitation_mean = (ranges['Hesitation Score'][0] + ranges['Hesitation Score'][1]) / 2
        if hesitation_std is None:
            hesitation_std = (ranges['Hesitation Score'][1] - ranges['Hesitation Score'][0]) / 4
        
        if eye_contact_mean is None:
            eye_contact_mean = (ranges['Eye Contact Score'][0] + ranges['Eye Contact Score'][1]) / 2
        if eye_contact_std is None:
            eye_contact_std = (ranges['Eye Contact Score'][1] - ranges['Eye Contact Score'][0]) / 4
        
        engagement, confidence, hesitation, eye_contact = self.generate_correlated_behavioral_features(
            engagement_mean, engagement_std,
            confidence_mean, confidence_std,
            hesitation_mean, hesitation_std,
            eye_contact_mean, eye_contact_std,
            performance_index,
            size=1
        )
        
        return {
            'Engagement Score': round(engagement, 3),
            'Confidence Score': round(confidence, 3),
            'Hesitation Score': round(hesitation, 3),
            'Eye Contact Score': round(eye_contact, 3)
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
    
    def validate_behavioral_consistency(
        self,
        confidence: float,
        hesitation: float,
        correctness: int
    ) -> bool:
        """
        Validate that behavioral features are logically consistent.
        
        Args:
            confidence: Confidence score
            hesitation: Hesitation score
            correctness: Correctness score
            
        Returns:
            True if features are consistent, False otherwise
        """
        # High confidence should not have very high hesitation
        if confidence > self.config.MAX_CONFIDENCE_FOR_HESITATION and hesitation > self.config.MAX_HESITATION_FOR_HIGH_CONFIDENCE:
            return False
        
        # Low correctness should not have very high confidence
        if correctness < self.config.MIN_CORRECTNESS_FOR_HIGH_CONFIDENCE and confidence > self.config.MAX_CONFIDENCE_FOR_LOW_CORRECTNESS:
            return False
        
        return True
