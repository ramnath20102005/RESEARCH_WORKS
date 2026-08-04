"""
Configuration module for AIPD-100K Dataset Generator.

This module contains all configurable parameters for generating the Adaptive Interview Policy Dataset.
All thresholds, ranges, and distribution parameters are centralized here for easy modification
and reproducibility.
"""

import dataclasses
from typing import Dict, List, Tuple
from enum import Enum


class DifficultyLevel(Enum):
    """Interview difficulty levels."""
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class InterviewPolicy(Enum):
    """Interview policy labels (output classes)."""
    INCREASE_DIFFICULTY = "Increase Difficulty"
    MAINTAIN_DIFFICULTY = "Maintain Difficulty"
    REDUCE_DIFFICULTY = "Reduce Difficulty"
    PROBE_MISSING_CONCEPT = "Probe Missing Concept"
    ASK_APPLICATION_QUESTION = "Ask Application Question"
    ASK_FOLLOW_UP_QUESTION = "Ask Follow-up Question"
    SWITCH_TOPIC = "Switch Topic"


@dataclasses.dataclass
class DatasetConfig:
    """Main dataset configuration."""
    
    # Dataset size parameters
    TOTAL_SAMPLES: int = 100000
    RANDOM_SEED: int = 42
    
    # Output paths
    OUTPUT_DIR: str = "output"
    DATASET_FILENAME: str = "AIPD_100K.csv"
    STATISTICS_FILENAME: str = "dataset_statistics.json"
    PLOTS_DIR: str = "output/plots"
    
    # Target distribution for each policy class
    POLICY_DISTRIBUTION: Dict[InterviewPolicy, int] = dataclasses.field(default_factory=lambda: {
        InterviewPolicy.INCREASE_DIFFICULTY: 15,  # TEST: Reduced for testing
        InterviewPolicy.MAINTAIN_DIFFICULTY: 15,  # TEST: Reduced for testing
        InterviewPolicy.REDUCE_DIFFICULTY: 15,  # TEST: Reduced for testing
        InterviewPolicy.PROBE_MISSING_CONCEPT: 15,  # TEST: Reduced for testing
        InterviewPolicy.ASK_APPLICATION_QUESTION: 15,  # TEST: Reduced for testing
        InterviewPolicy.ASK_FOLLOW_UP_QUESTION: 15,  # TEST: Reduced for testing
        InterviewPolicy.SWITCH_TOPIC: 10,  # TEST: Reduced for testing
    })
    
    # Feature ranges
    CORRECTNESS_RANGE: Tuple[int, int] = (0, 100)
    COVERAGE_RANGE: Tuple[int, int] = (0, 100)
    REASONING_RANGE: Tuple[int, int] = (0, 100)
    MISSING_CONCEPTS_RANGE: Tuple[int, int] = (0, 8)
    
    # Behavioral feature ranges
    ENGAGEMENT_RANGE: Tuple[float, float] = (0.0, 1.0)
    CONFIDENCE_RANGE: Tuple[float, float] = (0.0, 1.0)
    HESITATION_RANGE: Tuple[float, float] = (0.0, 1.0)
    EYE_CONTACT_RANGE: Tuple[float, float] = (0.0, 1.0)
    
    # Streak ranges
    CORRECT_STREAK_RANGE: Tuple[int, int] = (0, 5)
    WRONG_STREAK_RANGE: Tuple[int, int] = (0, 5)
    
    # GPU acceleration (optional)
    USE_GPU: bool = False
    
    # Validation thresholds
    MAX_CORRECTNESS_FOR_MISSING: int = 95
    MAX_MISSING_FOR_HIGH_CORRECTNESS: int = 3
    MAX_CONFIDENCE_FOR_HESITATION: float = 0.95
    MAX_HESITATION_FOR_HIGH_CONFIDENCE: float = 0.80
    MIN_CORRECTNESS_FOR_HIGH_CONFIDENCE: int = 30
    MAX_CONFIDENCE_FOR_LOW_CORRECTNESS: float = 0.90
    
    # Maximum regeneration attempts for invalid rows
    MAX_REGENERATION_ATTEMPTS: int = 100


@dataclasses.dataclass
class BehavioralConstraints:
    """Behavioral feature generation constraints based on candidate performance."""
    
    # Excellent candidate (Correctness > 90)
    EXCELLENT_ENGAGEMENT_RANGE: Tuple[float, float] = (0.80, 0.98)
    EXCELLENT_CONFIDENCE_RANGE: Tuple[float, float] = (0.80, 0.98)
    EXCELLENT_HESITATION_RANGE: Tuple[float, float] = (0.05, 0.20)
    EXCELLENT_EYE_CONTACT_RANGE: Tuple[float, float] = (0.80, 0.98)
    
    # Good candidate (Correctness 70-90)
    GOOD_ENGAGEMENT_RANGE: Tuple[float, float] = (0.65, 0.90)
    GOOD_CONFIDENCE_RANGE: Tuple[float, float] = (0.60, 0.85)
    GOOD_HESITATION_RANGE: Tuple[float, float] = (0.15, 0.40)
    GOOD_EYE_CONTACT_RANGE: Tuple[float, float] = (0.65, 0.90)
    
    # Average candidate (Correctness 50-70)
    AVERAGE_ENGAGEMENT_RANGE: Tuple[float, float] = (0.45, 0.70)
    AVERAGE_CONFIDENCE_RANGE: Tuple[float, float] = (0.45, 0.70)
    AVERAGE_HESITATION_RANGE: Tuple[float, float] = (0.35, 0.60)
    AVERAGE_EYE_CONTACT_RANGE: Tuple[float, float] = (0.45, 0.70)
    
    # Poor candidate (Correctness < 50)
    POOR_ENGAGEMENT_RANGE: Tuple[float, float] = (0.20, 0.55)
    POOR_CONFIDENCE_RANGE: Tuple[float, float] = (0.20, 0.50)
    POOR_HESITATION_RANGE: Tuple[float, float] = (0.60, 0.95)
    POOR_EYE_CONTACT_RANGE: Tuple[float, float] = (0.20, 0.55)


@dataclasses.dataclass
class DistributionParameters:
    """Parameters for truncated normal distributions used in feature generation."""
    
    # Default standard deviations for semantic features
    SEMANTIC_STD: float = 3.0
    BEHAVIORAL_STD: float = 0.05
    
    # Correlation strength (0-1, higher = stronger correlation)
    CORRELATION_STRENGTH: float = 0.7


# Global configuration instance
config = DatasetConfig()
behavioral_constraints = BehavioralConstraints()
distribution_params = DistributionParameters()


def get_config() -> DatasetConfig:
    """Get the global dataset configuration."""
    return config


def get_behavioral_constraints() -> BehavioralConstraints:
    """Get the global behavioral constraints."""
    return behavioral_constraints


def get_distribution_params() -> DistributionParameters:
    """Get the global distribution parameters."""
    return distribution_params
