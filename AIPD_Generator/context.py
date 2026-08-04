"""
Interview context feature generation module for AIPD-100K Dataset Generator.

This module handles the generation of interview context features (Difficulty, Correct Streak, Wrong Streak)
with the constraint that correct and wrong streaks cannot both be positive simultaneously.
"""

import numpy as np
from typing import Dict, Optional, List
from scipy import stats

from config import get_config, DifficultyLevel


class ContextFeatureGenerator:
    """Generates interview context features with realistic constraints."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize the context feature generator."""
        self.random_state = np.random.RandomState(random_seed)
        self.config = get_config()
    
    def generate_difficulty(
        self,
        allowed_difficulties: Optional[List[str]] = None,
        probabilities: Optional[List[float]] = None
    ) -> str:
        """
        Generate a difficulty level.
        
        Args:
            allowed_difficulties: List of allowed difficulty levels (default: all)
            probabilities: Probability weights for each difficulty (default: uniform)
            
        Returns:
            Difficulty level string
        """
        if allowed_difficulties is None:
            allowed_difficulties = [level.value for level in DifficultyLevel]
        
        if probabilities is None:
            probabilities = [1.0 / len(allowed_difficulties)] * len(allowed_difficulties)
        
        # Normalize probabilities
        probabilities = np.array(probabilities) / sum(probabilities)
        
        return self.random_state.choice(allowed_difficulties, p=probabilities)
    
    def generate_streaks(
        self,
        correct_streak_mean: float = None,
        correct_streak_std: float = None,
        wrong_streak_mean: float = None,
        wrong_streak_std: float = None,
        force_one_active: bool = True
    ) -> Dict[str, int]:
        """
        Generate correct and wrong streaks with the constraint that they cannot both be positive.
        
        Args:
            correct_streak_mean: Target mean for correct streak
            correct_streak_std: Target std for correct streak
            wrong_streak_mean: Target mean for wrong streak
            wrong_streak_std: Target std for wrong streak
            force_one_active: If True, ensure at least one streak is positive
            
        Returns:
            Dictionary with 'Correct Streak' and 'Wrong Streak' values
        """
        # Set default means/stds if not provided
        if correct_streak_mean is None:
            correct_streak_mean = 1.0
        if correct_streak_std is None:
            correct_streak_std = 1.0
        if wrong_streak_mean is None:
            wrong_streak_mean = 1.0
        if wrong_streak_std is None:
            wrong_streak_std = 1.0
        
        # Generate initial streak values
        correct_streak = self.generate_truncated_normal(
            correct_streak_mean, correct_streak_std,
            self.config.CORRECT_STREAK_RANGE[0],
            self.config.CORRECT_STREAK_RANGE[1]
        )
        wrong_streak = self.generate_truncated_normal(
            wrong_streak_mean, wrong_streak_std,
            self.config.WRONG_STREAK_RANGE[0],
            self.config.WRONG_STREAK_RANGE[1]
        )
        
        # Round to integers
        correct_streak = int(round(correct_streak))
        wrong_streak = int(round(wrong_streak))
        
        # Ensure both are not positive simultaneously
        if correct_streak > 0 and wrong_streak > 0:
            # Randomly choose which one to keep positive
            if self.random_state.random() < 0.5:
                wrong_streak = 0
            else:
                correct_streak = 0
        
        # If force_one_active, ensure at least one is positive
        if force_one_active and correct_streak == 0 and wrong_streak == 0:
            if self.random_state.random() < 0.5:
                correct_streak = self.random_state.randint(1, 3)
            else:
                wrong_streak = self.random_state.randint(1, 3)
        
        return {
            'Correct Streak': correct_streak,
            'Wrong Streak': wrong_streak
        }
    
    def generate_context_for_policy(
        self,
        policy: str,
        difficulty_override: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Generate context features tailored for a specific policy.
        
        Args:
            policy: The target interview policy
            difficulty_override: Optional override for difficulty level
            
        Returns:
            Dictionary with context features
        """
        # Generate difficulty based on policy requirements
        if difficulty_override:
            difficulty = difficulty_override
        else:
            difficulty = self._get_difficulty_for_policy(policy)
        
        # Generate streaks based on policy requirements
        streaks = self._get_streaks_for_policy(policy)
        
        return {
            'Difficulty': difficulty,
            'Correct Streak': streaks['Correct Streak'],
            'Wrong Streak': streaks['Wrong Streak']
        }
    
    def _get_difficulty_for_policy(self, policy: str) -> str:
        """Get appropriate difficulty level for a given policy."""
        # Specific difficulty requirements based on policy
        if policy == "Increase Difficulty":
            # Typically from Easy going up
            return self.generate_difficulty(
                allowed_difficulties=[DifficultyLevel.EASY.value],
                probabilities=[1.0]
            )
        elif policy == "Reduce Difficulty":
            # Typically from Medium/Hard going down
            return self.generate_difficulty(
                allowed_difficulties=[DifficultyLevel.MEDIUM.value, DifficultyLevel.HARD.value],
                probabilities=[0.6, 0.4]
            )
        elif policy == "Switch Topic":
            # Typically from Hard
            return self.generate_difficulty(
                allowed_difficulties=[DifficultyLevel.HARD.value],
                probabilities=[1.0]
            )
        elif policy == "Ask Application Question":
            # Typically from Medium
            return self.generate_difficulty(
                allowed_difficulties=[DifficultyLevel.MEDIUM.value],
                probabilities=[1.0]
            )
        elif policy == "Ask Follow-up Question":
            # Can be from Easy or Medium
            return self.generate_difficulty(
                allowed_difficulties=[DifficultyLevel.EASY.value, DifficultyLevel.MEDIUM.value],
                probabilities=[0.4, 0.6]
            )
        elif policy == "Probe Missing Concept":
            # Can be from Easy or Medium
            return self.generate_difficulty(
                allowed_difficulties=[DifficultyLevel.EASY.value, DifficultyLevel.MEDIUM.value],
                probabilities=[0.4, 0.6]
            )
        else:  # Maintain Difficulty
            # Can be any difficulty
            return self.generate_difficulty(
                allowed_difficulties=[DifficultyLevel.EASY.value, DifficultyLevel.MEDIUM.value, DifficultyLevel.HARD.value],
                probabilities=[0.3, 0.4, 0.3]
            )
    
    def _get_streaks_for_policy(self, policy: str) -> Dict[str, int]:
        """Get appropriate streak values for a given policy."""
        if policy == "Increase Difficulty":
            # Should have positive correct streak
            return self.generate_streaks(
                correct_streak_mean=2.5,
                correct_streak_std=0.8,
                wrong_streak_mean=0.2,
                wrong_streak_std=0.4,
                force_one_active=True
            )
        elif policy == "Reduce Difficulty":
            # Should have positive wrong streak or low correct streak
            if self.random_state.random() < 0.7:
                # Mostly wrong streak
                return self.generate_streaks(
                    correct_streak_mean=0.3,
                    correct_streak_std=0.5,
                    wrong_streak_mean=3.5,
                    wrong_streak_std=1.0,
                    force_one_active=True
                )
            else:
                # Low correct streak
                return self.generate_streaks(
                    correct_streak_mean=0.5,
                    correct_streak_std=0.5,
                    wrong_streak_mean=0.5,
                    wrong_streak_std=0.5,
                    force_one_active=False
                )
        elif policy == "Switch Topic":
            # Should have high correct streak
            return self.generate_streaks(
                correct_streak_mean=3.5,
                correct_streak_std=0.8,
                wrong_streak_mean=0.1,
                wrong_streak_std=0.3,
                force_one_active=True
            )
        elif policy == "Ask Application Question":
            # Can have low correct streak or zero
            return self.generate_streaks(
                correct_streak_mean=1.5,
                correct_streak_std=0.8,
                wrong_streak_mean=0.3,
                wrong_streak_std=0.5,
                force_one_active=False
            )
        elif policy == "Ask Follow-up Question":
            # Can have low streaks
            return self.generate_streaks(
                correct_streak_mean=1.0,
                correct_streak_std=0.8,
                wrong_streak_mean=0.5,
                wrong_streak_std=0.6,
                force_one_active=False
            )
        elif policy == "Probe Missing Concept":
            # Can have low streaks
            return self.generate_streaks(
                correct_streak_mean=1.0,
                correct_streak_std=0.8,
                wrong_streak_mean=0.8,
                wrong_streak_std=0.8,
                force_one_active=False
            )
        else:  # Maintain Difficulty
            # Can have low streaks or zero
            return self.generate_streaks(
                correct_streak_mean=1.0,
                correct_streak_std=0.8,
                wrong_streak_mean=1.0,
                wrong_streak_std=0.8,
                force_one_active=False
            )
    
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
    
    def validate_context_consistency(
        self,
        difficulty: str,
        correct_streak: int,
        wrong_streak: int,
        policy: str
    ) -> bool:
        """
        Validate that context features are logically consistent with the policy.
        
        Args:
            difficulty: Difficulty level
            correct_streak: Correct streak count
            wrong_streak: Wrong streak count
            policy: Interview policy
            
        Returns:
            True if context is consistent, False otherwise
        """
        # Both streaks cannot be positive
        if correct_streak > 0 and wrong_streak > 0:
            return False
        
        # Easy difficulty should not have Reduce Difficulty policy
        if difficulty == DifficultyLevel.EASY.value and policy == "Reduce Difficulty":
            return False
        
        # Hard difficulty should not have Increase Difficulty policy
        if difficulty == DifficultyLevel.HARD.value and policy == "Increase Difficulty":
            return False
        
        return True
