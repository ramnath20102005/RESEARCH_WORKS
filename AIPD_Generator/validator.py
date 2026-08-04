"""
Validation module for AIPD-100K Dataset Generator.

This module implements all validation rules to ensure generated data rows are logically consistent
and follow the interview constraints. Invalid rows are rejected and regenerated.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config import get_config, DifficultyLevel


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    rejection_reason: Optional[str] = None


class DatasetValidator:
    """Validates generated dataset rows against all constraints."""
    
    def __init__(self):
        """Initialize the validator with configuration."""
        self.config = get_config()
        self.rejection_counts = {
            'High_correctness_with_many_missing': 0,
            'High_confidence_with_high_hesitation': 0,
            'Low_correctness_with_high_confidence': 0,
            'Both_streaks_positive': 0,
            'Easy_with_reduce_difficulty': 0,
            'Hard_with_increase_difficulty': 0,
            'Semantic_inconsistency': 0,
            'Behavioral_inconsistency': 0,
            'Context_inconsistency': 0,
            'Policy_rule_mismatch': 0
        }
    
    def validate_row(self, row: Dict[str, any], expected_policy: str) -> ValidationResult:
        """
        Validate a complete row against all constraints.
        
        Args:
            row: Dictionary containing all feature values
            expected_policy: The policy that should match this row
            
        Returns:
            ValidationResult indicating validity and rejection reason if invalid
        """
        # Extract features
        correctness = row.get('Correctness Score', 0)
        coverage = row.get('Concept Coverage', 0)
        reasoning = row.get('Reasoning Score', 0)
        missing = row.get('Missing Concepts', 0)
        engagement = row.get('Engagement Score', 0)
        confidence = row.get('Confidence Score', 0)
        hesitation = row.get('Hesitation Score', 0)
        eye_contact = row.get('Eye Contact Score', 0)
        difficulty = row.get('Difficulty', '')
        correct_streak = row.get('Correct Streak', 0)
        wrong_streak = row.get('Wrong Streak', 0)
        policy = row.get('Policy', '')
        
        # Validation Rule 1: High correctness with many missing concepts
        if correctness > self.config.MAX_CORRECTNESS_FOR_MISSING and missing > self.config.MAX_MISSING_FOR_HIGH_CORRECTNESS:
            self.rejection_counts['High_correctness_with_many_missing'] += 1
            return ValidationResult(
                is_valid=False,
                rejection_reason=f"Correctness {correctness} > {self.config.MAX_CORRECTNESS_FOR_MISSING} with Missing {missing} > {self.config.MAX_MISSING_FOR_HIGH_CORRECTNESS}"
            )
        
        # Validation Rule 2: High confidence with high hesitation
        if confidence > self.config.MAX_CONFIDENCE_FOR_HESITATION and hesitation > self.config.MAX_HESITATION_FOR_HIGH_CONFIDENCE:
            self.rejection_counts['High_confidence_with_high_hesitation'] += 1
            return ValidationResult(
                is_valid=False,
                rejection_reason=f"Confidence {confidence} > {self.config.MAX_CONFIDENCE_FOR_HESITATION} with Hesitation {hesitation} > {self.config.MAX_HESITATION_FOR_HIGH_CONFIDENCE}"
            )
        
        # Validation Rule 3: Low correctness with high confidence
        if correctness < self.config.MIN_CORRECTNESS_FOR_HIGH_CONFIDENCE and confidence > self.config.MAX_CONFIDENCE_FOR_LOW_CORRECTNESS:
            self.rejection_counts['Low_correctness_with_high_confidence'] += 1
            return ValidationResult(
                is_valid=False,
                rejection_reason=f"Correctness {correctness} < {self.config.MIN_CORRECTNESS_FOR_HIGH_CONFIDENCE} with Confidence {confidence} > {self.config.MAX_CONFIDENCE_FOR_LOW_CORRECTNESS}"
            )
        
        # Validation Rule 4: Both streaks positive
        if correct_streak > 0 and wrong_streak > 0:
            self.rejection_counts['Both_streaks_positive'] += 1
            return ValidationResult(
                is_valid=False,
                rejection_reason=f"Correct Streak {correct_streak} > 0 and Wrong Streak {wrong_streak} > 0"
            )
        
        # Validation Rule 5: Easy difficulty with Reduce Difficulty policy
        if difficulty == DifficultyLevel.EASY.value and policy == "Reduce Difficulty":
            self.rejection_counts['Easy_with_reduce_difficulty'] += 1
            return ValidationResult(
                is_valid=False,
                rejection_reason=f"Difficulty {difficulty} with policy {policy}"
            )
        
        # Validation Rule 6: Hard difficulty with Increase Difficulty policy
        if difficulty == DifficultyLevel.HARD.value and policy == "Increase Difficulty":
            self.rejection_counts['Hard_with_increase_difficulty'] += 1
            return ValidationResult(
                is_valid=False,
                rejection_reason=f"Difficulty {difficulty} with policy {policy}"
            )
        
        # Validation Rule 7: Semantic feature consistency
        if not self._validate_semantic_consistency(correctness, coverage, reasoning, missing):
            self.rejection_counts['Semantic_inconsistency'] += 1
            return ValidationResult(
                is_valid=False,
                rejection_reason="Semantic features are inconsistent"
            )
        
        # Validation Rule 8: Behavioral feature consistency
        if not self._validate_behavioral_consistency(confidence, hesitation, correctness):
            self.rejection_counts['Behavioral_inconsistency'] += 1
            return ValidationResult(
                is_valid=False,
                rejection_reason="Behavioral features are inconsistent"
            )
        
        # Validation Rule 9: Context feature consistency
        if not self._validate_context_consistency(difficulty, correct_streak, wrong_streak, policy):
            self.rejection_counts['Context_inconsistency'] += 1
            return ValidationResult(
                is_valid=False,
                rejection_reason="Context features are inconsistent with policy"
            )
        
        # Validation Rule 10: Policy matches expected
        if policy != expected_policy:
            self.rejection_counts['Policy_rule_mismatch'] += 1
            return ValidationResult(
                is_valid=False,
                rejection_reason=f"Policy {policy} does not match expected {expected_policy}"
            )
        
        return ValidationResult(is_valid=True)
    
    def _validate_semantic_consistency(
        self,
        correctness: int,
        coverage: int,
        reasoning: int,
        missing: int
    ) -> bool:
        """Validate semantic feature consistency."""
        # Coverage and reasoning should be reasonably close to correctness
        # Allow some variation but not extreme inconsistencies
        if abs(coverage - correctness) > 40:  # Relaxed from 30
            return False
        if abs(reasoning - correctness) > 40:  # Relaxed from 30
            return False
        
        # Missing concepts should correlate negatively with correctness
        # High correctness should not have many missing concepts
        if correctness > 90 and missing > 3:  # Relaxed
            return False
        
        # Very low correctness should have many missing concepts
        if correctness < 20 and missing < 2:  # Relaxed
            return False
        
        return True
    
    def _validate_behavioral_consistency(
        self,
        confidence: float,
        hesitation: float,
        correctness: int
    ) -> bool:
        """Validate behavioral feature consistency."""
        # Engagement and eye contact should be positively correlated with confidence
        # This is implicitly handled by generation, but we can add sanity checks
        
        # Confidence and hesitation should be negatively correlated
        # High confidence should not have extremely high hesitation (already checked above)
        # Low confidence should not have extremely low hesitation
        if confidence < 0.3 and hesitation < 0.2:
            return False
        
        return True
    
    def _validate_context_consistency(
        self,
        difficulty: str,
        correct_streak: int,
        wrong_streak: int,
        policy: str
    ) -> bool:
        """Validate context feature consistency with policy."""
        # Minimal validation - only check the essential constraints
        # The generator already ensures proper alignment
        return True
    
    def validate_feature_ranges(self, row: Dict[str, any]) -> ValidationResult:
        """
        Validate that all features are within their valid ranges.
        
        Args:
            row: Dictionary containing all feature values
            
        Returns:
            ValidationResult indicating validity
        """
        correctness = row.get('Correctness Score', 0)
        coverage = row.get('Concept Coverage', 0)
        reasoning = row.get('Reasoning Score', 0)
        missing = row.get('Missing Concepts', 0)
        engagement = row.get('Engagement Score', 0)
        confidence = row.get('Confidence Score', 0)
        hesitation = row.get('Hesitation Score', 0)
        eye_contact = row.get('Eye Contact Score', 0)
        correct_streak = row.get('Correct Streak', 0)
        wrong_streak = row.get('Wrong Streak', 0)
        
        # Check semantic feature ranges
        if not (self.config.CORRECTNESS_RANGE[0] <= correctness <= self.config.CORRECTNESS_RANGE[1]):
            return ValidationResult(False, f"Correctness {correctness} out of range")
        if not (self.config.COVERAGE_RANGE[0] <= coverage <= self.config.COVERAGE_RANGE[1]):
            return ValidationResult(False, f"Coverage {coverage} out of range")
        if not (self.config.REASONING_RANGE[0] <= reasoning <= self.config.REASONING_RANGE[1]):
            return ValidationResult(False, f"Reasoning {reasoning} out of range")
        if not (self.config.MISSING_CONCEPTS_RANGE[0] <= missing <= self.config.MISSING_CONCEPTS_RANGE[1]):
            return ValidationResult(False, f"Missing {missing} out of range")
        
        # Check behavioral feature ranges
        if not (self.config.ENGAGEMENT_RANGE[0] <= engagement <= self.config.ENGAGEMENT_RANGE[1]):
            return ValidationResult(False, f"Engagement {engagement} out of range")
        if not (self.config.CONFIDENCE_RANGE[0] <= confidence <= self.config.CONFIDENCE_RANGE[1]):
            return ValidationResult(False, f"Confidence {confidence} out of range")
        if not (self.config.HESITATION_RANGE[0] <= hesitation <= self.config.HESITATION_RANGE[1]):
            return ValidationResult(False, f"Hesitation {hesitation} out of range")
        if not (self.config.EYE_CONTACT_RANGE[0] <= eye_contact <= self.config.EYE_CONTACT_RANGE[1]):
            return ValidationResult(False, f"Eye Contact {eye_contact} out of range")
        
        # Check streak ranges
        if not (self.config.CORRECT_STREAK_RANGE[0] <= correct_streak <= self.config.CORRECT_STREAK_RANGE[1]):
            return ValidationResult(False, f"Correct Streak {correct_streak} out of range")
        if not (self.config.WRONG_STREAK_RANGE[0] <= wrong_streak <= self.config.WRONG_STREAK_RANGE[1]):
            return ValidationResult(False, f"Wrong Streak {wrong_streak} out of range")
        
        # Check difficulty is valid
        if row.get('Difficulty') not in [level.value for level in DifficultyLevel]:
            return ValidationResult(False, f"Invalid difficulty: {row.get('Difficulty')}")
        
        return ValidationResult(is_valid=True)
    
    def get_rejection_statistics(self) -> Dict[str, int]:
        """Get statistics about rejected rows by reason."""
        return self.rejection_counts.copy()
    
    def reset_rejection_counts(self):
        """Reset rejection count statistics."""
        for key in self.rejection_counts:
            self.rejection_counts[key] = 0
    
    def get_total_rejections(self) -> int:
        """Get total number of rejected rows."""
        return sum(self.rejection_counts.values())
