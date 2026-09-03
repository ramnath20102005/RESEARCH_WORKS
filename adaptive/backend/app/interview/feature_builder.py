"""
Feature builder for the Adaptive Interview System.

Constructs the exact 11-feature vector used by TabPFN during training,
ensuring feature order, encoding, and ranges match the AIPD-100K dataset.
"""

import logging
import random
from typing import Dict, Any, List, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Builds the exact 11-feature vector for TabPFN prediction.
    
    Feature order (from AIPD-100K training config):
    1. Correctness Score (0-100, LLM-derived)
    2. Concept Coverage (0-100, LLM-derived)
    3. Reasoning Score (0-100, LLM-derived)
    4. Missing Concepts (0-8, LLM-derived)
    5. Engagement Score (0.0-1.0, temporary random)
    6. Confidence Score (0.0-1.0, temporary random)
    7. Hesitation Score (0.0-1.0, temporary random)
    8. Eye Contact Score (0.0-1.0, temporary random)
    9. Difficulty (0-2, LLM-derived, encoded: Easy=0, Medium=1, Hard=2)
    10. Correct Streak (0-5, calculated from session)
    11. Wrong Streak (0-5, calculated from session)
    """
    
    # Exact feature order from training config
    FEATURE_ORDER = [
        "Correctness Score",
        "Concept Coverage",
        "Reasoning Score",
        "Missing Concepts",
        "Engagement Score",
        "Confidence Score",
        "Hesitation Score",
        "Eye Contact Score",
        "Difficulty",
        "Correct Streak",
        "Wrong Streak"
    ]
    
    # Difficulty encoding (from training config)
    DIFFICULTY_ENCODING = {
        "Easy": 0,
        "Medium": 1,
        "Hard": 2
    }
    
    # Behavioral feature ranges (from dataset config)
    BEHAVIORAL_RANGES = {
        "Engagement Score": (0.0, 1.0),
        "Confidence Score": (0.0, 1.0),
        "Hesitation Score": (0.0, 1.0),
        "Eye Contact Score": (0.0, 1.0)
    }
    
    # VERIFIED feature ranges from training dataset
    VERIFIED_FEATURE_RANGES = {
        "Correctness Score": (3, 100),      # Training min was 3, but we allow 0-100
        "Concept Coverage": (0, 100),
        "Reasoning Score": (0, 100),
        "Missing Concepts": (0, 8),
        "Engagement Score": (0.0, 1.0),
        "Confidence Score": (0.0, 1.0),
        "Hesitation Score": (0.0, 1.0),
        "Eye Contact Score": (0.0, 1.0),
        "Difficulty": (0, 2),
        "Correct Streak": (0, 5),
        "Wrong Streak": (0, 5)
    }
    
    # Behavioral constraints by performance level (from dataset config)
    BEHAVIORAL_CONSTRAINTS = {
        "excellent": {  # Correctness > 90
            "Engagement Score": (0.80, 0.98),
            "Confidence Score": (0.80, 0.98),
            "Hesitation Score": (0.05, 0.20),
            "Eye Contact Score": (0.80, 0.98)
        },
        "good": {  # Correctness 70-90
            "Engagement Score": (0.65, 0.90),
            "Confidence Score": (0.60, 0.85),
            "Hesitation Score": (0.15, 0.40),
            "Eye Contact Score": (0.65, 0.90)
        },
        "average": {  # Correctness 50-70
            "Engagement Score": (0.45, 0.70),
            "Confidence Score": (0.45, 0.70),
            "Hesitation Score": (0.35, 0.60),
            "Eye Contact Score": (0.45, 0.70)
        },
        "poor": {  # Correctness < 50
            "Engagement Score": (0.20, 0.55),
            "Confidence Score": (0.20, 0.50),
            "Hesitation Score": (0.60, 0.95),
            "Eye Contact Score": (0.20, 0.55)
        }
    }
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the feature builder.
        
        Args:
            random_seed: Seed for reproducible behavioral feature generation
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        logger.info(f"[FeatureBuilder] Initialized with seed {random_seed}")
    
    def _get_performance_level(self, correctness_score: int) -> str:
        """
        Determine performance level based on correctness score.
        
        Args:
            correctness_score: The LLM-derived correctness score (0-100)
        
        Returns:
            Performance level key (excellent, good, average, poor)
        """
        if correctness_score > 90:
            return "excellent"
        elif correctness_score >= 70:
            return "good"
        elif correctness_score >= 50:
            return "average"
        else:
            return "poor"
    
    def _generate_behavioral_feature(
        self,
        feature_name: str,
        performance_level: str
    ) -> float:
        """
        Generate a temporary behavioral feature based on performance level.
        
        Args:
            feature_name: Name of the behavioral feature
            performance_level: Current performance level
        
        Returns:
            Generated feature value within valid range
        """
        constraints = self.BEHAVIORAL_CONSTRAINTS.get(performance_level, self.BEHAVIORAL_CONSTRAINTS["average"])
        min_val, max_val = constraints.get(feature_name, self.BEHAVIORAL_RANGES[feature_name])
        
        # Generate random value within constrained range
        value = random.uniform(min_val, max_val)
        
        # Clamp to valid range
        value = max(0.0, min(1.0, value))
        
        return round(value, 4)
    
    def build_feature_vector(
        self,
        llm_evaluation: Dict[str, Any],
        correct_streak: int,
        wrong_streak: int
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Build the exact 11-feature vector for TabPFN prediction.
        
        Args:
            llm_evaluation: Dictionary containing LLM-derived semantic features
            correct_streak: Current correct answer streak (from session)
            wrong_streak: Current wrong answer streak (from session)
        
        Returns:
            Tuple of (feature_vector, feature_dict)
            - feature_vector: List of 11 float values in exact order
            - feature_dict: Dictionary mapping feature names to values
        """
        # Extract semantic features from LLM evaluation
        semantic_data = llm_evaluation.get('semantic', {})
        question_assessment = llm_evaluation.get('question_assessment', {})
        
        correctness_score = semantic_data.get('correctness_score', 50)
        concept_coverage = semantic_data.get('concept_coverage', 50)
        reasoning_score = semantic_data.get('reasoning_score', 50)
        missing_concepts = semantic_data.get('missing_concepts', 3)
        
        # Log feature sources
        logger.info("")
        logger.info("=" * 60)
        logger.info("[FEATURE SOURCE]")
        logger.info("=" * 60)
        logger.info(f"Correctness Score:")
        logger.info(f"    value = {correctness_score}")
        logger.info(f"    source = NVIDIA NIM / Gemini evaluation")
        logger.info(f"Concept Coverage:")
        logger.info(f"    value = {concept_coverage}")
        logger.info(f"    source = NVIDIA NIM / Gemini evaluation")
        logger.info(f"Reasoning Score:")
        logger.info(f"    value = {reasoning_score}")
        logger.info(f"    source = NVIDIA NIM / Gemini evaluation")
        logger.info(f"Missing Concepts:")
        logger.info(f"    value = {missing_concepts}")
        logger.info(f"    source = NVIDIA NIM / Gemini evaluation")
        logger.info("")
        
        # Determine performance level for behavioral constraints
        performance_level = self._get_performance_level(correctness_score)
        
        # Generate temporary behavioral features
        engagement_score = self._generate_behavioral_feature("Engagement Score", performance_level)
        confidence_score = self._generate_behavioral_feature("Confidence Score", performance_level)
        hesitation_score = self._generate_behavioral_feature("Hesitation Score", performance_level)
        eye_contact_score = self._generate_behavioral_feature("Eye Contact Score", performance_level)
        
        logger.info(f"Engagement Score:")
        logger.info(f"    value = {engagement_score}")
        logger.info(f"    source = temporary behavioral placeholder")
        logger.info(f"Confidence Score:")
        logger.info(f"    value = {confidence_score}")
        logger.info(f"    source = temporary behavioral placeholder")
        logger.info(f"Hesitation Score:")
        logger.info(f"    value = {hesitation_score}")
        logger.info(f"    source = temporary behavioral placeholder")
        logger.info(f"Eye Contact Score:")
        logger.info(f"    value = {eye_contact_score}")
        logger.info(f"    source = temporary behavioral placeholder")
        logger.info("")
        
        # Encode difficulty
        question_difficulty = question_assessment.get('question_difficulty', 'Medium')
        difficulty_encoded = self.DIFFICULTY_ENCODING.get(question_difficulty, 1)
        
        logger.info(f"Difficulty:")
        logger.info(f"    value = {difficulty_encoded} ({question_difficulty})")
        logger.info(f"    source = current interview context / LLM")
        logger.info("")
        
        # Clamp streaks to valid range (0-5)
        correct_streak_clamped = max(0, min(5, correct_streak))
        wrong_streak_clamped = max(0, min(5, wrong_streak))
        
        logger.info(f"Correct Streak:")
        logger.info(f"    value = {correct_streak_clamped}")
        logger.info(f"    source = interview state")
        logger.info(f"Wrong Streak:")
        logger.info(f"    value = {wrong_streak_clamped}")
        logger.info(f"    source = interview state")
        logger.info("=" * 60)
        logger.info("")
        
        # Build feature dictionary
        feature_dict = {
            "Correctness Score": float(correctness_score),
            "Concept Coverage": float(concept_coverage),
            "Reasoning Score": float(reasoning_score),
            "Missing Concepts": float(missing_concepts),
            "Engagement Score": engagement_score,
            "Confidence Score": confidence_score,
            "Hesitation Score": hesitation_score,
            "Eye Contact Score": eye_contact_score,
            "Difficulty": float(difficulty_encoded),
            "Correct Streak": float(correct_streak_clamped),
            "Wrong Streak": float(wrong_streak_clamped)
        }
        
        # Build feature vector in exact order
        feature_vector = [feature_dict[feature] for feature in self.FEATURE_ORDER]
        
        # Log feature values for debugging
        logger.info(f"[FeatureBuilder] Feature vector built:")
        for i, (name, value) in enumerate(zip(self.FEATURE_ORDER, feature_vector)):
            logger.info(f"  {i+1}. {name}: {value}")
        
        return feature_vector, feature_dict
    
    def validate_feature_vector(self, feature_vector: List[float]) -> bool:
        """
        Validate that the feature vector matches VERIFIED training ranges.
        
        Args:
            feature_vector: The feature vector to validate
        
        Returns:
            True if valid, False otherwise
        """
        if len(feature_vector) != 11:
            logger.error(f"[FeatureBuilder] Invalid feature vector length: {len(feature_vector)} (expected 11)")
            return False
        
        # Validate ranges using VERIFIED training dataset ranges
        for i, (feature_name, value) in enumerate(zip(self.FEATURE_ORDER, feature_vector)):
            min_val, max_val = self.VERIFIED_FEATURE_RANGES[feature_name]
            # For Correctness Score, allow 0-100 (training min was 3 but we allow broader range)
            if feature_name == "Correctness Score":
                min_val, max_val = 0, 100
            
            if not (min_val <= value <= max_val):
                logger.error(f"[FeatureBuilder] Invalid {feature_name}: {value} (expected range: {min_val}-{max_val})")
                return False
        
        logger.info("[FeatureBuilder] Feature vector validation passed")
        return True
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about the feature schema.
        
        Returns:
            Dictionary containing feature schema information
        """
        return {
            'feature_order': self.FEATURE_ORDER,
            'n_features': len(self.FEATURE_ORDER),
            'difficulty_encoding': self.DIFFICULTY_ENCODING,
            'behavioral_ranges': self.BEHAVIORAL_RANGES,
            'behavioral_constraints': self.BEHAVIORAL_CONSTRAINTS
        }
