"""
Policy rules and feature distribution definitions for AIPD-100K Dataset Generator.

This module implements the literature-derived interview rules that determine the next interview policy
based on the candidate's current state. Each rule includes specific feature distribution parameters
to generate realistic synthetic data.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from scipy import stats

from config import InterviewPolicy, DifficultyLevel, get_config, get_distribution_params


@dataclass
class FeatureDistribution:
    """Defines the distribution parameters for a single feature."""
    mean: float
    std: float
    min_val: float
    max_val: float
    
    def sample(self, size: int = 1, random_state: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Sample from a truncated normal distribution."""
        if random_state is None:
            random_state = np.random
        
        # Truncated normal distribution
        a, b = (self.min_val - self.mean) / self.std, (self.max_val - self.mean) / self.std
        samples = stats.truncnorm.rvs(a, b, loc=self.mean, scale=self.std, size=size, random_state=random_state)
        
        # Ensure samples are within bounds (handle edge cases)
        samples = np.clip(samples, self.min_val, self.max_val)
        
        if size == 1:
            return samples[0]
        return samples


@dataclass
class PolicyRule:
    """Defines a complete policy rule with conditions and feature distributions."""
    
    policy: InterviewPolicy
    rule_id: int
    description: str
    
    # Rule conditions (for validation)
    conditions: Dict[str, any]
    
    # Feature distributions for generation
    distributions: Dict[str, FeatureDistribution]
    
    def matches_conditions(self, row: Dict[str, any]) -> bool:
        """Check if a row matches this rule's conditions."""
        for feature, condition in self.conditions.items():
            if isinstance(condition, tuple):
                # Range condition (min, max)
                if not (condition[0] <= row.get(feature, float('-inf')) <= condition[1]):
                    return False
            elif isinstance(condition, list):
                # List of acceptable values
                if row.get(feature) not in condition:
                    return False
            elif isinstance(condition, dict):
                # Complex condition
                if 'min' in condition and row.get(feature, float('-inf')) < condition['min']:
                    return False
                if 'max' in condition and row.get(feature, float('inf')) > condition['max']:
                    return False
                if 'eq' in condition and row.get(feature) != condition['eq']:
                    return False
                if 'ne' in condition and row.get(feature) == condition['ne']:
                    return False
                if 'gte' in condition and row.get(feature, float('-inf')) < condition['gte']:
                    return False
                if 'lte' in condition and row.get(feature, float('inf')) > condition['lte']:
                    return False
            else:
                # Exact match
                if row.get(feature) != condition:
                    return False
        return True
    
    def generate_features(self, random_state: np.random.RandomState) -> Dict[str, any]:
        """Generate features according to this rule's distributions."""
        row = {}
        for feature, dist in self.distributions.items():
            row[feature] = dist.sample(size=1, random_state=random_state)
        return row


class RuleEngine:
    """Manages all policy rules and provides rule matching/generation functionality."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize the rule engine with all policy rules."""
        self.random_state = np.random.RandomState(random_seed)
        self.rules: List[PolicyRule] = []
        self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize all policy rules with their conditions and distributions."""
        config = get_config()
        dist_params = get_distribution_params()
        
        # Rule 1: Switch Topic (Hard difficulty, excellent performance, long correct streak)
        self.rules.append(PolicyRule(
            policy=InterviewPolicy.SWITCH_TOPIC,
            rule_id=1,
            description="Candidate has mastered a Hard topic with consecutive correct answers",
            conditions={
                'Difficulty': [DifficultyLevel.HARD.value],
                'Correctness Score': {'gte': 90},
                'Concept Coverage': {'gte': 88},
                'Reasoning Score': {'gte': 88},
                'Missing Concepts': {'lte': 1},
                'Correct Streak': {'gte': 3}
            },
            distributions={
                'Correctness Score': FeatureDistribution(mean=95, std=1.5, min_val=90, max_val=100),
                'Concept Coverage': FeatureDistribution(mean=93, std=2.0, min_val=88, max_val=100),
                'Reasoning Score': FeatureDistribution(mean=94, std=1.5, min_val=88, max_val=100),
                'Missing Concepts': FeatureDistribution(mean=0.2, std=0.3, min_val=0, max_val=1),
                'Engagement Score': FeatureDistribution(mean=0.92, std=0.03, min_val=0.85, max_val=0.98),
                'Confidence Score': FeatureDistribution(mean=0.91, std=0.03, min_val=0.85, max_val=0.98),
                'Hesitation Score': FeatureDistribution(mean=0.09, std=0.03, min_val=0.05, max_val=0.15),
                'Eye Contact Score': FeatureDistribution(mean=0.92, std=0.03, min_val=0.85, max_val=0.98),
                'Difficulty': DifficultyLevel.HARD.value,
                'Correct Streak': FeatureDistribution(mean=4.0, std=0.5, min_val=3, max_val=5),
                'Wrong Streak': 0
            }
        ))
        
        # Rule 2: Reduce Difficulty (Poor performance or wrong streak)
        self.rules.append(PolicyRule(
            policy=InterviewPolicy.REDUCE_DIFFICULTY,
            rule_id=2,
            description="Candidate struggling with current difficulty level",
            conditions={
                'OR': [
                    {'Correctness Score': {'lt': 45}},
                    {'Wrong Streak': {'gte': 3}}
                ]
            },
            distributions={
                'Correctness Score': FeatureDistribution(mean=35, std=8.0, min_val=0, max_val=45),
                'Concept Coverage': FeatureDistribution(mean=38, std=8.0, min_val=0, max_val=50),
                'Reasoning Score': FeatureDistribution(mean=33, std=8.0, min_val=0, max_val=48),
                'Missing Concepts': FeatureDistribution(mean=5.5, std=1.5, min_val=3, max_val=8),
                'Engagement Score': FeatureDistribution(mean=0.35, std=0.12, min_val=0.20, max_val=0.55),
                'Confidence Score': FeatureDistribution(mean=0.32, std=0.12, min_val=0.20, max_val=0.50),
                'Hesitation Score': FeatureDistribution(mean=0.75, std=0.12, min_val=0.60, max_val=0.95),
                'Eye Contact Score': FeatureDistribution(mean=0.33, std=0.12, min_val=0.20, max_val=0.55),
                'Difficulty': [DifficultyLevel.MEDIUM.value, DifficultyLevel.HARD.value],
                'Correct Streak': FeatureDistribution(mean=0.5, std=0.5, min_val=0, max_val=2),
                'Wrong Streak': FeatureDistribution(mean=3.5, std=1.0, min_val=3, max_val=5)
            }
        ))
        
        # Rule 3: Probe Missing Concept (Good performance but missing concepts)
        self.rules.append(PolicyRule(
            policy=InterviewPolicy.PROBE_MISSING_CONCEPT,
            rule_id=3,
            description="Good performance but with conceptual gaps",
            conditions={
                'Missing Concepts': {'gte': 4},
                'Correctness Score': {'gte': 50}
            },
            distributions={
                'Correctness Score': FeatureDistribution(mean=65, std=10.0, min_val=50, max_val=85),
                'Concept Coverage': FeatureDistribution(mean=62, std=10.0, min_val=45, max_val=80),
                'Reasoning Score': FeatureDistribution(mean=68, std=10.0, min_val=50, max_val=85),
                'Missing Concepts': FeatureDistribution(mean=5.0, std=1.2, min_val=4, max_val=8),
                'Engagement Score': FeatureDistribution(mean=0.58, std=0.12, min_val=0.45, max_val=0.75),
                'Confidence Score': FeatureDistribution(mean=0.55, std=0.12, min_val=0.45, max_val=0.72),
                'Hesitation Score': FeatureDistribution(mean=0.45, std=0.12, min_val=0.30, max_val=0.60),
                'Eye Contact Score': FeatureDistribution(mean=0.57, std=0.12, min_val=0.45, max_val=0.75),
                'Difficulty': [DifficultyLevel.EASY.value, DifficultyLevel.MEDIUM.value],
                'Correct Streak': FeatureDistribution(mean=1.0, std=0.8, min_val=0, max_val=3),
                'Wrong Streak': FeatureDistribution(mean=0.8, std=0.8, min_val=0, max_val=2)
            }
        ))
        
        # Rule 4: Ask Application Question (Good theoretical understanding on Medium)
        self.rules.append(PolicyRule(
            policy=InterviewPolicy.ASK_APPLICATION_QUESTION,
            rule_id=4,
            description="Theory is understood, test application skills",
            conditions={
                'Correctness Score': {'gte': 80},
                'Concept Coverage': {'gte': 80},
                'Reasoning Score': {'gte': 80},
                'Missing Concepts': {'lte': 2},
                'Difficulty': DifficultyLevel.MEDIUM.value
            },
            distributions={
                'Correctness Score': FeatureDistribution(mean=86, std=3.0, min_val=80, max_val=95),
                'Concept Coverage': FeatureDistribution(mean=84, std=3.0, min_val=80, max_val=95),
                'Reasoning Score': FeatureDistribution(mean=85, std=3.0, min_val=80, max_val=95),
                'Missing Concepts': FeatureDistribution(mean=1.0, std=0.8, min_val=0, max_val=2),
                'Engagement Score': FeatureDistribution(mean=0.78, std=0.08, min_val=0.65, max_val=0.90),
                'Confidence Score': FeatureDistribution(mean=0.75, std=0.08, min_val=0.60, max_val=0.85),
                'Hesitation Score': FeatureDistribution(mean=0.22, std=0.08, min_val=0.15, max_val=0.40),
                'Eye Contact Score': FeatureDistribution(mean=0.76, std=0.08, min_val=0.65, max_val=0.90),
                'Difficulty': DifficultyLevel.MEDIUM.value,
                'Correct Streak': FeatureDistribution(mean=1.5, std=0.8, min_val=0, max_val=3),
                'Wrong Streak': FeatureDistribution(mean=0.3, std=0.5, min_val=0, max_val=1)
            }
        ))
        
        # Rule 5: Ask Follow-up Question (Good reasoning, need deeper probing)
        self.rules.append(PolicyRule(
            policy=InterviewPolicy.ASK_FOLLOW_UP_QUESTION,
            rule_id=5,
            description="Strong reasoning, need deeper understanding check",
            conditions={
                'Correctness Score': (70, 85),
                'Concept Coverage': (65, 80),
                'Reasoning Score': {'gte': 80},
                'Missing Concepts': {'lte': 3}
            },
            distributions={
                'Correctness Score': FeatureDistribution(mean=77, std=4.0, min_val=70, max_val=85),
                'Concept Coverage': FeatureDistribution(mean=72, std=4.0, min_val=65, max_val=80),
                'Reasoning Score': FeatureDistribution(mean=84, std=3.0, min_val=80, max_val=92),
                'Missing Concepts': FeatureDistribution(mean=1.8, std=0.8, min_val=0, max_val=3),
                'Engagement Score': FeatureDistribution(mean=0.70, std=0.10, min_val=0.55, max_val=0.85),
                'Confidence Score': FeatureDistribution(mean=0.67, std=0.10, min_val=0.55, max_val=0.82),
                'Hesitation Score': FeatureDistribution(mean=0.30, std=0.10, min_val=0.18, max_val=0.45),
                'Eye Contact Score': FeatureDistribution(mean=0.68, std=0.10, min_val=0.55, max_val=0.85),
                'Difficulty': [DifficultyLevel.EASY.value, DifficultyLevel.MEDIUM.value],
                'Correct Streak': FeatureDistribution(mean=1.0, std=0.8, min_val=0, max_val=3),
                'Wrong Streak': FeatureDistribution(mean=0.5, std=0.6, min_val=0, max_val=2)
            }
        ))
        
        # Rule 6: Increase Difficulty (Excellent performance on Easy)
        self.rules.append(PolicyRule(
            policy=InterviewPolicy.INCREASE_DIFFICULTY,
            rule_id=6,
            description="Excellent performance on Easy with consecutive correct answers",
            conditions={
                'Difficulty': DifficultyLevel.EASY.value,
                'Correctness Score': {'gte': 90},
                'Concept Coverage': {'gte': 85},
                'Correct Streak': {'gte': 2}
            },
            distributions={
                'Correctness Score': FeatureDistribution(mean=95, std=2.0, min_val=90, max_val=100),
                'Concept Coverage': FeatureDistribution(mean=92, std=2.5, min_val=85, max_val=100),
                'Reasoning Score': FeatureDistribution(mean=93, std=2.0, min_val=85, max_val=100),
                'Missing Concepts': FeatureDistribution(mean=0.3, std=0.4, min_val=0, max_val=2),
                'Engagement Score': FeatureDistribution(mean=0.90, std=0.04, min_val=0.82, max_val=0.98),
                'Confidence Score': FeatureDistribution(mean=0.88, std=0.04, min_val=0.82, max_val=0.98),
                'Hesitation Score': FeatureDistribution(mean=0.10, std=0.04, min_val=0.05, max_val=0.18),
                'Eye Contact Score': FeatureDistribution(mean=0.89, std=0.04, min_val=0.82, max_val=0.98),
                'Difficulty': DifficultyLevel.EASY.value,
                'Correct Streak': FeatureDistribution(mean=3.0, std=0.7, min_val=2, max_val=5),
                'Wrong Streak': 0
            }
        ))
        
        # Rule 7: Maintain Difficulty (Default/No strong trend)
        self.rules.append(PolicyRule(
            policy=InterviewPolicy.MAINTAIN_DIFFICULTY,
            rule_id=7,
            description="No strong trend, maintain current difficulty",
            conditions={},  # This is the default rule
            distributions={
                'Correctness Score': FeatureDistribution(mean=60, std=15.0, min_val=45, max_val=85),
                'Concept Coverage': FeatureDistribution(mean=58, std=15.0, min_val=42, max_val=82),
                'Reasoning Score': FeatureDistribution(mean=62, std=15.0, min_val=45, max_val=85),
                'Missing Concepts': FeatureDistribution(mean=3.0, std=1.5, min_val=1, max_val=6),
                'Engagement Score': FeatureDistribution(mean=0.55, std=0.15, min_val=0.35, max_val=0.80),
                'Confidence Score': FeatureDistribution(mean=0.52, std=0.15, min_val=0.35, max_val=0.78),
                'Hesitation Score': FeatureDistribution(mean=0.48, std=0.15, min_val=0.25, max_val=0.70),
                'Eye Contact Score': FeatureDistribution(mean=0.54, std=0.15, min_val=0.35, max_val=0.80),
                'Difficulty': [DifficultyLevel.EASY.value, DifficultyLevel.MEDIUM.value, DifficultyLevel.HARD.value],
                'Correct Streak': FeatureDistribution(mean=1.0, std=0.8, min_val=0, max_val=3),
                'Wrong Streak': FeatureDistribution(mean=1.0, std=0.8, min_val=0, max_val=3)
            }
        ))
    
    def get_rule_for_policy(self, policy: InterviewPolicy) -> PolicyRule:
        """Get the rule that generates a specific policy."""
        for rule in self.rules:
            if rule.policy == policy:
                return rule
        raise ValueError(f"No rule found for policy: {policy}")
    
    def match_rule(self, row: Dict[str, any]) -> Optional[PolicyRule]:
        """
        Match a row to the first applicable rule (top-to-bottom priority).
        
        Args:
            row: Dictionary of feature values
            
        Returns:
            First matching rule, or None if no rule matches
        """
        for rule in self.rules:
            if self._rule_matches(rule, row):
                return rule
        return None
    
    def _rule_matches(self, rule: PolicyRule, row: Dict[str, any]) -> bool:
        """Check if a specific rule matches the given row."""
        conditions = rule.conditions
        
        # Handle OR conditions
        if 'OR' in conditions:
            for or_condition in conditions['OR']:
                if self._condition_matches(or_condition, row):
                    return True
            return False
        
        # Handle regular conditions
        return self._condition_matches(conditions, row)
    
    def _condition_matches(self, conditions: Dict[str, any], row: Dict[str, any]) -> bool:
        """Check if conditions match the row."""
        for feature, condition in conditions.items():
            if isinstance(condition, tuple):
                # Range condition (min, max)
                if not (condition[0] <= row.get(feature, float('-inf')) <= condition[1]):
                    return False
            elif isinstance(condition, list):
                # List of acceptable values
                if row.get(feature) not in condition:
                    return False
            elif isinstance(condition, dict):
                # Complex condition
                if 'min' in condition and row.get(feature, float('-inf')) < condition['min']:
                    return False
                if 'max' in condition and row.get(feature, float('inf')) > condition['max']:
                    return False
                if 'eq' in condition and row.get(feature) != condition['eq']:
                    return False
                if 'ne' in condition and row.get(feature) == condition['ne']:
                    return False
                if 'gte' in condition and row.get(feature, float('-inf')) < condition['gte']:
                    return False
                if 'lte' in condition and row.get(feature, float('inf')) > condition['lte']:
                    return False
                if 'lt' in condition and row.get(feature, float('inf')) >= condition['lt']:
                    return False
                if 'gt' in condition and row.get(feature, float('-inf')) <= condition['gt']:
                    return False
            else:
                # Exact match
                if row.get(feature) != condition:
                    return False
        return True
    
    def get_all_rules(self) -> List[PolicyRule]:
        """Get all policy rules."""
        return self.rules
    
    def get_rule_summary(self) -> Dict[str, str]:
        """Get a summary of all rules."""
        return {rule.policy.value: rule.description for rule in self.rules}
