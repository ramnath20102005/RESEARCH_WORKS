"""
Main dataset generator module for AIPD-100K Dataset Generator.

This module orchestrates the generation of the complete dataset by coordinating
semantic, behavioral, and context feature generation, applying policy rules,
and validating all generated rows.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import time

from config import get_config, InterviewPolicy
from rules import RuleEngine, PolicyRule, FeatureDistribution
from semantic import SemanticFeatureGenerator
from behavior import BehavioralFeatureGenerator
from context import ContextFeatureGenerator
from validator import DatasetValidator, ValidationResult


class DatasetGenerator:
    """Main generator for the AIPD-100K dataset."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the dataset generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.config = get_config()
        self.random_seed = random_seed
        self.random_state = np.random.RandomState(random_seed)
        
        # Initialize sub-generators
        self.rule_engine = RuleEngine(random_seed)
        self.semantic_generator = SemanticFeatureGenerator(random_seed)
        self.behavioral_generator = BehavioralFeatureGenerator(random_seed)
        self.context_generator = ContextFeatureGenerator(random_seed)
        self.validator = DatasetValidator()
        
        # Statistics
        self.generation_stats = {
            'total_attempts': 0,
            'successful_generations': 0,
            'regenerated_rows': 0,
            'policy_counts': {policy.value: 0 for policy in InterviewPolicy}
        }
        
        # Rejected rows tracking
        self.rejected_rows = []
    
    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate the complete AIPD-100K dataset.
        
        Returns:
            DataFrame containing the generated dataset
        """
        print(f"Generating AIPD-100K dataset with {self.config.TOTAL_SAMPLES} samples...")
        print(f"Random seed: {self.random_seed}")
        print(f"Target distribution: {self.config.POLICY_DISTRIBUTION}")
        print()
        
        # Generate samples for each policy
        all_rows = []
        
        for policy, target_count in self.config.POLICY_DISTRIBUTION.items():
            print(f"Generating {target_count} samples for policy: {policy.value}")
            policy_rows = self._generate_policy_samples(policy, target_count)
            all_rows.extend(policy_rows)
            print(f"  Generated {len(policy_rows)} valid samples for {policy.value}")
            print()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_rows)
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        print(f"Dataset generation complete!")
        print(f"Total samples: {len(df)}")
        print(f"Total generation attempts: {self.generation_stats['total_attempts']}")
        print(f"Regenerated rows: {self.generation_stats['regenerated_rows']}")
        
        return df
    
    def _generate_policy_samples(
        self,
        policy: InterviewPolicy,
        target_count: int
    ) -> List[Dict[str, any]]:
        """
        Generate samples for a specific policy.
        
        Args:
            policy: The policy to generate samples for
            target_count: Number of samples to generate
            
        Returns:
            List of valid row dictionaries
        """
        rule = self.rule_engine.get_rule_for_policy(policy)
        rows = []
        attempts = 0
        max_attempts = target_count * self.config.MAX_REGENERATION_ATTEMPTS
        
        with tqdm(total=target_count, desc=f"  Generating {policy.value}") as pbar:
            while len(rows) < target_count and attempts < max_attempts:
                attempts += 1
                self.generation_stats['total_attempts'] += 1
                
                # Generate a row using the rule's distributions
                row = self._generate_row_from_rule(rule)
                
                # Validate the row
                validation_result = self.validator.validate_row(row, policy.value)
                
                if validation_result.is_valid:
                    rows.append(row)
                    self.generation_stats['successful_generations'] += 1
                    self.generation_stats['policy_counts'][policy.value] += 1
                    pbar.update(1)
                else:
                    self.generation_stats['regenerated_rows'] += 1
                    # Track rejected row with details
                    self.rejected_rows.append({
                        'row': row.copy(),
                        'validation_rule': validation_result.rejection_reason,
                        'rule_id': rule.rule_id,
                        'policy': policy.value
                    })
                    # Row will be regenerated in next iteration
        
        if len(rows) < target_count:
            print(f"  Warning: Only generated {len(rows)} of {target_count} samples for {policy.value}")
            print(f"  Total attempts: {attempts}")
        
        return rows
    
    def _generate_row_from_rule(self, rule: PolicyRule) -> Dict[str, any]:
        """
        Generate a single row using a policy rule's distributions.
        
        Args:
            rule: The policy rule to use for generation
            
        Returns:
            Dictionary containing all feature values
        """
        row = {}
        
        # Generate semantic features using the rule's distributions
        correctness_dist = rule.distributions.get('Correctness Score')
        coverage_dist = rule.distributions.get('Concept Coverage')
        reasoning_dist = rule.distributions.get('Reasoning Score')
        missing_dist = rule.distributions.get('Missing Concepts')
        
        # Generate semantic features with correlations
        semantic_features = self.semantic_generator.generate_single_sample(
            correctness_mean=correctness_dist.mean if correctness_dist else 50,
            correctness_std=correctness_dist.std if correctness_dist else 15,
            coverage_mean=coverage_dist.mean if coverage_dist else 50,
            coverage_std=coverage_dist.std if coverage_dist else 15,
            reasoning_mean=reasoning_dist.mean if reasoning_dist else 50,
            reasoning_std=reasoning_dist.std if reasoning_dist else 15,
            missing_mean=missing_dist.mean if missing_dist else 3,
            missing_std=missing_dist.std if missing_dist else 2
        )
        row.update(semantic_features)
        
        # Calculate Performance Index
        performance_index = self.semantic_generator.calculate_performance_index(
            row['Correctness Score'],
            row['Concept Coverage'],
            row['Reasoning Score']
        )
        
        # Generate behavioral features based on Performance Index
        behavioral_features = self.behavioral_generator.generate_single_sample(
            performance_index=performance_index,
            engagement_mean=rule.distributions.get('Engagement Score').mean if rule.distributions.get('Engagement Score') else None,
            engagement_std=rule.distributions.get('Engagement Score').std if rule.distributions.get('Engagement Score') else None,
            confidence_mean=rule.distributions.get('Confidence Score').mean if rule.distributions.get('Confidence Score') else None,
            confidence_std=rule.distributions.get('Confidence Score').std if rule.distributions.get('Confidence Score') else None,
            hesitation_mean=rule.distributions.get('Hesitation Score').mean if rule.distributions.get('Hesitation Score') else None,
            hesitation_std=rule.distributions.get('Hesitation Score').std if rule.distributions.get('Hesitation Score') else None,
            eye_contact_mean=rule.distributions.get('Eye Contact Score').mean if rule.distributions.get('Eye Contact Score') else None,
            eye_contact_std=rule.distributions.get('Eye Contact Score').std if rule.distributions.get('Eye Contact Score') else None
        )
        row.update(behavioral_features)
        
        # Generate context features
        difficulty_value = rule.distributions.get('Difficulty')
        if isinstance(difficulty_value, list):
            difficulty = self.context_generator.generate_difficulty(
                allowed_difficulties=difficulty_value
            )
        else:
            difficulty = difficulty_value
        
        correct_streak_dist = rule.distributions.get('Correct Streak')
        wrong_streak_dist = rule.distributions.get('Wrong Streak')
        
        if isinstance(correct_streak_dist, FeatureDistribution) and isinstance(wrong_streak_dist, FeatureDistribution):
            streaks = self.context_generator.generate_streaks(
                correct_streak_mean=correct_streak_dist.mean,
                correct_streak_std=correct_streak_dist.std,
                wrong_streak_mean=wrong_streak_dist.mean,
                wrong_streak_std=wrong_streak_dist.std,
                force_one_active=False
            )
        elif isinstance(correct_streak_dist, int) or isinstance(wrong_streak_dist, int):
            # Fixed values
            streaks = {
                'Correct Streak': correct_streak_dist if isinstance(correct_streak_dist, int) else 0,
                'Wrong Streak': wrong_streak_dist if isinstance(wrong_streak_dist, int) else 0
            }
        else:
            # Default generation
            streaks = self.context_generator.generate_streaks(force_one_active=False)
        
        row['Difficulty'] = difficulty
        row['Correct Streak'] = streaks['Correct Streak']
        row['Wrong Streak'] = streaks['Wrong Streak']
        
        # Add the policy label
        row['Policy'] = rule.policy.value
        
        # Add Rule_ID for internal tracking (will be removed before export)
        row['Rule_ID'] = f"Rule_{rule.rule_id:02d}"
        
        return row
    
    def save_dataset(self, df: pd.DataFrame, output_path: str, remove_rule_id: bool = True):
        """
        Save the generated dataset to a CSV file.
        
        Args:
            df: DataFrame to save
            output_path: Path to save the CSV file
            remove_rule_id: Whether to remove Rule_ID column before saving
        """
        # Remove Rule_ID if present and requested
        if remove_rule_id and 'Rule_ID' in df.columns:
            df = df.drop('Rule_ID', axis=1)
        
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to: {output_path}")
    
    def save_rejected_rows(self, output_path: str):
        """
        Save rejected rows with rejection reasons to a CSV file.
        
        Args:
            output_path: Path to save the rejected rows CSV file
        """
        if not self.rejected_rows:
            print("No rejected rows to save.")
            return
        
        # Convert to DataFrame
        rejected_df = pd.DataFrame(self.rejected_rows)
        
        # Flatten the nested 'row' dictionary
        row_data = pd.DataFrame(rejected_df['row'].tolist())
        rejected_df = pd.concat([rejected_df.drop('row', axis=1), row_data], axis=1)
        
        # Reorder columns for readability
        cols = ['rule_id', 'policy', 'validation_rule'] + [col for col in rejected_df.columns if col not in ['rule_id', 'policy', 'validation_rule']]
        rejected_df = rejected_df[cols]
        
        rejected_df.to_csv(output_path, index=False)
        print(f"Rejected rows saved to: {output_path} ({len(rejected_df)} rows)")
    
    def reset_rejected_rows(self):
        """Reset rejected rows tracking."""
        self.rejected_rows = []
    
    def get_generation_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the generation process.
        
        Returns:
            Dictionary containing generation statistics
        """
        return {
            'total_attempts': self.generation_stats['total_attempts'],
            'successful_generations': self.generation_stats['successful_generations'],
            'regenerated_rows': self.generation_stats['regenerated_rows'],
            'success_rate': self.generation_stats['successful_generations'] / self.generation_stats['total_attempts'] if self.generation_stats['total_attempts'] > 0 else 0,
            'policy_counts': self.generation_stats['policy_counts'],
            'validation_rejections': self.validator.get_rejection_statistics()
        }
    
    def validate_final_dataset(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate the final dataset for integrity and consistency.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check total count
        if len(df) != self.config.TOTAL_SAMPLES:
            issues.append(f"Dataset has {len(df)} samples, expected {self.config.TOTAL_SAMPLES}")
        
        # Check for missing values
        if df.isnull().any().any():
            missing_cols = df.columns[df.isnull().any()].tolist()
            issues.append(f"Missing values in columns: {missing_cols}")
        
        # Check policy distribution
        policy_counts = df['Policy'].value_counts().to_dict()
        for policy, expected_count in self.config.POLICY_DISTRIBUTION.items():
            actual_count = policy_counts.get(policy.value, 0)
            if actual_count != expected_count:
                issues.append(f"Policy {policy.value}: expected {expected_count}, got {actual_count}")
        
        # Validate each row
        invalid_rows = 0
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            validation_result = self.validator.validate_row(row_dict, row_dict['Policy'])
            if not validation_result.is_valid:
                invalid_rows += 1
                issues.append(f"Invalid row: {validation_result.rejection_reason}")
        
        if invalid_rows > 0:
            issues.append(f"Found {invalid_rows} invalid rows in final dataset")
        
        is_valid = len(issues) == 0
        return is_valid, issues
