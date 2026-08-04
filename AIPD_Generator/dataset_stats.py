"""
Statistics generation module for AIPD-100K Dataset Generator.

This module generates comprehensive statistics about the generated dataset,
including class distributions, feature ranges, correlations, and generation metrics.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional
from pathlib import Path


class DatasetStatistics:
    """Generates and saves comprehensive dataset statistics."""
    
    def __init__(self):
        """Initialize the statistics generator."""
        self.semantic_features = [
            'Correctness Score',
            'Concept Coverage',
            'Reasoning Score',
            'Missing Concepts'
        ]
        
        self.behavioral_features = [
            'Engagement Score',
            'Confidence Score',
            'Hesitation Score',
            'Eye Contact Score'
        ]
        
        self.context_features = [
            'Difficulty',
            'Correct Streak',
            'Wrong Streak'
        ]
        
        self.all_features = self.semantic_features + self.behavioral_features + self.context_features
    
    def generate_statistics(
        self,
        df: pd.DataFrame,
        generation_stats: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive statistics for the dataset.
        
        Args:
            df: DataFrame containing the dataset
            generation_stats: Optional generation statistics from the generator
            
        Returns:
            Dictionary containing all statistics
        """
        stats = {
            'dataset_info': self._generate_dataset_info(df),
            'class_distribution': self._generate_class_distribution(df),
            'feature_statistics': self._generate_feature_statistics(df),
            'correlations': self._generate_correlations(df),
            'policy_specific_stats': self._generate_policy_specific_stats(df),
            'generation_metrics': generation_stats if generation_stats else {}
        }
        
        return stats
    
    def _generate_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic dataset information."""
        return {
            'total_samples': len(df),
            'total_features': len(df.columns) - 1,  # Exclude Policy column
            'num_classes': df['Policy'].nunique(),
            'classes': df['Policy'].unique().tolist(),
            'feature_names': self.all_features,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
    
    def _generate_class_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate class distribution statistics."""
        policy_counts = df['Policy'].value_counts()
        policy_percentages = df['Policy'].value_counts(normalize=True) * 100
        
        distribution = {}
        for policy in policy_counts.index:
            distribution[policy] = {
                'count': int(policy_counts[policy]),
                'percentage': float(policy_percentages[policy])
            }
        
        return distribution
    
    def _generate_feature_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate detailed statistics for each feature."""
        feature_stats = {}
        
        for feature in self.all_features:
            if feature not in df.columns:
                continue
            
            col_data = df[feature]
            
            if col_data.dtype in ['int64', 'float64']:
                # Numerical features
                feature_stats[feature] = {
                    'dtype': str(col_data.dtype),
                    'count': int(col_data.count()),
                    'missing': int(col_data.isnull().sum()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median()),
                    'std': float(col_data.std()),
                    'q25': float(col_data.quantile(0.25)),
                    'q75': float(col_data.quantile(0.75))
                }
            else:
                # Categorical features
                feature_stats[feature] = {
                    'dtype': str(col_data.dtype),
                    'count': int(col_data.count()),
                    'missing': int(col_data.isnull().sum()),
                    'unique': int(col_data.nunique()),
                    'most_common': col_data.mode()[0] if len(col_data.mode()) > 0 else None,
                    'value_counts': col_data.value_counts().to_dict()
                }
        
        return feature_stats
    
    def _generate_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate correlation matrices for different feature groups."""
        correlations = {}
        
        # Semantic feature correlations
        semantic_df = df[self.semantic_features].select_dtypes(include=[np.number])
        if not semantic_df.empty:
            correlations['semantic_features'] = semantic_df.corr().to_dict()
        
        # Behavioral feature correlations
        behavioral_df = df[self.behavioral_features].select_dtypes(include=[np.number])
        if not behavioral_df.empty:
            correlations['behavioral_features'] = behavioral_df.corr().to_dict()
        
        # All numerical features correlation
        numerical_df = df.select_dtypes(include=[np.number])
        if not numerical_df.empty:
            correlations['all_numerical'] = numerical_df.corr().to_dict()
        
        return correlations
    
    def _generate_policy_specific_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistics for each policy class."""
        policy_stats = {}
        
        for policy in df['Policy'].unique():
            policy_df = df[df['Policy'] == policy]
            
            policy_stats[policy] = {
                'count': int(len(policy_df)),
                'percentage': float(len(policy_df) / len(df) * 100),
                'feature_means': {},
                'difficulty_distribution': policy_df['Difficulty'].value_counts().to_dict()
            }
            
            # Calculate mean values for numerical features
            for feature in self.semantic_features + self.behavioral_features:
                if feature in policy_df.columns and policy_df[feature].dtype in ['int64', 'float64']:
                    policy_stats[policy]['feature_means'][feature] = float(policy_df[feature].mean())
        
        return policy_stats
    
    def save_statistics(self, stats: Dict[str, Any], output_path: str):
        """
        Save statistics to a JSON file.
        
        Args:
            stats: Statistics dictionary
            output_path: Path to save the JSON file
        """
        # Convert to JSON-serializable format
        json_stats = self._make_json_serializable(stats)
        
        with open(output_path, 'w') as f:
            json.dump(json_stats, f, indent=2)
        
        print(f"Statistics saved to: {output_path}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert objects to JSON-serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._make_json_serializable(obj.tolist())
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def print_summary(self, stats: Dict[str, Any]):
        """
        Print a summary of the statistics.
        
        Args:
            stats: Statistics dictionary
        """
        print("\n" + "="*60)
        print("DATASET STATISTICS SUMMARY")
        print("="*60)
        
        print("\nDataset Info:")
        print(f"  Total samples: {stats['dataset_info']['total_samples']}")
        print(f"  Total features: {stats['dataset_info']['total_features']}")
        print(f"  Number of classes: {stats['dataset_info']['num_classes']}")
        print(f"  Memory usage: {stats['dataset_info']['memory_usage_mb']:.2f} MB")
        
        print("\nClass Distribution:")
        for policy, dist in stats['class_distribution'].items():
            print(f"  {policy}:")
            print(f"    Count: {dist['count']}")
            print(f"    Percentage: {dist['percentage']:.2f}%")
        
        print("\nFeature Statistics (Semantic):")
        for feature in self.semantic_features:
            if feature in stats['feature_statistics']:
                fs = stats['feature_statistics'][feature]
                print(f"  {feature}:")
                print(f"    Range: [{fs['min']:.2f}, {fs['max']:.2f}]")
                print(f"    Mean: {fs['mean']:.2f}")
                print(f"    Std: {fs['std']:.2f}")
        
        print("\nFeature Statistics (Behavioral):")
        for feature in self.behavioral_features:
            if feature in stats['feature_statistics']:
                fs = stats['feature_statistics'][feature]
                print(f"  {feature}:")
                print(f"    Range: [{fs['min']:.3f}, {fs['max']:.3f}]")
                print(f"    Mean: {fs['mean']:.3f}")
                print(f"    Std: {fs['std']:.3f}")
        
        if stats['generation_metrics']:
            print("\nGeneration Metrics:")
            gm = stats['generation_metrics']
            print(f"  Total attempts: {gm.get('total_attempts', 'N/A')}")
            print(f"  Successful generations: {gm.get('successful_generations', 'N/A')}")
            print(f"  Regenerated rows: {gm.get('regenerated_rows', 'N/A')}")
            print(f"  Success rate: {gm.get('success_rate', 0):.2%}")
            
            if 'validation_rejections' in gm:
                print("\n  Validation Rejections:")
                for reason, count in gm['validation_rejections'].items():
                    if count > 0:
                        print(f"    {reason}: {count}")
        
        print("\n" + "="*60)
