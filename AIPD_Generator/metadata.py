"""
Metadata generation module for AIPD-100K Dataset Generator.

This module generates comprehensive metadata for each dataset generation phase,
including generation parameters, statistics, and provenance information.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class MetadataGenerator:
    """Generates metadata for dataset generation phases."""
    
    def __init__(self, generator_version: str = "1.0.0"):
        """
        Initialize the metadata generator.
        
        Args:
            generator_version: Version of the generator
        """
        self.generator_version = generator_version
    
    def generate_metadata(
        self,
        dataset_name: str,
        dataset_version: str,
        random_seed: int,
        rows_requested: int,
        rows_generated: int,
        rejected_rows: int,
        accepted_rows: int,
        rule_count: int,
        feature_count: int,
        policy_count: int,
        generation_time_seconds: float,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive metadata for a dataset generation phase.
        
        Args:
            dataset_name: Name of the dataset (e.g., "AIPD_100")
            dataset_version: Version of the dataset
            random_seed: Random seed used for generation
            rows_requested: Number of rows requested
            rows_generated: Total rows generated (including rejected)
            rejected_rows: Number of rows rejected during validation
            accepted_rows: Number of rows accepted (final dataset size)
            rule_count: Number of policy rules used
            feature_count: Number of features
            policy_count: Number of policy classes
            generation_time_seconds: Time taken for generation
            additional_info: Optional additional information
            
        Returns:
            Dictionary containing all metadata
        """
        metadata = {
            "dataset_info": {
                "name": dataset_name,
                "version": dataset_version,
                "generation_date": datetime.now().isoformat(),
                "generator_version": self.generator_version
            },
            "generation_parameters": {
                "random_seed": random_seed,
                "rows_requested": rows_requested,
                "rows_generated": rows_generated,
                "rejected_rows": rejected_rows,
                "accepted_rows": accepted_rows,
                "success_rate": accepted_rows / rows_generated if rows_generated > 0 else 0,
                "generation_time_seconds": generation_time_seconds
            },
            "dataset_structure": {
                "rule_count": rule_count,
                "feature_count": feature_count,
                "policy_count": policy_count,
                "policies": [
                    "Increase Difficulty",
                    "Maintain Difficulty",
                    "Reduce Difficulty",
                    "Probe Missing Concept",
                    "Ask Application Question",
                    "Ask Follow-up Question",
                    "Switch Topic"
                ]
            },
            "feature_groups": {
                "semantic_features": [
                    "Correctness Score",
                    "Concept Coverage",
                    "Reasoning Score",
                    "Missing Concepts"
                ],
                "behavioral_features": [
                    "Engagement Score",
                    "Confidence Score",
                    "Hesitation Score",
                    "Eye Contact Score"
                ],
                "context_features": [
                    "Difficulty",
                    "Correct Streak",
                    "Wrong Streak"
                ]
            },
            "methodology": {
                "rule_based_generation": True,
                "correlated_feature_generation": True,
                "performance_index_based_behavioral": True,
                "validation_with_regeneration": True,
                "deterministic": True
            }
        }
        
        if additional_info:
            metadata["additional_info"] = additional_info
        
        return metadata
    
    def save_metadata(self, metadata: Dict[str, Any], output_path: str):
        """
        Save metadata to a JSON file.
        
        Args:
            metadata: Metadata dictionary
            output_path: Path to save the JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def print_metadata_summary(self, metadata: Dict[str, Any]):
        """
        Print a summary of the metadata.
        
        Args:
            metadata: Metadata dictionary
        """
        print("\n" + "="*60)
        print("DATASET METADATA")
        print("="*60)
        
        print("\nDataset Info:")
        print(f"  Name: {metadata['dataset_info']['name']}")
        print(f"  Version: {metadata['dataset_info']['version']}")
        print(f"  Generation Date: {metadata['dataset_info']['generation_date']}")
        print(f"  Generator Version: {metadata['dataset_info']['generator_version']}")
        
        print("\nGeneration Parameters:")
        print(f"  Random Seed: {metadata['generation_parameters']['random_seed']}")
        print(f"  Rows Requested: {metadata['generation_parameters']['rows_requested']}")
        print(f"  Rows Generated: {metadata['generation_parameters']['rows_generated']}")
        print(f"  Rejected Rows: {metadata['generation_parameters']['rejected_rows']}")
        print(f"  Accepted Rows: {metadata['generation_parameters']['accepted_rows']}")
        print(f"  Success Rate: {metadata['generation_parameters']['success_rate']:.2%}")
        print(f"  Generation Time: {metadata['generation_parameters']['generation_time_seconds']:.2f}s")
        
        print("\nDataset Structure:")
        print(f"  Rules: {metadata['dataset_structure']['rule_count']}")
        print(f"  Features: {metadata['dataset_structure']['feature_count']}")
        print(f"  Policies: {metadata['dataset_structure']['policy_count']}")
        
        print("\n" + "="*60)
