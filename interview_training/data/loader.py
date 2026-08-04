"""
Data loader and validation module for Interview Training Pipeline.

This module handles loading the AIPD-100K dataset and validating its structure
before any preprocessing or model training.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

from configs.config import Config


class DataLoader:
    """Loads and validates the AIPD-100K dataset."""
    
    def __init__(self, config: Config):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_dataset(self) -> pd.DataFrame:
        """
        Load the AIPD-100K dataset from the specified path.
        
        Returns:
            DataFrame containing the dataset
            
        Raises:
            FileNotFoundError: If dataset file does not exist
            ValueError: If dataset has invalid structure
        """
        dataset_path = Path(self.config.data.DATASET_PATH)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
        
        self.logger.info(f"Loading dataset from: {dataset_path}")
        
        try:
            df = pd.read_csv(dataset_path)
            self.logger.info(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate the dataset structure and content.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'shape': df.shape,
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        }
        
        # Check 1: Verify shape
        if df.shape[0] == 0:
            validation_results['is_valid'] = False
            validation_results['issues'].append("Dataset is empty")
        
        # Check 2: Verify required columns exist
        required_columns = self.config.data.FEATURE_COLUMNS + [self.config.data.TARGET_COLUMN]
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_columns}")
        
        # Check 3: Verify no missing values in critical columns
        if df[self.config.data.FEATURE_COLUMNS].isnull().any().any():
            missing_cols = df[self.config.data.FEATURE_COLUMNS].columns[df[self.config.data.FEATURE_COLUMNS].isnull().any()].tolist()
            validation_results['issues'].append(f"Missing values in feature columns: {missing_cols}")
        
        # Check 4: Verify target column has no missing values
        if df[self.config.data.TARGET_COLUMN].isnull().any():
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing values in target column: {self.config.data.TARGET_COLUMN}")
        
        # Check 5: Verify target column contains all expected classes
        unique_classes = df[self.config.data.TARGET_COLUMN].unique()
        expected_classes = set(self.config.data.POLICY_CLASSES)
        actual_classes = set(unique_classes)
        
        if not expected_classes.issubset(actual_classes):
            missing_classes = expected_classes - actual_classes
            validation_results['issues'].append(f"Missing expected classes: {missing_classes}")
        
        # Check 6: Verify data types
        dtype_issues = []
        for col in self.config.data.SEMANTIC_FEATURES + self.config.data.BEHAVIORAL_FEATURES:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                dtype_issues.append(f"{col} should be numeric")
        
        for col in self.config.data.CONTEXT_FEATURES:
            if col in df.columns:
                if col == "Difficulty" and not pd.api.types.is_string_dtype(df[col]):
                    dtype_issues.append(f"{col} should be string/object")
                elif col in ["Correct Streak", "Wrong Streak"] and not pd.api.types.is_integer_dtype(df[col]):
                    dtype_issues.append(f"{col} should be integer")
        
        if dtype_issues:
            validation_results['issues'].extend(dtype_issues)
        
        # Check 7: Verify feature ranges
        range_issues = []
        for col in self.config.data.SEMANTIC_FEATURES:
            if col in df.columns:
                if col == "Missing Concepts":
                    if not df[col].between(0, 8).all():
                        range_issues.append(f"{col} values out of range [0, 8]")
                else:
                    if not df[col].between(0, 100).all():
                        range_issues.append(f"{col} values out of range [0, 100]")
        
        for col in self.config.data.BEHAVIORAL_FEATURES:
            if col in df.columns:
                if not df[col].between(0, 1).all():
                    range_issues.append(f"{col} values out of range [0, 1]")
        
        for col in ["Correct Streak", "Wrong Streak"]:
            if col in df.columns:
                if not df[col].between(0, 5).all():
                    range_issues.append(f"{col} values out of range [0, 5]")
        
        if range_issues:
            validation_results['issues'].extend(range_issues)
        
        # Check 8: Verify categorical column values
        if "Difficulty" in df.columns:
            valid_difficulties = set(self.config.data.DIFFICULTY_ENCODING.keys())
            actual_difficulties = set(df["Difficulty"].unique())
            invalid_difficulties = actual_difficulties - valid_difficulties
            if invalid_difficulties:
                validation_results['issues'].append(f"Invalid difficulty values: {invalid_difficulties}")
        
        # Update validation status
        if validation_results['issues']:
            validation_results['is_valid'] = False
        
        return validation_results
    
    def generate_validation_report(self, validation_results: Dict[str, any]) -> str:
        """
        Generate a human-readable validation report.
        
        Args:
            validation_results: Dictionary containing validation results
            
        Returns:
            String containing the validation report
        """
        report = []
        report.append("="*60)
        report.append("DATASET VALIDATION REPORT")
        report.append("="*60)
        report.append("")
        
        # Basic info
        report.append("Dataset Shape:")
        report.append(f"  Rows: {validation_results['shape'][0]:,}")
        report.append(f"  Columns: {validation_results['shape'][1]}")
        report.append(f"  Memory Usage: {validation_results['memory_usage']:.2f} MB")
        report.append("")
        
        # Missing values
        report.append("Missing Values:")
        for col, count in validation_results['missing_values'].items():
            if count > 0:
                report.append(f"  {col}: {count} ({count/validation_results['shape'][0]*100:.2f}%)")
        if sum(validation_results['missing_values'].values()) == 0:
            report.append("  No missing values")
        report.append("")
        
        # Duplicates
        report.append(f"Duplicate Rows: {validation_results['duplicates']:,}")
        report.append("")
        
        # Data types
        report.append("Data Types:")
        for col, dtype in validation_results['dtypes'].items():
            report.append(f"  {col}: {dtype}")
        report.append("")
        
        # Validation status
        if validation_results['is_valid']:
            report.append("✓ VALIDATION PASSED")
        else:
            report.append("✗ VALIDATION FAILED")
            report.append("")
            report.append("Issues:")
            for issue in validation_results['issues']:
                report.append(f"  - {issue}")
        
        report.append("")
        report.append("="*60)
        
        return "\n".join(report)
