"""
Preprocessing and encoding module for Interview Training Pipeline.

This module handles feature encoding, train/test splitting, and data preparation
for model training.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

from configs.config import Config


class DataPreprocessor:
    """Preprocesses and encodes the dataset for model training."""
    
    def __init__(self, config: Config):
        """
        Initialize the preprocessor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize encoders
        self.difficulty_encoder = None
        self.policy_encoder = None
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: DataFrame to encode
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        # Encode Difficulty column
        if "Difficulty" in df_encoded.columns:
            self.logger.info("Encoding Difficulty column")
            df_encoded["Difficulty"] = df_encoded["Difficulty"].map(self.config.data.DIFFICULTY_ENCODING)
            self.logger.info(f"Difficulty encoding: {self.config.data.DIFFICULTY_ENCODING}")
        
        return df_encoded
    
    def encode_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
        """
        Encode the target variable (Policy).
        
        Args:
            df: DataFrame with target column
            
        Returns:
            Tuple of (DataFrame with encoded target, fitted LabelEncoder)
        """
        df_encoded = df.copy()
        
        if self.config.data.TARGET_COLUMN not in df_encoded.columns:
            raise ValueError(f"Target column '{self.config.data.TARGET_COLUMN}' not found in dataset")
        
        self.logger.info(f"Encoding target column: {self.config.data.TARGET_COLUMN}")
        
        # Initialize and fit LabelEncoder
        self.policy_encoder = LabelEncoder()
        df_encoded[self.config.data.TARGET_COLUMN] = self.policy_encoder.fit_transform(
            df_encoded[self.config.data.TARGET_COLUMN]
        )
        
        # Store the class mapping
        self.class_mapping = {
            i: class_name 
            for i, class_name in enumerate(self.policy_encoder.classes_)
        }
        
        self.logger.info(f"Class mapping: {self.class_mapping}")
        
        return df_encoded, self.policy_encoder
    
    def split_data(
        self,
        df: pd.DataFrame,
        features: List[str],
        target: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split the dataset into train, validation, and test sets.
        
        Args:
            df: DataFrame to split
            features: List of feature column names
            target: Target column name
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        X = df[features]
        y = df[target]
        
        self.logger.info(f"Splitting data: Train={self.config.split.TRAIN_RATIO:.0%}, "
                        f"Val={self.config.split.VAL_RATIO:.0%}, Test={self.config.split.TEST_RATIO:.0%}")
        
        # First split: train + val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=self.config.split.TEST_RATIO,
            random_state=self.config.split.RANDOM_STATE,
            stratify=y if self.config.split.STRATIFY else None
        )
        
        # Second split: train vs val
        val_ratio_adjusted = self.config.split.VAL_RATIO / (1 - self.config.split.TEST_RATIO)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_ratio_adjusted,
            random_state=self.config.split.RANDOM_STATE,
            stratify=y_train_val if self.config.split.STRATIFY else None
        )
        
        self.logger.info(f"Train set: {X_train.shape[0]:,} samples")
        self.logger.info(f"Validation set: {X_val.shape[0]:,} samples")
        self.logger.info(f"Test set: {X_test.shape[0]:,} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, LabelEncoder]:
        """
        Complete data preparation pipeline.
        
        Args:
            df: Raw dataset DataFrame
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, policy_encoder)
        """
        self.logger.info("Starting data preparation pipeline")
        
        # Step 1: Encode categorical features
        df_encoded = self.encode_categorical_features(df)
        
        # Step 2: Encode target
        df_encoded, policy_encoder = self.encode_target(df_encoded)
        
        # Step 3: Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            df_encoded,
            self.config.data.FEATURE_COLUMNS,
            self.config.data.TARGET_COLUMN
        )
        
        self.logger.info("Data preparation completed")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, policy_encoder
    
    def get_feature_info(self, X_train: pd.DataFrame) -> Dict[str, any]:
        """
        Get information about the prepared features.
        
        Args:
            X_train: Training features DataFrame
            
        Returns:
            Dictionary containing feature information
        """
        return {
            'n_features': X_train.shape[1],
            'feature_names': X_train.columns.tolist(),
            'feature_types': X_train.dtypes.to_dict(),
            'feature_ranges': {
                col: {
                    'min': X_train[col].min(),
                    'max': X_train[col].max(),
                    'mean': X_train[col].mean(),
                    'std': X_train[col].std()
                }
                for col in X_train.columns
            }
        }
