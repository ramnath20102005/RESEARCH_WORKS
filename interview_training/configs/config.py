"""
Configuration module for Interview Training Pipeline.

This module contains all configurable parameters for the training pipeline,
including data paths, model hyperparameters, and evaluation settings.
"""

import dataclasses
from typing import Dict, List, Tuple, Optional
from pathlib import Path


@dataclasses.dataclass
class DataConfig:
    """Data loading and processing configuration."""
    
    # Dataset path (relative to project root)
    DATASET_PATH: str = "../AIPD_Generator/output/AIPD_100000/AIPD_100000.csv"
    
    # Feature columns
    SEMANTIC_FEATURES: List[str] = dataclasses.field(default_factory=lambda: [
        "Correctness Score",
        "Concept Coverage", 
        "Reasoning Score",
        "Missing Concepts"
    ])
    
    BEHAVIORAL_FEATURES: List[str] = dataclasses.field(default_factory=lambda: [
        "Engagement Score",
        "Confidence Score",
        "Hesitation Score",
        "Eye Contact Score"
    ])
    
    CONTEXT_FEATURES: List[str] = dataclasses.field(default_factory=lambda: [
        "Difficulty",
        "Correct Streak",
        "Wrong Streak"
    ])
    
    # Target column
    TARGET_COLUMN: str = "Policy"
    
    # All feature columns
    @property
    def FEATURE_COLUMNS(self) -> List[str]:
        return self.SEMANTIC_FEATURES + self.BEHAVIORAL_FEATURES + self.CONTEXT_FEATURES
    
    # Categorical columns
    CATEGORICAL_COLUMNS: List[str] = dataclasses.field(default_factory=lambda: ["Difficulty"])
    
    # Difficulty encoding
    DIFFICULTY_ENCODING: Dict[str, int] = dataclasses.field(default_factory=lambda: {
        "Easy": 0,
        "Medium": 1,
        "Hard": 2
    })
    
    # Policy classes
    POLICY_CLASSES: List[str] = dataclasses.field(default_factory=lambda: [
        "Increase Difficulty",
        "Maintain Difficulty", 
        "Reduce Difficulty",
        "Probe Missing Concept",
        "Ask Application Question",
        "Ask Follow-up Question",
        "Switch Topic"
    ])


@dataclasses.dataclass
class SplitConfig:
    """Train/validation/test split configuration."""
    
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    RANDOM_STATE: int = 42
    STRATIFY: bool = True  # Preserve class distribution


@dataclasses.dataclass
class ModelConfig:
    """Model training configuration."""
    
    # Models to train
    MODELS_TO_TRAIN: List[str] = dataclasses.field(default_factory=lambda: [
        "random_forest",
        "xgboost", 
        "catboost",
        "tabpfn"  # TabPFN as primary research model
    ])
    
    # Random Forest hyperparameters
    RF_N_ESTIMATORS: int = 100
    RF_MAX_DEPTH: Optional[int] = None
    RF_MIN_SAMPLES_SPLIT: int = 2
    RF_MIN_SAMPLES_LEAF: int = 1
    RF_RANDOM_STATE: int = 42
    
    # XGBoost hyperparameters
    XGB_N_ESTIMATORS: int = 100
    XGB_MAX_DEPTH: int = 6
    XGB_LEARNING_RATE: float = 0.1
    XGB_SUBSAMPLE: float = 0.8
    XGB_COLSAMPLE_BYTREE: float = 0.8
    XGB_RANDOM_STATE: int = 42
    
    # CatBoost hyperparameters
    CAT_ITERATIONS: int = 100
    CAT_DEPTH: int = 6
    CAT_LEARNING_RATE: float = 0.1
    CAT_RANDOM_STATE: int = 42
    CAT_VERBOSE: bool = False
    
    # TabPFN hyperparameters
    TABPFN_DEVICE: str = "cuda"  # or "cuda" if available
    TABPFN_OVERWRITE_CALLBACK: bool = False
    
    # General training settings
    CROSS_VALIDATION_FOLDS: int = 5
    EARLY_STOPPING_ROUNDS: int = 10


@dataclasses.dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    
    # Metrics to compute
    METRICS: List[str] = dataclasses.field(default_factory=lambda: [
        "accuracy",
        "precision_macro",
        "recall_macro", 
        "f1_macro",
        "f1_weighted"
    ])
    
    # Confusion matrix settings
    CONFUSION_MATRIX_NORMALIZE: str = "true"  # 'true', 'pred', 'all', or None
    
    # Classification report settings
    CLASSIFICATION_REPORT_OUTPUT_DICT: bool = True


@dataclasses.dataclass
class OutputConfig:
    """Output configuration."""
    
    # Output directories
    OUTPUT_DIR: str = "outputs"
    MODELS_DIR: str = "outputs/models"
    PLOTS_DIR: str = "outputs/plots"
    REPORTS_DIR: str = "outputs/reports"
    METRICS_DIR: str = "outputs/metrics"
    
    # File naming
    MODEL_FILE_PREFIX: str = "model"
    METRICS_FILE: str = "metrics.csv"
    COMPARISON_FILE: str = "model_comparison.csv"
    
    # Plot settings
    PLOT_DPI: int = 300
    PLOT_STYLE: str = "seaborn-v0_8-whitegrid"
    FIGURE_SIZE: Tuple[int, int] = (12, 8)


@dataclasses.dataclass
class ExplainabilityConfig:
    """Explainability configuration."""
    
    # SHAP settings
    USE_SHAP: bool = True
    SHAP_SAMPLE_SIZE: int = 1000  # Number of samples for SHAP analysis
    
    # Feature importance settings
    COMPUTE_FEATURE_IMPORTANCE: bool = True
    FEATURE_IMPORTANCE_METHOD: str = "gain"  # 'gain', 'split', 'weight' for tree models


@dataclasses.dataclass
class LoggingConfig:
    """Logging configuration."""
    
    LOG_FILE: str = "training.log"
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    CONSOLE_LOGGING: bool = True


class Config:
    """Main configuration class."""
    
    def __init__(self):
        """Initialize all configuration sections."""
        self.data = DataConfig()
        self.split = SplitConfig()
        self.model = ModelConfig()
        self.evaluation = EvaluationConfig()
        self.output = OutputConfig()
        self.explainability = ExplainabilityConfig()
        self.logging = LoggingConfig()
    
    def create_output_directories(self):
        """Create all output directories."""
        from pathlib import Path
        
        directories = [
            self.output.OUTPUT_DIR,
            self.output.MODELS_DIR,
            self.output.PLOTS_DIR,
            self.output.REPORTS_DIR,
            self.output.METRICS_DIR
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = Config()
