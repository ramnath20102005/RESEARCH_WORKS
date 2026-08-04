"""
TabPFN-specific configuration for Interview Training Pipeline.

This module contains configuration parameters specific to TabPFN model training
and evaluation, including GPU detection, license handling, and model parameters.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class TabPFNConfig:
    """TabPFN model configuration."""
    
    # Device configuration
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Random seed for reproducibility
    RANDOM_SEED: int = 42
    
    # Model parameters
    N_ENSEMBLE_CONFIGURATIONS: int = 32
    NO_PREPROCESS_CPU: bool = True
    
    # Inference configuration
    INFERENCE_MODE: str = "classification"  # or "regression"
    
    # Batch size for prediction (to avoid CUDA OOM)
    # Will be auto-adjusted based on GPU memory availability
    BATCH_SIZE: int = 256  # Starting point for auto-detection
    
    # Training sizes for scalability experiment
    # Evaluates TabPFN performance across different training set sizes
    # due to computational constraints of foundation models
    TRAIN_SIZES: list = field(default_factory=lambda: [5000, 10000, 20000])
    
    # License handling
    SKIP_LICENSE_CHECK: bool = False
    LICENSE_ACCEPTED: bool = False
    
    # Performance tracking
    TRACK_TRAINING_TIME: bool = True
    TRACK_INFERENCE_TIME: bool = True
    TRACK_MEMORY_USAGE: bool = True
    
    @staticmethod
    def print_gpu_info():
        """Print GPU information for research documentation."""
        print("="*60)
        print("GPU Information")
        print("="*60)
        print(f"GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"PyTorch Version: {torch.__version__}")
            print(f"Device Used: {torch.cuda.get_device_name(0)}")
        else:
            print("Device Used: CPU")
        print("="*60)


# Create global instance
tabpfn_config = TabPFNConfig()
