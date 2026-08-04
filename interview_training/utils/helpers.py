"""
Utility modules for Interview Training Pipeline.

This module provides logging and helper functions for the training pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str,
    log_file: str = "training.log",
    log_level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    console_logging: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file
        log_level: Logging level
        log_format: Log message format
        console_logging: Whether to enable console logging
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_timestamp() -> str:
    """Get current timestamp as a formatted string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_directory(directory: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_model(model, model_name: str, output_dir: str):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model object
        model_name: Name for the model file
        output_dir: Directory to save the model
    """
    import joblib
    
    output_path = Path(output_dir) / f"{model_name}.pkl"
    ensure_directory(output_dir)
    
    joblib.dump(model, output_path)
    print(f"Model saved to: {output_path}")


def load_model(model_name: str, output_dir: str):
    """
    Load a trained model from disk.
    
    Args:
        model_name: Name of the model file
        output_dir: Directory containing the model
        
    Returns:
        Loaded model object
    """
    import joblib
    
    model_path = Path(output_dir) / f"{model_name}.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return joblib.load(model_path)


def calculate_training_time(start_time, end_time) -> float:
    """
    Calculate training time in seconds.
    
    Args:
        start_time: Start timestamp
        end_time: End timestamp
        
    Returns:
        Training time in seconds
    """
    return end_time - start_time


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"
