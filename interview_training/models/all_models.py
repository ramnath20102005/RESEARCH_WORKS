"""
Model implementations for Interview Training Pipeline.

This module implements the four baseline models and TabPFN for comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import logging
import time
import os
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cat
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

from configs.config import Config


class BaseModel:
    """Base class for all models."""
    
    def __init__(self, model_name: str, config: Config):
        """
        Initialize the base model.
        
        Args:
            model_name: Name of the model
            config: Configuration object
        """
        self.model_name = model_name
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.training_time = None
    
    def train(self, X_train, y_train):
        """Train the model."""
        raise NotImplementedError
    
    def predict(self, X):
        """Make predictions."""
        raise NotImplementedError
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            return None
    
    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        """Evaluate the model."""
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted')
        }
        
        return metrics


class RandomForestModel(BaseModel):
    """Random Forest classifier."""
    
    def __init__(self, config: Config):
        super().__init__("random_forest", config)
    
    def train(self, X_train, y_train):
        """Train Random Forest classifier."""
        self.logger.info("Training Random Forest classifier")
        
        start_time = time.time()
        
        self.model = RandomForestClassifier(
            n_estimators=self.config.model.RF_N_ESTIMATORS,
            max_depth=self.config.model.RF_MAX_DEPTH,
            min_samples_split=self.config.model.RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=self.config.model.RF_MIN_SAMPLES_LEAF,
            random_state=self.config.model.RF_RANDOM_STATE,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        self.logger.info(f"Random Forest training completed in {self.training_time:.2f}s")
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)


class XGBoostModel(BaseModel):
    """XGBoost classifier."""
    
    def __init__(self, config: Config):
        super().__init__("xgboost", config)
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
    
    def train(self, X_train, y_train):
        """Train XGBoost classifier."""
        self.logger.info("Training XGBoost classifier")
        
        start_time = time.time()
        
        self.model = xgb.XGBClassifier(
            n_estimators=self.config.model.XGB_N_ESTIMATORS,
            max_depth=self.config.model.XGB_MAX_DEPTH,
            learning_rate=self.config.model.XGB_LEARNING_RATE,
            subsample=self.config.model.XGB_SUBSAMPLE,
            colsample_bytree=self.config.model.XGB_COLSAMPLE_BYTREE,
            random_state=self.config.model.XGB_RANDOM_STATE,
            eval_metric='mlogloss'
        )
        
        self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        self.logger.info(f"XGBoost training completed in {self.training_time:.2f}s")
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)


class CatBoostModel(BaseModel):
    """CatBoost classifier."""
    
    def __init__(self, config: Config):
        super().__init__("catboost", config)
        
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed. Install with: pip install catboost")
    
    def train(self, X_train, y_train):
        """Train CatBoost classifier."""
        self.logger.info("Training CatBoost classifier")
        
        start_time = time.time()
        
        self.model = cat.CatBoostClassifier(
            iterations=self.config.model.CAT_ITERATIONS,
            depth=self.config.model.CAT_DEPTH,
            learning_rate=self.config.model.CAT_LEARNING_RATE,
            random_state=self.config.model.CAT_RANDOM_STATE,
            verbose=self.config.model.CAT_VERBOSE
        )
        
        self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        self.logger.info(f"CatBoost training completed in {self.training_time:.2f}s")
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)


class TabPFNModel(BaseModel):
    """TabPFN classifier."""
    
    def __init__(self, config: Config):
        super().__init__("tabpfn", config)
        
        if not TABPFN_AVAILABLE:
            raise ImportError("TabPFN is not installed. Install with: pip install tabpfn")
        
        # Import TabPFN-specific config
        from configs.tabpfn_config import tabpfn_config
        self.tabpfn_config = tabpfn_config
        self._optimal_batch_size = None  # Cache for auto-detected batch size
    
    def train(self, X_train, y_train):
        """Train TabPFN classifier."""
        self.logger.info("="*60)
        self.logger.info("INITIALIZING TABPFN")
        self.logger.info("="*60)
        
        # Print GPU information for research documentation
        self.tabpfn_config.print_gpu_info()
        
        # Set up environment for TabPFN authentication (TabPFN 8.2+ uses TABPFN_TOKEN)
        tabpfn_token = os.getenv("TABPFN_TOKEN")
        tabpfn_no_browser = os.getenv("TABPFN_NO_BROWSER", "false").lower() == "true"
        
        if tabpfn_token:
            self.logger.info("Loaded TABPFN_TOKEN successfully")
            self.logger.info("Using non-interactive authentication")
            os.environ["TABPFN_TOKEN"] = tabpfn_token
        else:
            self.logger.warning("No TABPFN_TOKEN found in environment")
            self.logger.warning("TabPFN will attempt interactive authentication")
        
        if tabpfn_no_browser:
            self.logger.info("TABPFN_NO_BROWSER is set - disabling browser-based login")
            os.environ["TABPFN_NO_BROWSER"] = "true"
        
        start_time = time.time()
        
        try:
            self.logger.info("Initializing TabPFNClassifier...")
            self.logger.info(f"Device: {self.tabpfn_config.DEVICE}")
            
            self.model = TabPFNClassifier(
                device=self.tabpfn_config.DEVICE
            )
            
            self.logger.info("TabPFNClassifier initialized successfully")
            self.logger.info("Starting training (fit)...")
            
            self.model.fit(X_train, y_train)
            
            self.training_time = time.time() - start_time
            self.logger.info(f"TabPFN training completed in {self.training_time:.2f}s")
            self.logger.info("="*60)
            self.logger.info("TABPFN TRAINING COMPLETED SUCCESSFULLY")
            self.logger.info("="*60)
            
            # Auto-detect optimal batch size after training (cached)
            if self._optimal_batch_size is None:
                self._optimal_batch_size = self._auto_detect_batch_size(X_train)
                self.logger.info(f"Auto-detected optimal batch size: {self._optimal_batch_size}")
            else:
                self.logger.info(f"Using cached optimal batch size: {self._optimal_batch_size}")
            
        except Exception as e:
            self.logger.error("="*60)
            self.logger.error("TABPFN TRAINING FAILED")
            self.logger.error("="*60)
            self.logger.error(f"Error: {str(e)}")
            
            # Handle license-related errors gracefully
            if "license" in str(e).lower() or "authentication" in str(e).lower():
                self.logger.error("TabPFN license not accepted or authentication failed")
                self.logger.error("Please ensure:")
                self.logger.error("1. TABPFN_TOKEN is set in .env file")
                self.logger.error("2. The token is valid and accepted at https://ux.priorlabs.ai/account/licenses")
                self.logger.error("3. TABPFN_NO_BROWSER is set to 'true' to avoid Windows browser issues")
                raise RuntimeError("TabPFN license/authentication failed. Please follow the instructions above.")
            else:
                self.logger.error("Full traceback:")
                import traceback
                traceback.print_exc()
                raise
    
    def _auto_detect_batch_size(self, X_sample):
        """
        Auto-detect optimal batch size by testing decreasing sizes.
        
        Args:
            X_sample: Sample data to test batch sizes with
            
        Returns:
            Optimal batch size that fits in GPU memory
        """
        import torch
        batch_sizes = [256, 128, 64, 32]
        
        for batch_size in batch_sizes:
            try:
                # Test with a small sample
                test_X = X_sample[:min(batch_size, len(X_sample))]
                _ = self.model.predict_proba(test_X)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.logger.info(f"Batch size {batch_size} test successful")
                return batch_size
                
            except Exception as e:
                if "CUDA out of memory" in str(e).lower():
                    self.logger.warning(f"Batch size {batch_size} caused CUDA OOM, trying smaller size")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    # If it's not a memory error, break and return current batch size
                    self.logger.warning(f"Batch size {batch_size} failed with non-memory error: {e}")
                    break
        
        # Fallback to smallest batch size
        self.logger.warning("All batch sizes failed, using fallback size of 32")
        return 32
    
    def predict(self, X):
        """Make predictions with batching to avoid CUDA OOM."""
        print("Running TabPFN predictions...")
        self.logger.info("Running TabPFN predictions...")
        start_time = time.time()
        
        # Use cached optimal batch size or fallback to config
        batch_size = self._optimal_batch_size if self._optimal_batch_size else self.tabpfn_config.BATCH_SIZE
        n_samples = len(X)
        predictions = []
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        print(f"Batch size: {batch_size}, Total samples: {n_samples}, Batches: {n_batches}")
        self.logger.info(f"Batch size: {batch_size}, Total samples: {n_samples}, Batches: {n_batches}")
        
        try:
            import torch
            for batch_idx, i in enumerate(range(0, n_samples, batch_size)):
                end = min(i + batch_size, n_samples)
                batch_X = X[i:end]
                print(f"Predicting batch {batch_idx + 1}/{n_batches} (samples {i}-{end})")
                self.logger.info(f"Predicting batch {batch_idx + 1}/{n_batches} (samples {i}-{end})")
                batch_pred = self.model.predict(batch_X)
                predictions.append(batch_pred)
                
                # Clear GPU memory between batches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            predictions = np.concatenate(predictions)
            inference_time = time.time() - start_time
            print(f"TabPFN predictions completed in {inference_time:.2f}s (batched)")
            self.logger.info(f"TabPFN predictions completed in {inference_time:.2f}s (batched)")
            return predictions
            
        except Exception as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA OOM with batch_size={batch_size}, retrying with smaller batch")
                self.logger.warning(f"CUDA OOM with batch_size={batch_size}, retrying with smaller batch")
                # Retry with half batch size
                return self._predict_with_retry(X, batch_size // 2)
            else:
                raise
    
    def _predict_with_retry(self, X, batch_size):
        """Retry prediction with smaller batch size."""
        import torch
        n_samples = len(X)
        predictions = []
        
        if batch_size < 1:
            batch_size = 1
        
        self.logger.info(f"Retrying prediction with batch_size={batch_size}")
        
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch_X = X[i:end]
            batch_pred = self.model.predict(batch_X)
            predictions.append(batch_pred)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return np.concatenate(predictions)
    
    def predict_proba(self, X):
        """Get prediction probabilities with batching to avoid CUDA OOM."""
        print("Generating TabPFN prediction probabilities...")
        self.logger.info("Generating TabPFN prediction probabilities...")
        if hasattr(self.model, 'predict_proba'):
            start_time = time.time()
            
            # Use cached optimal batch size or fallback to config
            batch_size = self._optimal_batch_size if self._optimal_batch_size else self.tabpfn_config.BATCH_SIZE
            n_samples = len(X)
            probabilities = []
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            print(f"Batch size: {batch_size}, Total samples: {n_samples}, Batches: {n_batches}")
            self.logger.info(f"Batch size: {batch_size}, Total samples: {n_samples}, Batches: {n_batches}")
            
            try:
                import torch
                for batch_idx, i in enumerate(range(0, n_samples, batch_size)):
                    end = min(i + batch_size, n_samples)
                    batch_X = X[i:end]
                    print(f"Predicting batch {batch_idx + 1}/{n_batches} (samples {i}-{end})")
                    self.logger.info(f"Predicting batch {batch_idx + 1}/{n_batches} (samples {i}-{end})")
                    batch_proba = self.model.predict_proba(batch_X)
                    probabilities.append(batch_proba)
                    
                    # Clear GPU memory between batches
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                probabilities = np.vstack(probabilities)
                inference_time = time.time() - start_time
                print(f"TabPFN prediction probabilities generated in {inference_time:.2f}s (batched)")
                self.logger.info(f"TabPFN prediction probabilities generated in {inference_time:.2f}s (batched)")
                return probabilities
                
            except Exception as e:
                if "CUDA out of memory" in str(e):
                    print(f"CUDA OOM with batch_size={batch_size}, retrying with smaller batch")
                    self.logger.warning(f"CUDA OOM with batch_size={batch_size}, retrying with smaller batch")
                    # Retry with half batch size
                    return self._predict_proba_with_retry(X, batch_size // 2)
                else:
                    raise
        else:
            print("TabPFN does not support predict_proba")
            self.logger.warning("TabPFN does not support predict_proba")
            return None
    
    def _predict_proba_with_retry(self, X, batch_size):
        """Retry predict_proba with smaller batch size."""
        import torch
        n_samples = len(X)
        probabilities = []
        
        if batch_size < 1:
            batch_size = 1
        
        self.logger.info(f"Retrying predict_proba with batch_size={batch_size}")
        
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch_X = X[i:end]
            batch_proba = self.model.predict_proba(batch_X)
            probabilities.append(batch_proba)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return np.vstack(probabilities)


def get_model(model_name: str, config: Config) -> BaseModel:
    """
    Factory function to get a model instance.
    
    Args:
        model_name: Name of the model to instantiate
        config: Configuration object
        
    Returns:
        Model instance
    """
    models = {
        "random_forest": RandomForestModel,
        "xgboost": XGBoostModel,
        "catboost": CatBoostModel,
        "tabpfn": TabPFNModel
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](config)
