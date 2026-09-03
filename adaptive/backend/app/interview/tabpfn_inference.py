"""
TabPFN inference module for the Adaptive Interview System.

Loads the trained TabPFN model and predicts interview policies
from 11-dimensional feature vectors.
"""

import logging
import time
import joblib
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TabPFNInference:
    """
    TabPFN inference engine for adaptive interview policy prediction.
    
    Uses the trained tabpfn_10000.pkl model from the interview_training pipeline.
    """
    
    # Policy labels (VERIFIED from training LabelEncoder - alphabetical order)
    # This is the ACTUAL mapping used during TabPFN training
    VERIFIED_POLICY_MAPPING = {
        0: "Ask Application Question",
        1: "Ask Follow-up Question",
        2: "Increase Difficulty",
        3: "Maintain Difficulty",
        4: "Probe Missing Concept",
        5: "Reduce Difficulty",
        6: "Switch Topic"
    }
    
    # Feature names (VERIFIED from training - exact order)
    FEATURE_NAMES = [
        "Correctness Score",
        "Concept Coverage",
        "Reasoning Score",
        "Missing Concepts",
        "Engagement Score",
        "Confidence Score",
        "Hesitation Score",
        "Eye Contact Score",
        "Difficulty",
        "Correct Streak",
        "Wrong Streak"
    ]
    
    # Feature ranges (VERIFIED from training dataset)
    FEATURE_RANGES = {
        "Correctness Score": (3, 100),      # int64
        "Concept Coverage": (0, 100),       # int64
        "Reasoning Score": (0, 100),       # int64
        "Missing Concepts": (0, 8),        # int64
        "Engagement Score": (0.0, 1.0),    # float64
        "Confidence Score": (0.0, 1.0),    # float64
        "Hesitation Score": (0.0, 1.0),    # float64
        "Eye Contact Score": (0.0, 1.0),    # float64
        "Difficulty": (0, 2),              # encoded: Easy=0, Medium=1, Hard=2
        "Correct Streak": (0, 5),          # int64
        "Wrong Streak": (0, 5)             # int64
    }
    
    # Path to trained model (relative to project root)
    MODEL_PATH = Path(__file__).parent.parent.parent.parent.parent / "interview_training" / "outputs" / "models" / "tabpfn_10000.pkl"
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the TabPFN inference engine.
        
        Args:
            model_path: Optional custom path to the trained model file
        """
        self.model_path = model_path or self.MODEL_PATH
        self.model = None
        self.class_mapping = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained TabPFN model."""
        try:
            import time
            load_start = time.perf_counter()
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"TabPFN model not found at {self.model_path}")
            
            logger.info(f"[TabPFN] Loading model from {self.model_path}")
            
            # Use joblib to load the model (more robust for sklearn-like models)
            model_data = joblib.load(self.model_path)
            
            # Handle different model formats
            if isinstance(model_data, dict):
                # If model contains metadata
                self.model = model_data.get('model')
                # Override with verified mapping
                self.class_mapping = self.VERIFIED_POLICY_MAPPING
            else:
                # If model is loaded directly
                self.model = model_data
                # Use verified class mapping
                self.class_mapping = self.VERIFIED_POLICY_MAPPING
            
            if self.model is None:
                raise ValueError("Model not found in loaded file")
            
            load_time = time.perf_counter() - load_start
            logger.info(f"[PERF][STARTUP] TabPFN model load: {load_time*1000:.0f} ms")
            logger.info(f"[TabPFN] Model loaded successfully")
            logger.info("[ARCH] TabPFN model loaded and cached")
            
            # Log verified policy mapping
            logger.info(f"[TabPFN] VERIFIED Policy Mapping:")
            for class_idx, policy_label in self.class_mapping.items():
                logger.info(f"[TabPFN]   {class_idx} → {policy_label}")
            
        except Exception as e:
            logger.error(f"[TabPFN] Failed to load model: {str(e)}")
            raise Exception(f"Failed to load TabPFN model: {str(e)}")
    
    def predict_policy(
        self,
        feature_vector: list,
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Predict the next interview policy from a feature vector.
        
        Uses single-pass inference: predict_proba() + np.argmax()
        
        Args:
            feature_vector: 11-dimensional feature vector in exact training order
            return_probabilities: Whether to return prediction probabilities
        
        Returns:
            Dictionary containing:
                - predicted_class: Numeric class prediction
                - predicted_policy: String policy label
                - probabilities: Optional probability distribution (if requested)
        """
        try:
            logger.info(f"[PERF] tabpfn_prediction: 0.000s")
            predict_start = time.perf_counter()
            
            # Validate input
            if len(feature_vector) != 11:
                raise ValueError(f"Feature vector must have 11 dimensions, got {len(feature_vector)}")
            
            # Convert to DataFrame with feature names to avoid warning
            features_df = pd.DataFrame([feature_vector], columns=self.FEATURE_NAMES)
            
            # CRITICAL: Print ACTUAL input immediately before model.predict_proba()
            print("\n" + "=" * 70)
            print("[TABPFN ACTUAL INPUT]")
            print("=" * 70)
            print(f"X = {features_df.values}")
            print(f"X type = {type(features_df.values)}")
            print(f"X shape = {features_df.shape}")
            
            print("\nFeature values:")
            for i, value in enumerate(features_df.values[0]):
                print(f"Feature {i+1}: {value}")
            
            print("\nFeature order:")
            for i, name in enumerate(self.FEATURE_NAMES):
                print(f"{i+1}. {name}")
            
            print("\nFeature values with names:")
            for i, (name, value) in enumerate(zip(self.FEATURE_NAMES, features_df.values[0])):
                print(f"{i+1}. {name} = {value}")
            
            print("=" * 70)
            
            # CRITICAL: Verify exactly 11 features
            assert features_df.shape == (1, 11), f"INVALID TABPFN INPUT SHAPE: {features_df.shape}"
            
            # Verify feature order matches DataFrame columns
            assert list(features_df.columns) == self.FEATURE_NAMES, "Feature column order mismatch"
            
            print("[TABPFN] Input validation passed")
            print("")
            
            print("=" * 70)
            print("[TABPFN INFERENCE]")
            print("=" * 70)
            print(f"Model: {self.model_path}")
            print(f"Model type: {type(self.model).__name__}")
            print(f"Number of features: {features_df.shape[1]}")
            print(f"Classes: {self.model.classes_ if hasattr(self.model, 'classes_') else 'N/A'}")
            print("")
            
            # Single-pass inference: predict_proba + argmax
            probabilities = self.model.predict_proba(features_df)[0]
            predicted_class = int(np.argmax(probabilities, axis=0))
            
            print("[TABPFN ACTUAL PREDICTION]")
            print(f"Predicted class = {predicted_class}")
            print(f"Probabilities = {probabilities}")
            
            # Map to policy label using VERIFIED mapping
            predicted_policy = self.class_mapping.get(predicted_class, "Maintain Difficulty")
            
            print("[TABPFN POLICY]")
            print(f"Predicted policy = {predicted_policy}")
            print("=" * 70)
            print("")
            
            predict_time = time.perf_counter() - predict_start
            logger.info(f"[PERF] tabpfn_prediction: {predict_time:.3f}s")
            
            result = {
                'predicted_class': predicted_class,
                'predicted_policy': predicted_policy
            }
            
            # Optionally return probabilities
            if return_probabilities:
                result['probabilities'] = {
                    self.class_mapping.get(i, f"Class_{i}"): float(prob)
                    for i, prob in enumerate(probabilities)
                }
                logger.info(f"[TabPFN] Prediction probabilities: {result['probabilities']}")
            
            logger.info(f"[TabPFN] Prediction: {predicted_policy} (class {predicted_class})")
            
            return result
            
        except Exception as e:
            logger.error(f"[TabPFN] Prediction failed: {str(e)}")
            raise Exception(f"TabPFN prediction failed: {str(e)}")
    
    def get_policy_mapping(self) -> dict:
        """
        Return the verified class-to-policy mapping.
        
        Returns:
            Dictionary mapping class IDs to policy names
        """
        return self.class_mapping
    
    def predict_batch(
        self,
        feature_vectors: list,
        return_probabilities: bool = False
    ) -> list:
        """
        Predict policies for multiple feature vectors.
        
        Args:
            feature_vectors: List of 11-dimensional feature vectors
            return_probabilities: Whether to return prediction probabilities
        
        Returns:
            List of prediction dictionaries
        """
        try:
            # Convert to numpy array
            features = np.array(feature_vectors)
            
            if features.shape[1] != 11:
                raise ValueError(f"Feature vectors must have 11 dimensions, got {features.shape[1]}")
            
            logger.info(f"[TabPFN] Predicting batch of {len(feature_vectors)} samples")
            
            # Make predictions
            predicted_classes = self.model.predict(features)
            
            results = []
            for i, predicted_class in enumerate(predicted_classes):
                predicted_policy = self.class_mapping.get(
                    predicted_class,
                    self.POLICY_CLASSES[predicted_class] if predicted_class < len(self.POLICY_CLASSES) else "Maintain Difficulty"
                )
                
                result = {
                    'predicted_class': int(predicted_class),
                    'predicted_policy': predicted_policy
                }
                
                if return_probabilities and hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(features[i:i+1])[0]
                    result['probabilities'] = {
                        self.class_mapping.get(j, f"Class_{j}"): float(prob)
                        for j, prob in enumerate(probabilities)
                    }
                
                results.append(result)
            
            logger.info(f"[TabPFN] Batch prediction complete")
            
            return results
            
        except Exception as e:
            logger.error(f"[TabPFN] Batch prediction failed: {str(e)}")
            raise Exception(f"TabPFN batch prediction failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_path': str(self.model_path),
            'model_type': type(self.model).__name__,
            'class_mapping': self.class_mapping,
            'n_classes': len(self.VERIFIED_POLICY_MAPPING),
            'feature_ranges': self.FEATURE_RANGES
        }
    
    def validate_feature_vector(self, feature_vector: list) -> Tuple[bool, str]:
        """
        Validate a feature vector before prediction using VERIFIED training ranges.
        
        Args:
            feature_vector: The feature vector to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(feature_vector) != 11:
            return False, f"Feature vector must have 11 dimensions, got {len(feature_vector)}"
        
        # Check for NaN or infinite values
        if any(np.isnan(feature_vector)):
            return False, "Feature vector contains NaN values"
        
        if any(np.isinf(feature_vector)):
            return False, "Feature vector contains infinite values"
        
        # Validate ranges using VERIFIED training dataset ranges
        feature_names = [
            "Correctness Score", "Concept Coverage", "Reasoning Score",
            "Missing Concepts", "Engagement Score", "Confidence Score",
            "Hesitation Score", "Eye Contact Score", "Difficulty",
            "Correct Streak", "Wrong Streak"
        ]
        
        for i, (name, value) in enumerate(zip(feature_names, feature_vector)):
            min_val, max_val = self.FEATURE_RANGES[name]
            if not (min_val <= value <= max_val):
                return False, f"Invalid {name}: {value} (expected range: {min_val}-{max_val})"
        
        return True, "Valid"
