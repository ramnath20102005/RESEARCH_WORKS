"""
Explainability module for Interview Training Pipeline.

This module generates SHAP-based feature importance and explanations for trained models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path
import logging

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from configs.config import Config


class ExplainabilityAnalyzer:
    """Generates feature importance and explanations for models."""
    
    def __init__(self, config: Config):
        """
        Initialize the explainability analyzer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP is not installed. Install with: pip install shap")
    
    def generate_feature_importance(
        self,
        model,
        X_train,
        feature_names: List[str],
        model_name: str,
        output_dir: Path
    ):
        """
        Generate feature importance for a model.
        
        Args:
            model: Trained model
            X_train: Training features
            feature_names: List of feature names
            model_name: Name of the model
            output_dir: Directory to save results
        """
        if not SHAP_AVAILABLE:
            self.logger.warning("Skipping feature importance: SHAP not available")
            return
        
        self.logger.info(f"Generating feature importance for {model_name}")
        
        try:
            # Initialize SHAP explainer
            if hasattr(model, 'feature_importances_'):
                # Tree-based model
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_train)
                
                # Generate summary plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_train, feature_names, show=False)
                plt.title(f'SHAP Summary Plot - {model_name}', fontweight='bold')
                plt.tight_layout()
                plt.savefig(output_dir / f'shap_summary_{model_name}.png',
                           dpi=self.config.output.PLOT_DPI, bbox_inches='tight')
                plt.close()
                
                # Generate bar plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_train, feature_names, plot_type='bar', show=False)
                plt.title(f'SHAP Feature Importance - {model_name}', fontweight='bold')
                plt.tight_layout()
                plt.savefig(output_dir / f'shap_bar_{model_name}.png',
                           dpi=self.config.output.PLOT_DPI, bbox_inches='tight')
                plt.close()
                
                # Extract mean absolute SHAP values
                mean_shap = np.abs(shap_values).mean(axis=0)
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': mean_shap
                }).sort_values('Importance', ascending=False)
                
                # Save to CSV
                feature_importance.to_csv(output_dir / f'feature_importance_{model_name}.csv', index=False)
                
                self.logger.info(f"Feature importance saved for {model_name}")
                
            else:
                self.logger.warning(f"Model {model_name} does not support SHAP TreeExplainer")
                
        except Exception as e:
            self.logger.error(f"Error generating feature importance for {model_name}: {e}")
