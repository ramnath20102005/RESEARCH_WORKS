"""
Permutation importance analysis for TabPFN explainability.

This module implements permutation importance as a model-agnostic explanation method
for TabPFN since SHAP support may be limited or unreliable for TabPFN.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import logging
import time

from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score

from configs.config import Config


class PermutationImportanceAnalyzer:
    """
    Permutation importance analyzer for model explainability.
    
    We use permutation importance instead of SHAP for TabPFN because:
    1. SHAP support for TabPFN is not well-established in current versions
    2. Permutation importance is model-agnostic and works with any classifier
    3. It provides intuitive feature importance scores
    4. It's computationally efficient and reproducible
    """
    
    def __init__(self, config: Config):
        """
        Initialize the permutation importance analyzer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def compute_permutation_importance(
        self,
        model,
        X_test,
        y_test,
        feature_names: List[str],
        n_repeats: int = 10,
        random_state: int = 42
    ) -> Dict[str, any]:
        """
        Compute permutation importance for a model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
            n_repeats: Number of permutation repeats
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing permutation importance results
        """
        self.logger.info(f"Computing permutation importance with {n_repeats} repeats")
        
        start_time = time.time()
        
        try:
            # Compute permutation importance
            result = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=n_repeats,
                random_state=random_state,
                scoring='f1_macro'  # Use F1 macro for multi-class
            )
            
            # Extract importance results
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance_mean': result.importances_mean,
                'importance_std': result.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            computation_time = time.time() - start_time
            
            self.logger.info(f"Permutation importance computed in {computation_time:.2f}s")
            
            return {
                'importance_df': importance_df,
                'importances': result.importances,
                'computation_time': computation_time,
                'n_repeats': n_repeats
            }
            
        except Exception as e:
            self.logger.error(f"Error computing permutation importance: {e}")
            return None
    
    def save_importance_results(
        self,
        importance_results: Dict[str, any],
        model_name: str,
        output_dir: Path
    ):
        """
        Save permutation importance results.
        
        Args:
            importance_results: Dictionary containing importance results
            model_name: Name of the model
            output_dir: Directory to save results
        """
        if importance_results is None:
            self.logger.warning("No importance results to save")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_path = output_path / f'feature_importance_{model_name}.csv'
        importance_results['importance_df'].to_csv(csv_path, index=False)
        
        self.logger.info(f"Feature importance saved to: {csv_path}")
    
    def plot_importance(
        self,
        importance_results: Dict[str, any],
        model_name: str,
        output_dir: Path
    ):
        """
        Plot feature importance.
        
        Args:
            importance_results: Dictionary containing importance results
            model_name: Name of the model
            output_dir: Directory to save plot
        """
        if importance_results is None:
            self.logger.warning("No importance results to plot")
            return
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        importance_df = importance_results['importance_df']
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bar chart with error bars
        bars = ax.barh(
            importance_df['feature'],
            importance_df['importance_mean'],
            xerr=importance_df['importance_std'],
            color=sns.color_palette()[0],
            alpha=0.7
        )
        
        ax.set_xlabel('Importance (F1 Macro Decrease)', fontweight='bold')
        ax.set_ylabel('Feature', fontweight='bold')
        ax.set_title(f'Permutation Importance - {model_name}', fontweight='bold')
        ax.invert_yaxis()  # Highest importance at top
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_path / f'feature_importance_{model_name}.png',
                   dpi=self.config.output.PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Feature importance plot saved to: {output_path / f'feature_importance_{model_name}.png'}")
