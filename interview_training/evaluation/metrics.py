"""
Evaluation module for Interview Training Pipeline.

This module implements evaluation metrics and visualization for model performance.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from pathlib import Path
import logging

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from configs.config import Config


class ModelEvaluator:
    """Evaluates model performance and generates evaluation reports."""
    
    def __init__(self, config: Config):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 8)
    
    def evaluate_model(
        self,
        model,
        X_test,
        y_test,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info(f"Evaluating {model_name}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
        
        self.logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1-Macro: {metrics['f1_macro']:.4f}")
        
        return metrics
    
    def evaluate_model_with_predictions(
        self,
        y_pred,
        X_test,
        y_test,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate a model using pre-computed predictions.
        
        Args:
            y_pred: Pre-computed predictions
            X_test: Test features (not used but kept for interface consistency)
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info(f"Evaluating {model_name} with pre-computed predictions")
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
        
        self.logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1-Macro: {metrics['f1_macro']:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        model_name: str,
        output_dir: Path
    ):
        """
        Plot confusion matrix for a model.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            model_name: Name of the model
            output_dir: Directory to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax)
        
        ax.set_title(f'Confusion Matrix - {model_name}', fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'confusion_matrix_{model_name}.png',
                   dpi=self.config.output.PLOT_DPI, bbox_inches='tight')
        plt.close()
    
    def compare_models(
        self,
        all_metrics: List[Dict[str, Any]],
        output_dir: Path
    ) -> pd.DataFrame:
        """
        Compare multiple models and generate comparison table.
        
        Args:
            all_metrics: List of metric dictionaries for each model
            output_dir: Directory to save comparison results
            
        Returns:
            DataFrame containing model comparison
        """
        self.logger.info("Comparing models")
        
        # Extract comparison metrics
        comparison_data = []
        for metrics in all_metrics:
            comparison_data.append({
                'Model': metrics['model_name'],
                'Accuracy': metrics['accuracy'],
                'Precision (Macro)': metrics['precision_macro'],
                'Recall (Macro)': metrics['recall_macro'],
                'F1 (Macro)': metrics['f1_macro'],
                'F1 (Weighted)': metrics['f1_weighted'],
                'Training Time (s)': metrics.get('training_time', 0),
                'Inference Time (s)': metrics.get('inference_time', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by F1-Macro
        comparison_df = comparison_df.sort_values('F1 (Macro)', ascending=False)
        
        # Save to CSV
        comparison_df.to_csv(output_dir / self.config.output.COMPARISON_FILE, index=False)
        
        # Generate comparison plot
        self.plot_model_comparison(comparison_df, output_dir)
        
        self.logger.info(f"Model comparison saved to: {output_dir / self.config.output.COMPARISON_FILE}")
        
        return comparison_df
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, output_dir: Path):
        """
        Plot model comparison bar chart.
        
        Args:
            comparison_df: DataFrame containing model comparison
            output_dir: Directory to save the plot
        """
        metrics_to_plot = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            comparison_df.plot(x='Model', y=metric, kind='bar', ax=axes[idx], color=sns.color_palette()[idx])
            axes[idx].set_title(metric, fontweight='bold')
            axes[idx].set_xlabel('Model', fontweight='bold')
            axes[idx].set_ylabel('Score', fontweight='bold')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].legend().remove()
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison.png', 
                   dpi=self.config.output.PLOT_DPI, bbox_inches='tight')
        plt.close()
    
    def save_all_metrics(self, all_metrics: List[Dict[str, Any]], output_dir: Path):
        """
        Save all model metrics to a CSV file.
        
        Args:
            all_metrics: List of metric dictionaries for each model
            output_dir: Directory to save metrics
        """
        metrics_data = []
        
        for metrics in all_metrics:
            # Flatten classification report
            class_report = metrics['classification_report']
            for class_name, class_metrics in class_report.items():
                if isinstance(class_metrics, dict):
                    metrics_data.append({
                        'Model': metrics['model_name'],
                        'Class': class_name,
                        'Precision': class_metrics.get('precision', 0),
                        'Recall': class_metrics.get('recall', 0),
                        'F1-Score': class_metrics.get('f1-score', 0),
                        'Support': class_metrics.get('support', 0)
                    })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(output_dir / self.config.output.METRICS_FILE, index=False)
        
        self.logger.info(f"All metrics saved to: {output_dir / self.config.output.METRICS_FILE}")
