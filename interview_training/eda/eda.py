"""
EDA module for Interview Training Pipeline.

This module performs exploratory data analysis and generates visualization
plots for the AIPD-100K dataset.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path
import logging

from configs.config import Config


class EDAAnalyzer:
    """Performs exploratory data analysis on the dataset."""
    
    def __init__(self, config: Config):
        """
        Initialize the EDA analyzer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def generate_eda_plots(self, df: pd.DataFrame, output_dir: str):
        """
        Generate all EDA visualization plots.
        
        Args:
            df: DataFrame to analyze
            output_dir: Directory to save plots
        """
        self.logger.info("Generating EDA plots")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate individual plots
        self.plot_class_distribution(df, output_path)
        self.plot_semantic_features(df, output_path)
        self.plot_behavioral_features(df, output_path)
        self.plot_context_features(df, output_path)
        self.plot_correlation_heatmap(df, output_path)
        self.plot_boxplots(df, output_path)
        self.plot_streak_analysis(df, output_path)
        self.plot_missing_concepts_analysis(df, output_path)
        
        self.logger.info(f"EDA plots saved to: {output_path}")
    
    def plot_class_distribution(self, df: pd.DataFrame, output_dir: Path):
        """Generate class distribution plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        target_col = self.config.data.TARGET_COLUMN
        policy_counts = df[target_col].value_counts().sort_index()
        
        colors = sns.color_palette("husl", len(policy_counts))
        bars = ax.bar(policy_counts.index, policy_counts.values, color=colors)
        
        ax.set_xlabel('Interview Policy', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Class Distribution of Interview Policies', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'class_distribution.png', dpi=self.config.output.PLOT_DPI, bbox_inches='tight')
        plt.close()
    
    def plot_semantic_features(self, df: pd.DataFrame, output_dir: Path):
        """Generate semantic feature distribution plots."""
        semantic_features = self.config.data.SEMANTIC_FEATURES
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(semantic_features):
            if feature in df.columns:
                axes[idx].hist(df[feature], bins=30, color=sns.color_palette()[idx], 
                            alpha=0.7, edgecolor='black')
                axes[idx].set_xlabel(feature, fontweight='bold')
                axes[idx].set_ylabel('Frequency', fontweight='bold')
                axes[idx].set_title(f'Distribution of {feature}', fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'semantic_feature_distribution.png', 
                   dpi=self.config.output.PLOT_DPI, bbox_inches='tight')
        plt.close()
    
    def plot_behavioral_features(self, df: pd.DataFrame, output_dir: Path):
        """Generate behavioral feature distribution plots."""
        behavioral_features = self.config.data.BEHAVIORAL_FEATURES
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(behavioral_features):
            if feature in df.columns:
                axes[idx].hist(df[feature], bins=30, color=sns.color_palette()[idx+4], 
                            alpha=0.7, edgecolor='black')
                axes[idx].set_xlabel(feature, fontweight='bold')
                axes[idx].set_ylabel('Frequency', fontweight='bold')
                axes[idx].set_title(f'Distribution of {feature}', fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'behavioral_feature_distribution.png', 
                   dpi=self.config.output.PLOT_DPI, bbox_inches='tight')
        plt.close()
    
    def plot_context_features(self, df: pd.DataFrame, output_dir: Path):
        """Generate context feature distribution plots."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Difficulty distribution
        if "Difficulty" in df.columns:
            df["Difficulty"].value_counts().plot(kind='bar', ax=axes[0], color=sns.color_palette()[0])
            axes[0].set_title('Difficulty Distribution', fontweight='bold')
            axes[0].set_xlabel('Difficulty', fontweight='bold')
            axes[0].set_ylabel('Count', fontweight='bold')
            axes[0].tick_params(axis='x', rotation=0)
        
        # Correct Streak distribution
        if "Correct Streak" in df.columns:
            axes[1].hist(df["Correct Streak"], bins=range(0, 7), color=sns.color_palette()[1],
                        alpha=0.7, edgecolor='black', align='left')
            axes[1].set_title('Correct Streak Distribution', fontweight='bold')
            axes[1].set_xlabel('Correct Streak', fontweight='bold')
            axes[1].set_ylabel('Frequency', fontweight='bold')
            axes[1].set_xticks(range(0, 6))
            axes[1].grid(True, alpha=0.3)
        
        # Wrong Streak distribution
        if "Wrong Streak" in df.columns:
            axes[2].hist(df["Wrong Streak"], bins=range(0, 7), color=sns.color_palette()[2],
                        alpha=0.7, edgecolor='black', align='left')
            axes[2].set_title('Wrong Streak Distribution', fontweight='bold')
            axes[2].set_xlabel('Wrong Streak', fontweight='bold')
            axes[2].set_ylabel('Frequency', fontweight='bold')
            axes[2].set_xticks(range(0, 6))
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'context_feature_distribution.png', 
                   dpi=self.config.output.PLOT_DPI, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, output_dir: Path):
        """Generate correlation heatmap for numerical features."""
        numerical_features = self.config.data.SEMANTIC_FEATURES + self.config.data.BEHAVIORAL_FEATURES
        numerical_df = df[numerical_features]
        
        correlation_matrix = numerical_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   fmt='.2f', ax=ax)
        ax.set_title('Feature Correlation Matrix', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmap.png', 
                   dpi=self.config.output.PLOT_DPI, bbox_inches='tight')
        plt.close()
    
    def plot_boxplots(self, df: pd.DataFrame, output_dir: Path):
        """Generate boxplots of features by policy class."""
        numerical_features = self.config.data.SEMANTIC_FEATURES + self.config.data.BEHAVIORAL_FEATURES
        target_col = self.config.data.TARGET_COLUMN
        
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        axes = axes.flatten()
        
        for idx, feature in enumerate(numerical_features):
            if feature in df.columns:
                sns.boxplot(data=df, x=target_col, y=feature, ax=axes[idx], 
                           hue=target_col, palette=sns.color_palette("husl", 7), legend=False)
                axes[idx].set_xlabel('Interview Policy', fontweight='bold')
                axes[idx].set_ylabel(feature, fontweight='bold')
                axes[idx].set_title(f'{feature} by Policy', fontweight='bold')
                axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'boxplots.png', dpi=self.config.output.PLOT_DPI, bbox_inches='tight')
        plt.close()
    
    def plot_streak_analysis(self, df: pd.DataFrame, output_dir: Path):
        """Generate streak analysis plots."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Correct Streak by Policy
            if "Correct Streak" in df.columns and self.config.data.TARGET_COLUMN in df.columns:
                correct_streak_by_policy = df.groupby(self.config.data.TARGET_COLUMN)["Correct Streak"].mean()
                correct_streak_by_policy.plot(kind='bar', ax=axes[0], color=sns.color_palette()[0])
                axes[0].set_title('Average Correct Streak by Policy', fontweight='bold')
                axes[0].set_xlabel('Policy', fontweight='bold')
                axes[0].set_ylabel('Average Correct Streak', fontweight='bold')
                axes[0].tick_params(axis='x', rotation=45)
            
            # Wrong Streak by Policy
            if "Wrong Streak" in df.columns and self.config.data.TARGET_COLUMN in df.columns:
                wrong_streak_by_policy = df.groupby(self.config.data.TARGET_COLUMN)["Wrong Streak"].mean()
                wrong_streak_by_policy.plot(kind='bar', ax=axes[1], color=sns.color_palette()[1])
                axes[1].set_title('Average Wrong Streak by Policy', fontweight='bold')
                axes[1].set_xlabel('Policy', fontweight='bold')
                axes[1].set_ylabel('Average Wrong Streak', fontweight='bold')
                axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'streak_analysis.png', dpi=self.config.output.PLOT_DPI)
            plt.close()
        except Exception as e:
            self.logger.warning(f"Could not generate streak analysis plot: {e}")
            plt.close('all')
    
    def plot_missing_concepts_analysis(self, df: pd.DataFrame, output_dir: Path):
        """Generate missing concepts analysis plot."""
        try:
            if "Missing Concepts" not in df.columns or self.config.data.TARGET_COLUMN not in df.columns:
                self.logger.warning("Missing Concepts or Policy column not found for analysis")
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            missing_by_policy = df.groupby(self.config.data.TARGET_COLUMN)["Missing Concepts"].mean()
            missing_by_policy.plot(kind='bar', ax=ax, color=sns.color_palette()[2])
            
            ax.set_title('Average Missing Concepts by Policy', fontweight='bold')
            ax.set_xlabel('Policy', fontweight='bold')
            ax.set_ylabel('Average Missing Concepts', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'missing_concepts_analysis.png', dpi=self.config.output.PLOT_DPI)
            plt.close()
        except Exception as e:
            self.logger.warning(f"Could not generate missing concepts analysis plot: {e}")
            plt.close('all')
    
    def generate_statistics_summary(self, df: pd.DataFrame, output_dir: Path):
        """
        Generate statistical summary of the dataset.
        
        Args:
            df: DataFrame to analyze
            output_dir: Directory to save statistics
        """
        self.logger.info("Generating statistical summary")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Dataset summary
        dataset_summary = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # Feature statistics
        numerical_features = self.config.data.SEMANTIC_FEATURES + self.config.data.BEHAVIORAL_FEATURES
        feature_stats = df[numerical_features].describe()
        
        # Policy distribution
        policy_distribution = df[self.config.data.TARGET_COLUMN].value_counts().to_dict()
        
        # Save to CSV
        pd.DataFrame([dataset_summary]).to_csv(output_path / 'dataset_summary.csv', index=False)
        feature_stats.to_csv(output_path / 'feature_statistics.csv')
        pd.DataFrame.from_dict(policy_distribution, orient='index', columns=['count']).to_csv(output_path / 'policy_distribution.csv')
        
        self.logger.info(f"Statistics saved to: {output_path}")
