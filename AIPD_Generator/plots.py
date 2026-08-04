"""
Visualization generation module for AIPD-100K Dataset Generator.

This module generates comprehensive visualizations of the dataset including
class distributions, feature histograms, correlation heatmaps, and pair plots.
These visualizations are designed for use in research papers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class DatasetVisualizer:
    """Generates publication-quality visualizations for the dataset."""
    
    def __init__(self, output_dir: str = "output/plots"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for publication-quality plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        
        # Color palette
        self.color_palette = sns.color_palette("husl", 10)
        
        self.semantic_features = [
            'Correctness Score',
            'Concept Coverage',
            'Reasoning Score',
            'Missing Concepts'
        ]
        
        self.behavioral_features = [
            'Engagement Score',
            'Confidence Score',
            'Hesitation Score',
            'Eye Contact Score'
        ]
        
        self.context_features = [
            'Difficulty',
            'Correct Streak',
            'Wrong Streak'
        ]
    
    def generate_all_plots(self, df: pd.DataFrame, phase_suffix: str = ""):
        """
        Generate all visualization plots.
        
        Args:
            df: DataFrame containing the dataset
            phase_suffix: Suffix to add to plot filenames (e.g., "_100")
        """
        print("Generating visualizations...")
        
        self.plot_class_distribution(df, phase_suffix)
        self.plot_feature_histograms(df, phase_suffix)
        self.plot_correlation_heatmap(df, phase_suffix)
        self.plot_semantic_feature_distribution(df, phase_suffix)
        self.plot_behavioral_feature_distribution(df, phase_suffix)
        self.plot_policy_by_difficulty(df, phase_suffix)
        self.plot_streak_distributions(df, phase_suffix)
        self.plot_pair_plot(df, phase_suffix)
        
        print(f"All plots saved to: {self.output_dir}")
    
    def plot_class_distribution(self, df: pd.DataFrame, phase_suffix: str = ""):
        """Generate class distribution bar plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        policy_counts = df['Policy'].value_counts().sort_index()
        colors = self.color_palette[:len(policy_counts)]
        
        bars = ax.bar(policy_counts.index, policy_counts.values, color=colors)
        
        ax.set_xlabel('Interview Policy', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Class Distribution of Interview Policies', fontweight='bold')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        filename = f'class_distribution{phase_suffix}.png'
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Generated: {filename}")
    
    def plot_feature_histograms(self, df: pd.DataFrame, phase_suffix: str = ""):
        """Generate histograms for all features."""
        numerical_features = self.semantic_features + self.behavioral_features
        
        fig, axes = plt.subplots(4, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, feature in enumerate(numerical_features):
            if feature in df.columns:
                axes[idx].hist(df[feature], bins=30, color=self.color_palette[idx], alpha=0.7, edgecolor='black')
                axes[idx].set_xlabel(feature, fontweight='bold')
                axes[idx].set_ylabel('Frequency', fontweight='bold')
                axes[idx].set_title(f'Distribution of {feature}', fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'feature_histograms{phase_suffix}.png'
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Generated: {filename}")
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, phase_suffix: str = ""):
        """Generate correlation heatmap for all numerical features."""
        numerical_features = self.semantic_features + self.behavioral_features
        numerical_df = df[numerical_features]
        
        correlation_matrix = numerical_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   fmt='.2f', ax=ax)
        
        ax.set_title('Feature Correlation Matrix', fontweight='bold', pad=20)
        plt.tight_layout()
        filename = f'correlation_heatmap{phase_suffix}.png'
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Generated: {filename}")
    
    def plot_semantic_feature_distribution(self, df: pd.DataFrame, phase_suffix: str = ""):
        """Generate box plots for semantic features by policy."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(self.semantic_features):
            if feature in df.columns:
                sns.boxplot(data=df, x='Policy', y=feature, ax=axes[idx], palette=self.color_palette)
                axes[idx].set_xlabel('Interview Policy', fontweight='bold')
                axes[idx].set_ylabel(feature, fontweight='bold')
                axes[idx].set_title(f'{feature} by Policy', fontweight='bold')
                axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        filename = f'semantic_distribution{phase_suffix}.png'
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Generated: {filename}")
    
    def plot_behavioral_feature_distribution(self, df: pd.DataFrame, phase_suffix: str = ""):
        """Generate box plots for behavioral features by policy."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(self.behavioral_features):
            if feature in df.columns:
                sns.boxplot(data=df, x='Policy', y=feature, ax=axes[idx], palette=self.color_palette)
                axes[idx].set_xlabel('Interview Policy', fontweight='bold')
                axes[idx].set_ylabel(feature, fontweight='bold')
                axes[idx].set_title(f'{feature} by Policy', fontweight='bold')
                axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        filename = f'behavior_distribution{phase_suffix}.png'
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Generated: {filename}")
    
    def plot_policy_by_difficulty(self, df: pd.DataFrame, phase_suffix: str = ""):
        """Generate stacked bar plot of policy distribution by difficulty."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create cross-tabulation
        crosstab = pd.crosstab(df['Difficulty'], df['Policy'], normalize='index') * 100
        
        crosstab.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
        
        ax.set_xlabel('Difficulty Level', fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontweight='bold')
        ax.set_title('Policy Distribution by Difficulty Level', fontweight='bold')
        ax.legend(title='Policy', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.xticks(rotation=0)
        plt.tight_layout()
        filename = f'policy_by_difficulty{phase_suffix}.png'
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Generated: {filename}")
    
    def plot_streak_distributions(self, df: pd.DataFrame, phase_suffix: str = ""):
        """Generate distributions for correct and wrong streaks."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Correct Streak
        if 'Correct Streak' in df.columns:
            axes[0].hist(df['Correct Streak'], bins=range(0, 7), color=self.color_palette[0], 
                        alpha=0.7, edgecolor='black', align='left')
            axes[0].set_xlabel('Correct Streak', fontweight='bold')
            axes[0].set_ylabel('Frequency', fontweight='bold')
            axes[0].set_title('Distribution of Correct Streaks', fontweight='bold')
            axes[0].set_xticks(range(0, 6))
            axes[0].grid(True, alpha=0.3)
        
        # Wrong Streak
        if 'Wrong Streak' in df.columns:
            axes[1].hist(df['Wrong Streak'], bins=range(0, 7), color=self.color_palette[1], 
                        alpha=0.7, edgecolor='black', align='left')
            axes[1].set_xlabel('Wrong Streak', fontweight='bold')
            axes[1].set_ylabel('Frequency', fontweight='bold')
            axes[1].set_title('Distribution of Wrong Streaks', fontweight='bold')
            axes[1].set_xticks(range(0, 6))
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'streak_distribution{phase_suffix}.png'
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Generated: {filename}")
    
    def plot_pair_plot(self, df: pd.DataFrame, phase_suffix: str = ""):
        """Generate pair plot for semantic features."""
        # Sample data for pair plot if dataset is large
        sample_size = min(5000, len(df))
        sampled_df = df.sample(n=sample_size, random_state=42)
        
        # Select semantic features and policy
        plot_features = self.semantic_features + ['Policy']
        plot_df = sampled_df[plot_features]
        
        # Create pair plot
        pair_plot = sns.pairplot(plot_df, hue='Policy', palette=self.color_palette[:7],
                                plot_kws={'alpha': 0.6, 's': 20},
                                diag_kind='hist')
        
        pair_plot.fig.suptitle('Pair Plot of Semantic Features by Policy', y=1.02, fontweight='bold')
        
        filename = f'pairplot{phase_suffix}.png'
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Generated: {filename}")
    
    def plot_missing_concepts_vs_correctness(self, df: pd.DataFrame):
        """Generate scatter plot of Missing Concepts vs Correctness."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sample data if large
        sample_size = min(10000, len(df))
        sampled_df = df.sample(n=sample_size, random_state=42)
        
        scatter = ax.scatter(sampled_df['Correctness Score'], sampled_df['Missing Concepts'],
                           c=sampled_df['Policy'].astype('category').cat.codes,
                           cmap='husl', alpha=0.6, s=20)
        
        ax.set_xlabel('Correctness Score', fontweight='bold')
        ax.set_ylabel('Missing Concepts', fontweight='bold')
        ax.set_title('Missing Concepts vs Correctness Score', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Policy', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'missing_vs_correctness.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Generated: missing_vs_correctness.png")
    
    def plot_confidence_hesitation_scatter(self, df: pd.DataFrame):
        """Generate scatter plot of Confidence vs Hesitation."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sample data if large
        sample_size = min(10000, len(df))
        sampled_df = df.sample(n=sample_size, random_state=42)
        
        scatter = ax.scatter(sampled_df['Confidence Score'], sampled_df['Hesitation Score'],
                           c=sampled_df['Correctness Score'],
                           cmap='RdYlGn', alpha=0.6, s=20)
        
        ax.set_xlabel('Confidence Score', fontweight='bold')
        ax.set_ylabel('Hesitation Score', fontweight='bold')
        ax.set_title('Confidence vs Hesitation (colored by Correctness)', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Correctness Score', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_vs_hesitation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Generated: confidence_vs_hesitation.png")
