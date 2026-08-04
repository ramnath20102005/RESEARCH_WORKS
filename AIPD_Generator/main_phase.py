"""
Phase-wise dataset generation for AIPD-100K Dataset Generator.

This module implements the phase-wise generation strategy for research-quality dataset creation,
with comprehensive validation, comparison reports, and documentation for each phase.
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

from config import get_config, InterviewPolicy
from generator import DatasetGenerator
from dataset_stats import DatasetStatistics
from plots import DatasetVisualizer
from metadata import MetadataGenerator
from comparison import ComparisonReportGenerator


class PhaseGenerator:
    """Manages phase-wise dataset generation."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the phase generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.config = get_config()
        self.random_seed = random_seed
        self.generator_version = "1.0.0"
        
        # Phase configurations
        self.phases = [
            {
                "name": "AIPD_100",
                "samples": 100,
                "description": "Initial test phase for manual inspection and rule verification"
            },
            {
                "name": "AIPD_1000",
                "samples": 1000,
                "description": "Statistical verification phase"
            },
            {
                "name": "AIPD_10000",
                "samples": 10000,
                "description": "Stability verification phase"
            },
            {
                "name": "AIPD_100000",
                "samples": 100000,
                "description": "Final production dataset"
            }
        ]
        
        # Store phase results for comparison
        self.phase_results: Dict[str, Dict[str, Any]] = {}
    
    def generate_phase(self, phase_config: Dict[str, Any], base_output_dir: str) -> Dict[str, Any]:
        """
        Generate a single phase dataset.
        
        Args:
            phase_config: Configuration for the phase
            base_output_dir: Base output directory
            
        Returns:
            Dictionary containing phase results
        """
        phase_name = phase_config["name"]
        target_samples = phase_config["samples"]
        
        print("\n" + "="*70)
        print(f"PHASE: {phase_name}")
        print(f"Description: {phase_config['description']}")
        print(f"Target Samples: {target_samples}")
        print("="*70)
        
        # Create phase-specific output directory
        phase_output_dir = Path(base_output_dir) / phase_name
        phase_output_dir.mkdir(parents=True, exist_ok=True)
        phase_plots_dir = phase_output_dir / "plots"
        phase_plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Update config for this phase
        self.config.TOTAL_SAMPLES = target_samples
        self.config.OUTPUT_DIR = str(phase_output_dir)
        self.config.PLOTS_DIR = str(phase_plots_dir)
        
        # Scale policy distribution proportionally
        original_total = sum(self.config.POLICY_DISTRIBUTION.values())
        scale_factor = target_samples / original_total
        self.config.POLICY_DISTRIBUTION = {
            policy: int(count * scale_factor)
            for policy, count in self.config.POLICY_DISTRIBUTION.items()
        }
        
        # Ensure total matches target (adjust for rounding)
        current_total = sum(self.config.POLICY_DISTRIBUTION.values())
        if current_total != target_samples:
            # Add/subtract from Switch Topic (usually smallest)
            diff = target_samples - current_total
            self.config.POLICY_DISTRIBUTION[InterviewPolicy.SWITCH_TOPIC] += diff
        
        start_time = time.time()
        
        # Initialize generator
        generator = DatasetGenerator(random_seed=self.random_seed)
        
        # Generate dataset
        print(f"\nGenerating {target_samples} samples...")
        df = generator.generate_dataset()
        
        # Validate final dataset
        print("Validating final dataset...")
        is_valid, issues = generator.validate_final_dataset(df)
        
        if is_valid:
            print("[PASS] Dataset validation passed")
        else:
            print("[FAIL] Dataset validation failed:")
            for issue in issues[:5]:  # Show first 5 issues
                print(f"  - {issue}")
            if len(issues) > 5:
                print(f"  ... and {len(issues) - 5} more issues")
        
        # Save dataset (remove Rule_ID)
        dataset_path = phase_output_dir / f"{phase_name}.csv"
        generator.save_dataset(df, str(dataset_path), remove_rule_id=True)
        
        # Save rejected rows
        rejected_path = phase_output_dir / "Rejected.csv"
        generator.save_rejected_rows(str(rejected_path))
        
        # Generate statistics
        print("Generating statistics...")
        stats_generator = DatasetStatistics()
        generation_stats = generator.get_generation_statistics()
        stats = stats_generator.generate_statistics(df, generation_stats)
        
        stats_path = phase_output_dir / "dataset_statistics.json"
        stats_generator.save_statistics(stats, str(stats_path))
        
        # Generate metadata
        print("Generating metadata...")
        metadata_generator = MetadataGenerator(self.generator_version)
        metadata = metadata_generator.generate_metadata(
            dataset_name=phase_name,
            dataset_version="1.0.0",
            random_seed=self.random_seed,
            rows_requested=target_samples,
            rows_generated=generation_stats['total_attempts'],
            rejected_rows=generation_stats['regenerated_rows'],
            accepted_rows=len(df),
            rule_count=7,
            feature_count=12,
            policy_count=7,
            generation_time_seconds=time.time() - start_time
        )
        
        metadata_path = phase_output_dir / "metadata.json"
        metadata_generator.save_metadata(metadata, str(metadata_path))
        metadata_generator.print_metadata_summary(metadata)
        
        # Generate plots
        print("Generating visualizations...")
        visualizer = DatasetVisualizer(str(phase_plots_dir))
        phase_suffix = f"_{target_samples}"
        visualizer.generate_all_plots(df, phase_suffix)
        
        # Generate phase-specific README
        self.generate_phase_readme(phase_name, phase_config, metadata, stats, phase_output_dir)
        
        # Reset generator for next phase
        generator.reset_rejected_rows()
        
        # Store results for comparison
        elapsed_time = time.time() - start_time
        phase_result = {
            "name": phase_name,
            "samples": target_samples,
            "df": df,
            "stats": stats,
            "metadata": metadata,
            "generation_time": elapsed_time,
            "is_valid": is_valid,
            "issues": issues
        }
        
        print(f"\nPhase {phase_name} completed in {elapsed_time:.2f} seconds")
        print(f"Dataset: {dataset_path}")
        print(f"Statistics: {stats_path}")
        print(f"Metadata: {metadata_path}")
        print(f"Rejected: {rejected_path}")
        
        return phase_result
    
    def generate_phase_readme(
        self,
        phase_name: str,
        phase_config: Dict[str, Any],
        metadata: Dict[str, Any],
        stats: Dict[str, Any],
        output_dir: Path
    ):
        """
        Generate a README for a specific phase.
        
        Args:
            phase_name: Name of the phase
            phase_config: Phase configuration
            metadata: Metadata dictionary
            stats: Statistics dictionary
            output_dir: Output directory
        """
        readme_content = f"""# {phase_name} Dataset

## Overview

{phase_config['description']}

- **Dataset Name**: {phase_name}
- **Total Samples**: {phase_config['samples']:,}
- **Generation Date**: {metadata['dataset_info']['generation_date']}
- **Random Seed**: {metadata['generation_parameters']['random_seed']}
- **Generator Version**: {metadata['dataset_info']['generator_version']}

## Generation Summary

| Metric | Value |
|--------|-------|
| Rows Requested | {metadata['generation_parameters']['rows_requested']:,} |
| Rows Generated | {metadata['generation_parameters']['rows_generated']:,} |
| Rejected Rows | {metadata['generation_parameters']['rejected_rows']:,} |
| Accepted Rows | {metadata['generation_parameters']['accepted_rows']:,} |
| Success Rate | {metadata['generation_parameters']['success_rate']:.2%} |
| Generation Time | {metadata['generation_parameters']['generation_time_seconds']:.2f}s |

## Feature Statistics

### Semantic Features

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
"""
        
        semantic_features = metadata['feature_groups']['semantic_features']
        for feature in semantic_features:
            if feature in stats['feature_statistics']:
                fs = stats['feature_statistics'][feature]
                readme_content += f"| {feature} | {fs['mean']:.2f} | {fs['std']:.2f} | {fs['min']:.2f} | {fs['max']:.2f} |\n"
        
        readme_content += "\n### Behavioral Features\n\n| Feature | Mean | Std | Min | Max |\n|---------|------|-----|-----|-----|\n"
        
        behavioral_features = metadata['feature_groups']['behavioral_features']
        for feature in behavioral_features:
            if feature in stats['feature_statistics']:
                fs = stats['feature_statistics'][feature]
                readme_content += f"| {feature} | {fs['mean']:.3f} | {fs['std']:.3f} | {fs['min']:.3f} | {fs['max']:.3f} |\n"
        
        readme_content += "\n## Policy Distribution\n\n"
        readme_content += "| Policy | Count | Percentage |\n"
        readme_content += "|--------|-------|------------|\n"
        
        for policy, dist in stats['class_distribution'].items():
            readme_content += f"| {policy} | {dist['count']:,} | {dist['percentage']:.2f}% |\n"
        
        readme_content += f"""
## Files Included

- `{phase_name}.csv` - Main dataset (without Rule_ID)
- `dataset_statistics.json` - Comprehensive statistics
- `metadata.json` - Generation metadata and provenance
- `Rejected.csv` - Rejected rows with validation reasons
- `plots/` - Visualization plots

## Plots

"""
        
        plot_files = [
            f"class_distribution_{phase_config['samples']}.png",
            f"feature_histograms_{phase_config['samples']}.png",
            f"correlation_heatmap_{phase_config['samples']}.png",
            f"semantic_distribution_{phase_config['samples']}.png",
            f"behavior_distribution_{phase_config['samples']}.png",
            f"policy_by_difficulty_{phase_config['samples']}.png",
            f"streak_distribution_{phase_config['samples']}.png",
            f"pairplot_{phase_config['samples']}.png"
        ]
        
        for plot_file in plot_files:
            readme_content += f"- `plots/{plot_file}`\n"
        
        readme_content += f"""
## Validation

"""
        
        if metadata['generation_parameters']['rejected_rows'] == 0:
            readme_content += "- All rows passed validation\n"
        else:
            readme_content += f"- {metadata['generation_parameters']['rejected_rows']:,} rows were rejected and regenerated\n"
            readme_content += "- See `Rejected.csv` for detailed rejection reasons\n"
        
        readme_content += f"""
## Methodology

This dataset was generated using the AIPD-100K Generator with the following methodology:

1. **Rule-Based Generation**: Each row originates from exactly one literature-derived policy rule
2. **Performance Index**: Behavioral features conditioned on Performance Index (0.45×Correctness + 0.30×Coverage + 0.25×Reasoning)
3. **Correlated Features**: Semantic and behavioral features generated with realistic correlations
4. **Validation**: Every row validated against consistency rules
5. **Regeneration**: Invalid rows automatically regenerated

## Reproducibility

To regenerate this exact dataset:

```bash
cd AIPD_Generator
python main_phase.py --phase {phase_name} --seed {metadata['generation_parameters']['random_seed']}
```

## Next Steps

"""
        
        if phase_name == "AIPD_100":
            readme_content += "- Review dataset manually\n- Verify all rules are working correctly\n- Check feature ranges and correlations\n- Proceed to Phase 2 (AIPD_1000)\n"
        elif phase_name == "AIPD_1000":
            readme_content += "- Review statistical properties\n- Compare with Phase 1 results\n- Verify distribution stability\n- Proceed to Phase 3 (AIPD_10000)\n"
        elif phase_name == "AIPD_10000":
            readme_content += "- Review statistical stability\n- Verify correlations are consistent\n- Check class balance\n- Proceed to Phase 4 (AIPD_100000)\n"
        else:
            readme_content += "- Final dataset ready for TabPFN training\n- Review all comparison reports\n- Use in IEEE research paper\n"
        
        readme_path = output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"Phase README saved to: {readme_path}")
    
    def generate_comparison_report(
        self,
        phase1_name: str,
        phase2_name: str,
        base_output_dir: str
    ):
        """
        Generate a comparison report between two phases.
        
        Args:
            phase1_name: Name of the first phase
            phase2_name: Name of the second phase
            base_output_dir: Base output directory
        """
        if phase1_name not in self.phase_results or phase2_name not in self.phase_results:
            print(f"Cannot generate comparison: missing phase results")
            return
        
        print(f"\nGenerating comparison report: {phase1_name} vs {phase2_name}")
        
        # Create reports directory
        reports_dir = Path(base_output_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataframes from CSV files to avoid memory issues
        import pandas as pd
        phase1_df = pd.read_csv(Path(base_output_dir) / phase1_name / f"{phase1_name}.csv")
        phase2_df = pd.read_csv(Path(base_output_dir) / phase2_name / f"{phase2_name}.csv")
        
        # Generate comparison
        comparison_generator = ComparisonReportGenerator()
        report = comparison_generator.generate_comparison_report(
            phase1_name,
            self.phase_results[phase1_name]['stats'],
            phase1_df,
            phase2_name,
            self.phase_results[phase2_name]['stats'],
            phase2_df
        )
        
        # Save report
        report_filename = f"comparison_{phase1_name}_vs_{phase2_name}.md"
        report_path = reports_dir / report_filename
        comparison_generator.save_comparison_report(report, str(report_path))
    
    def run_all_phases(self, base_output_dir: str = "output", start_phase: int = 0, end_phase: int = 4):
        """
        Run all phases sequentially with comparison reports.
        
        Args:
            base_output_dir: Base output directory
            start_phase: Starting phase index (0-3)
            end_phase: Ending phase index (0-4, exclusive)
        """
        print("="*70)
        print("AIPD-100K PHASE-WISE DATASET GENERATION")
        print("="*70)
        print(f"Random Seed: {self.random_seed}")
        print(f"Output Directory: {base_output_dir}")
        print(f"Phases: {start_phase} to {end_phase-1}")
        print("="*70)
        
        total_start_time = time.time()
        
        # Run each phase
        for i in range(start_phase, end_phase):
            if i >= len(self.phases):
                break
            
            phase_config = self.phases[i]
            
            # Reset generator state for each phase
            phase_result = self.generate_phase(phase_config, base_output_dir)
            self.phase_results[phase_result['name']] = phase_result
            
            # Generate comparison report if we have a previous phase
            if i > start_phase:
                prev_phase_name = self.phases[i-1]['name']
                current_phase_name = phase_config['name']
                self.generate_comparison_report(prev_phase_name, current_phase_name, base_output_dir)
        
        total_elapsed_time = time.time() - total_start_time
        
        print("\n" + "="*70)
        print("ALL PHASES COMPLETED")
        print("="*70)
        print(f"Total Time: {total_elapsed_time:.2f} seconds")
        print(f"Phases Completed: {len(self.phase_results)}")
        print("\nGenerated Datasets:")
        for phase_name, result in self.phase_results.items():
            print(f"  - {phase_name}: {result['samples']:,} samples ({'VALID' if result['is_valid'] else 'INVALID'})")
        print("\nComparison Reports:")
        reports_dir = Path(base_output_dir) / "reports"
        if reports_dir.exists():
            for report_file in reports_dir.glob("comparison_*.md"):
                print(f"  - {report_file.name}")
        print()


def main():
    """Main function to run phase-wise generation."""
    parser = argparse.ArgumentParser(
        description='Generate AIPD-100K dataset in phases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_phase.py                           # Run all phases
  python main_phase.py --phase AIPD_100         # Run specific phase
  python main_phase.py --start 0 --end 2        # Run phases 0 and 1
  python main_phase.py --seed 123               # Use custom random seed
        """
    )
    
    parser.add_argument(
        '--phase',
        type=str,
        choices=['AIPD_100', 'AIPD_1000', 'AIPD_10000', 'AIPD_100000'],
        help='Generate specific phase only'
    )
    
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Starting phase index (0-3)'
    )
    
    parser.add_argument(
        '--end',
        type=int,
        default=4,
        help='Ending phase index (0-4, exclusive)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Base output directory'
    )
    
    args = parser.parse_args()
    
    try:
        phase_generator = PhaseGenerator(random_seed=args.seed)
        
        if args.phase:
            # Run specific phase
            phase_index = {'AIPD_100': 0, 'AIPD_1000': 1, 'AIPD_10000': 2, 'AIPD_100000': 3}[args.phase]
            phase_config = phase_generator.phases[phase_index]
            phase_generator.generate_phase(phase_config, args.output)
        else:
            # Run all phases or range
            phase_generator.run_all_phases(args.output, args.start, args.end)
    
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
