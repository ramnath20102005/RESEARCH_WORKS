# AIPD_100000 Dataset

## Final Production Dataset

This is the final production dataset for the Adaptive Interview Policy Dataset (AIPD-100K) research project.

## Dataset Information

- **Dataset Name**: AIPD_100000
- **Total Samples**: 100,000
- **Features**: 12 (4 semantic, 4 behavioral, 3 context, 1 target)
- **File Size**: 6.5MB
- **Format**: CSV
- **Generation Date**: August 3, 2026
- **Random Seed**: 42
- **Generator Version**: 1.0.0

## Files

- **AIPD_100000.csv**: Main dataset file (100,000 rows × 13 columns)
- **dataset_statistics.json**: Comprehensive statistical analysis
- **metadata.json**: Generation metadata and quality metrics
- **Rejected.csv**: Rows that failed validation (for debugging)
- **plots/**: Publication-quality visualizations

## Dataset Structure

### Features (12)

**Semantic Features (4)**:
- Correctness Score (0-100)
- Concept Coverage (0-100)
- Reasoning Score (0-100)
- Missing Concepts (0-8)

**Behavioral Features (4)**:
- Engagement Score (0-1)
- Confidence Score (0-1)
- Hesitation Score (0-1)
- Eye Contact Score (0-1)

**Context Features (3)**:
- Difficulty (Easy/Medium/Hard)
- Correct Streak (0-5)
- Wrong Streak (0-5)

**Target Label (1)**:
- Next Interview Policy (7 classes)

### Target Classes

1. Increase Difficulty
2. Maintain Difficulty
3. Reduce Difficulty
4. Probe Missing Concept
5. Ask Application Question
6. Ask Follow-up Question
7. Switch Topic

## Quality Metrics

- **Validation Success Rate**: >95%
- **Regeneration Rate**: <5%
- **Class Balance**: Approximately balanced
- **Feature Correlations**: Realistic patterns

## Usage

Load the dataset in Python:

```python
import pandas as pd

# Load dataset
df = pd.read_csv('AIPD_100000.csv')

# Separate features and target
X = df.drop('Next Interview Policy', axis=1)
y = df['Next Interview Policy']
```

## Visualization

The `plots/` directory contains publication-quality visualizations:
- Class distribution
- Feature histograms
- Correlation heatmaps
- Policy-difficulty relationships
- Streak distributions

## References

This dataset is part of the research project:
"Adaptive AI Interview System Using Tabular Foundation Models for Interview Policy Prediction"

For full methodology and generation details, see the main project README.