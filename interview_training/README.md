# Interview Training - TabPFN Integration

## Project Overview

This project implements TabPFN as the sole adaptive decision engine for the Adaptive AI Interview System. TabPFN receives an 11-dimensional feature vector and predicts the optimal next interview policy from 7 possible outcomes.

## System Architecture

```
11-Dimensional Feature Vector → TabPFN → Interview Policy → LLM Question Generation
```

### Input Features (11 dimensions)
- **Semantic Features**: Correctness Score, Concept Coverage, Reasoning Score, Missing Concepts
- **Behavioral Features**: Engagement Score, Confidence Score, Hesitation Score, Eye Contact Score  
- **Context Features**: Difficulty, Correct Streak, Wrong Streak

### Output Policies (7 classes)
- Increase Difficulty
- Maintain Difficulty
- Reduce Difficulty
- Ask Follow-up
- Ask Conceptual Question
- Ask Practical Question
- End Interview

## Project Structure

```
interview_training/
├── configs/                  # Configuration files
│   ├── config.py            # Main configuration
│   └── tabpfn_config.py     # TabPFN-specific configuration
├── data/                     # Data processing modules
│   ├── loader.py            # Dataset loading
│   ├── preprocessing.py     # Feature engineering
│   └── validator.py         # Data validation
├── eda/                      # Exploratory Data Analysis
│   └── eda.py               # EDA analysis and visualization
├── evaluation/               # Model evaluation
│   ├── metrics.py           # Performance metrics
│   ├── research_report.py   # Research report generation
│   └── discussion_generator.py # Discussion generation
├── explainability/          # Model explainability
│   └── permutation_importance.py # Feature importance analysis
├── models/                   # Model implementations
│   └── all_models.py        # All model definitions (TabPFN + baselines)
├── utils/                    # Utility functions
│   ├── logger.py            # Logging utilities
│   └── helpers.py           # Helper functions
├── outputs/                  # Generated outputs
│   ├── classification_reports/ # Classification reports per model
│   ├── metrics/             # Performance metrics
│   ├── models/              # Trained model files (.pkl)
│   ├── plots/               # Visualization plots
│   ├── probabilities/       # Prediction probabilities (.npy)
│   └── reports/             # Research reports and analysis
├── .env                      # Environment variables (TabPFN token)
├── .gitignore               # Git ignore rules
├── requirements.txt          # Python dependencies
├── train.py                 # Main training pipeline
└── README.md                # This file
```

## Key Files

### Training Pipeline
- **train.py**: Main orchestration script for the entire training pipeline

### Configuration
- **configs/config.py**: Main configuration for data, output paths, and evaluation
- **configs/tabpfn_config.py**: TabPFN-specific configuration including training sizes

### Outputs
- **outputs/models/**: Trained models (tabpfn_5000.pkl, tabpfn_10000.pkl, tabpfn_20000.pkl, random_forest.pkl, xgboost.pkl, catboost.pkl)
- **outputs/reports/**: Research reports and analysis
- **outputs/plots/**: Confusion matrices and visualizations
- **outputs/classification_reports/**: Detailed classification reports per model

## Training Configuration

### TabPFN Training Sizes
- 5,000 samples (few-shot learning validation)
- 10,000 samples (optimal configuration)
- 20,000 samples (performance ceiling validation)

### Baseline Models
- Random Forest (69,999 samples)
- XGBoost (69,999 samples)
- CatBoost (69,999 samples)

**Note**: Baseline models are used only for research comparison and validation for the IEEE paper. They are not part of the production system.

## Requirements

### Hardware
- **GPU**: NVIDIA RTX 5050 or equivalent (8GB VRAM minimum)
- **CUDA**: 12.0+
- **PyTorch**: 2.0+

### Environment Variables
- **TABPFN_TOKEN**: TabPFN authentication token
- **TABPFN_NO_BROWSER**: Set to `true` for non-interactive authentication

### Python Dependencies
See `requirements.txt` for full list of dependencies.

## Usage

### Run Training Pipeline
```bash
python train.py
```

### Pipeline Stages
1. Dataset loading and validation
2. Exploratory Data Analysis
3. Data preprocessing and encoding
4. Model training (baselines + TabPFN scalability)
5. Model comparison
6. Metrics saving
7. Best model determination
8. Final model comparison generation
9. Research report generation

## Research Outputs

### Key Reports
- **final_experimental_analysis.md**: Comprehensive 10-section experimental analysis
- **tabpfn_system_integration.md**: TabPFN integration and system architecture
- **PROJECT_SCOPE_CLARIFICATION.md**: Project scope and architecture clarification
- **tabpfn_research_summary.md**: TabPFN research summary
- **final_model_comparison.csv**: Final model comparison table

### Performance Results
- **TabPFN (10K)**: 99.97% accuracy, 0.9997 F1-Macro, 48ms inference latency
- **TabPFN (20K)**: 99.99% accuracy, 0.9999 F1-Macro, 231ms inference latency
- **Random Forest**: 99.49% accuracy, 0.9952 F1-Macro, 0.07s inference time
- **XGBoost**: 99.24% accuracy, 0.9927 F1-Macro, 0.02s inference time
- **CatBoost**: 96.83% accuracy, 0.9693 F1-Macro, 0.01s inference time

## Selected Configuration

**TabPFN (10K Training Samples)** is selected as the sole adaptive decision engine for the Adaptive Interview System.

### Justification
- Superior performance (99.97% accuracy)
- Few-shot learning capability (optimal with 10K samples)
- Real-time inference (48ms latency)
- Foundation model generalization
- Future-proof transformer architecture

## Important Notes

### Baseline Models Purpose
Random Forest, XGBoost, and CatBoost are included **solely** as baseline models for experimental comparison in the IEEE research paper. They are **not** part of the production system architecture.

### TabPFN Role
TabPFN is the **sole** adaptive decision engine. It:
- Receives 11-dimensional feature vectors
- Predicts interview policies
- Does NOT generate questions
- Does NOT parse resumes
- Does NOT perform semantic evaluation

## Future Work

- TabPFN integration into complete interview pipeline
- End-to-end system testing
- Real-time performance validation
- IEEE paper finalization
- Production deployment preparation

---

**Status**: TabPFN validation complete, integration in progress  
**Date**: August 4, 2026  
**Hardware**: NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM)