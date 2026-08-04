# Git Repository Setup - .gitignore Configuration

## Overview
This document explains the .gitignore configuration for the Research_Project repository, optimized for GitHub deployment with research reproducibility in mind.

## Repository Structure

```
Research_Project/
├── .gitignore                    # Root-level gitignore
├── AIPD_Generator/
│   └── .gitignore               # Dataset generator gitignore
├── datasetprep/
│   └── .gitignore               # Dataset preparation gitignore
├── interview_training/
│   └── .gitignore               # Training pipeline gitignore
└── ResearchPapers/
    └── .gitignore               # Research papers gitignore
```

## Configuration Summary

### Root `.gitignore`
- **Excludes**: Environment variables, Python cache, IDE files, logs
- **Keeps**: All subprojects and their specific configurations
- **Purpose**: Clean root directory with proper Python/IDE exclusions

### `AIPD_Generator/.gitignore`
- **Excludes**: Small dataset sizes (AIPD_100, AIPD_1000, AIPD_10000)
- **Keeps**: Main dataset (AIPD_100000.csv - 6.5MB), metadata, reports
- **Purpose**: Keep main research dataset, exclude intermediate sizes

### `datasetprep/.gitignore`
- **Excludes**: Python cache, IDE files, logs
- **Keeps**: All dataset preparation scripts
- **Purpose**: Standard Python project configuration

### `interview_training/.gitignore`
- **Excludes**: 
  - Environment variables (.env with API keys)
  - Large binary files (probabilities - .npy files)
  - Large TabPFN models (200+MB each, exceed GitHub 100MB limit)
  - Model cache and training artifacts
- **Keeps**:
  - Small baseline models (Random Forest 35MB, XGBoost 1.7MB, CatBoost 427KB)
  - All research reports (.md files)
  - Metrics (.csv files)
  - Classification reports (.txt files)
  - Plots (.png files)
- **Purpose**: Balance research reproducibility with GitHub size limits

### `ResearchPapers/.gitignore`
- **Excludes**: Very large PDF files (Review4.pdf - 16MB)
- **Keeps**: Smaller reference papers (2.8-3.4MB each)
- **Purpose**: Keep accessible research references, exclude oversized files

## File Size Analysis

### Large Files Excluded
- **TabPFN Models**: 3 files × 200+MB each (exceeds GitHub 100MB limit)
- **Probabilities**: Binary files totaling ~2.5MB (excluded for cleanliness)
- **Review4.pdf**: 16MB (exceeds reasonable GitHub size)
- **Small Datasets**: AIPD_100, AIPD_1000, AIPD_10000 (intermediate files)

### Files Kept for Reproducibility
- **AIPD_100000.csv**: 6.5MB (main research dataset)
- **Baseline Models**: ~37MB total (Random Forest, XGBoost, CatBoost)
- **Research Reports**: ~50KB (documentation)
- **Metrics**: ~5KB (CSV files)
- **Plots**: ~2MB (PNG visualizations)

## GitHub Considerations

### 100MB File Limit
- TabPFN models (200+MB each) are excluded due to GitHub's 100MB file size limit
- These can be stored using Git LFS or alternative storage if needed

### Repository Size
- Estimated final repository size: ~50MB
- Well within GitHub limits for fast cloning and operations
- Maintains research reproducibility without excessive size

### Security
- `.env` files excluded to protect API keys and sensitive configuration
- Environment variables should be documented in README files

## Future Recommendations

### For Large Files
If you need to store TabPFN models or large datasets:
1. **Git LFS**: Install Git Large File Storage for files >100MB
2. **Alternative Storage**: Use external storage (S3, Google Drive) with documentation
3. **Model Versioning**: Consider model registries (MLflow, Hugging Face)

### For Datasets
If AIPD_100000.csv grows beyond reasonable size:
1. **Data Versioning**: Use DVC (Data Version Control)
2. **Sample Data**: Keep a sample in git, full data elsewhere
3. **Streaming**: Document data generation process for reproducibility

## Setup Instructions

### Initialize Git Repository
```bash
cd Research_Project
git init
git add .
git commit -m "Initial commit with .gitignore configuration"
```

### Verify Configuration
```bash
# Check what will be tracked
git status

# Check ignored files
git check-ignore -v *
```

### Push to GitHub
```bash
git remote add origin <your-repository-url>
git branch -M main
git push -u origin main
```

## Summary

The .gitignore configuration is optimized for:
- **Research Reproducibility**: Keeps essential models, datasets, and reports
- **GitHub Compatibility**: Respects 100MB file size limits
- **Security**: Excludes sensitive API keys and environment variables
- **Clean Repository**: Organized structure with clear documentation

The repository is ready for GitHub push with appropriate file exclusions while maintaining research reproducibility.

---

**Configuration Date**: August 4, 2026  
**Repository Size**: ~50MB estimated  
**Status**: Ready for GitHub deployment