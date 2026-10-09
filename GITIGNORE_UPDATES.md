# .gitignore Updates Summary

**Date:** 2026-10-09
**Purpose:** Prepare repository for GitHub with comprehensive .gitignore configurations

---

## Changes Made

### 1. Root `.gitignore` (Main Repository)

**Added:**
- Python testing artifacts: `.pytest_cache/`, `.coverage`, `htmlcov/`, `.mypy_cache/`
- Frontend build directories: `adaptive/frontend/.cache/`
- Additional IDE files: `*.sublime-project`, `*.sublime-workspace`, `.history/`
- Cache directories: `adaptive/backend/.cache/`
- Log directory: `logs/`, `training.log`
- More audio test files: `test_*.flac`
- Cleanup audit report: `CLEANUP_AUDIT_REPORT.md`

**Enhanced:**
- Better coverage for Python bytecode
- Comprehensive frontend build artifacts
- HuggingFace cache directory
- Better OS file coverage (Windows, macOS, Linux)

### 2. `adaptive/frontend/.gitignore`

**Added:**
- Dependency lock files: `.pnp`, `.pnp.js`
- Build directories: `build/`
- Testing coverage: `coverage/`, `.nyc_output/`
- Vite cache: `.vite/`

**Enhanced:**
- Better editor file coverage
- Testing artifacts

### 3. Existing .gitignore Files (No Changes)

- `AIPD_Generator/.gitignore` - Already well-configured for research data
- `interview_training/.gitignore` - Already excludes large models and artifacts
- `ResearchPapers/.gitignore` - Already excludes large PDFs

---

## What Will Be Ignored

### Python
- `__pycache__/`, `*.pyc`, `*.pyo`
- `.pytest_cache/`, `.coverage`, `htmlcov/`
- `.mypy_cache/`, `.dmypy.json`
- Build artifacts: `dist/`, `build/`, `*.egg-info/`

### Frontend
- `node_modules/`
- `dist/`, `build/`, `.vite/`
- Frontend cache: `.cache/`
- Log files: `npm-debug.log*`, `yarn-debug.log*`

### Models & Data
- Large TabPFN models: `tabpfn_*.pkl`
- Large datasets: `*.csv` (except small ones)
- Binary files: `*.npy`, `*.npz`
- ONNX models: `kokoro-v1.0.onnx`, `voices.bin`

### Test Artifacts
- Test audio: `test_*.wav`, `test_*.mp3`, `test_*.webm`, `test_*.flac`
- Parsed resume outputs: `adaptive/backend/app/output/*.json`
- Diagnostic scripts: `audit_*.py`, `benchmark_*.py`, `diagnose_*.py`
- Temporary test scripts: `test_nim_*.py`, `test_stt_*.py`, `test_qwen*.py`

### Cache & Logs
- `.cache/`, `.huggingface/`, `adaptive/backend/.cache/`
- `*.log`, `logs/`, `training.log`

### IDE & OS
- `.vscode/`, `.idea/`, `*.swp`, `*.swo`
- `.DS_Store`, `Thumbs.db`, `desktop.ini`
- `.history/`, `*.sublime-project`, `*.sublime-workspace`

### Security
- `.env`, `*.env` (API keys and secrets)

---

## What Will Be Included

### Source Code
- All Python source files in `adaptive/backend/app/`
- All React/JSX source files in `adaptive/frontend/src/`
- All training scripts in `interview_training/`
- All generator scripts in `AIPD_Generator/`

### Configuration
- `requirements.txt`, `pyproject.toml`
- `package.json`, `package-lock.json`
- `vite.config.js`
- `.env.example` (template, not actual secrets)

### Documentation
- `README.md` files
- Research reports in `interview_training/outputs/reports/`
- Classification reports
- Documentation in `AIPD_Generator/output/`

### Small Models
- `random_forest.pkl` (35MB - baseline model)
- `xgboost.pkl` (1.7MB - baseline model)
- `catboost.pkl` (427KB - baseline model)

### Research Data
- `AIPD_Generator/output/AIPD_100000/` (main dataset with metadata)
- Small research papers (< 100MB)

### Tests
- Organized tests in `adaptive/backend/tests/`
- Organized tests in `adaptive/backend/app/tests/`
- Root-level useful tests: `test_all_policies_local.py`, `test_no_answer_regression.py`, `test_streak_*.py`, `test_tabpfn_policies.py`

---

## Security Notes

**Never Commit:**
- `.env` files with actual API keys
- NVIDIA API keys
- Database credentials
- Any secrets or sensitive configuration

**Always Include:**
- `.env.example` (template)
- Configuration documentation
- Setup instructions

---

## Next Steps

1. **Review the .gitignore files** to ensure they match your needs
2. **Commit the .gitignore updates** to Git
3. **Run `git status`** to see what files will be tracked
4. **Remove any accidentally tracked files** using `git rm --cached`
5. **Commit and push** to GitHub

---

## Verification Commands

```bash
# Check what files will be tracked
git status

# Check ignored files
git check-ignore -v *

# Remove accidentally tracked files
git rm --cached <file>

# Stage and commit .gitignore updates
git add .gitignore
git add adaptive/frontend/.gitignore
git commit -m "Update .gitignore files for GitHub"
```
