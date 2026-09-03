# Adaptive AI Interview System

A complete research project implementing an adaptive interview system using Tabular Foundation Models (TabPFN) for intelligent question generation and candidate evaluation.

## 📁 Project Structure

```
Research_Project/
├── adaptive/                    # Main interview application (Frontend + Backend)
│   ├── frontend/               # React + Vite frontend
│   └── backend/                # FastAPI backend with Local LLM & TabPFN
│
├── interview_training/         # TabPFN model training pipeline
│   ├── data/                   # Training datasets
│   ├── models/                 # Trained TabPFN models
│   ├── outputs/                # Training outputs (metrics, reports)
│   └── train.py                # Training script
│
├── AIPD_Generator/            # Synthetic dataset generator
│   └── generator.py            # Dataset generation script
│
├── ResearchPapers/             # Literature and research documents
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- CUDA-capable GPU (for Local LLM)
- Git

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Research_Project
```

### 2. Setup Adaptive Interview System (Frontend + Backend)

#### Backend Setup

```bash
cd adaptive/backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
# Copy .env.example to .env and update values
# Required variables:
# LLM_PROVIDER=local_qwen
# LOCAL_QWEN_MODEL=Qwen/Qwen1.5-4B-Chat
# LOCAL_QWEN_DEVICE=auto
# LOCAL_QWEN_QUANTIZATION=int4

# Download required models (if not included)
# - TabPFN model: interview_training/outputs/models/tabpfn_10000.pkl
# - Kokoro TTS: backend/kokoro-v1.0.onnx
# - Kokoro voices: backend/voices.bin

# Start FastAPI server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Backend will run at: `http://127.0.0.1:8000`  
API Docs: `http://127.0.0.1:8000/docs`

#### Frontend Setup

```bash
cd adaptive/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will run at: `http://localhost:5173`

### 3. Setup Interview Training (Optional - for model training)

```bash
cd interview_training

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Prepare training data
# Place your dataset in data/ directory

# Train TabPFN model
python train.py --config configs/default_config.yaml

# Trained models will be saved in outputs/models/
```

### 4. Setup AIPD Generator (Optional - for dataset generation)

```bash
cd AIPD_Generator

# Install dependencies
pip install -r requirements.txt

# Generate synthetic dataset
python generator.py --config config.yaml

# Output will be saved in output/ directory
```

## 📡 API Endpoints

### Adaptive Interview Backend

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload-resume` | Upload and parse resume (PDF/DOCX) |
| GET | `/skills` | Get extracted technical skills |
| GET | `/resume-summary` | Get candidate profile summary |
| POST | `/interview/adaptive/start` | Start adaptive interview session |
| POST | `/interview/adaptive/answer` | Submit answer and get next question |
| POST | `/interview/transcribe` | Transcribe audio to text (Whisper) |
| GET | `/health` | Health check endpoint |

## 🎯 System Architecture

### Adaptive Interview Pipeline

1. **Resume Parsing** → Extract candidate profile and skills
2. **First Question Generation** → Local Qwen LLM generates initial question
3. **Answer Recording** → Capture audio response
4. **Speech-to-Text** → Whisper transcribes audio
5. **Semantic Evaluation** → Local Qwen LLM evaluates answer quality
6. **Feature Extraction** → Build 11-feature vector
7. **TabPFN Inference** → Predict next question policy
8. **Question Generation** → Local Qwen LLM generates next question
9. **Text-to-Speech** → Kokoro TTS synthesizes audio

### Technology Stack

- **Frontend**: React, Vite, TailwindCSS
- **Backend**: FastAPI, Python
- **LLM**: Local Qwen 1.5 4B (INT4 quantized)
- **Adaptive Policy**: TabPFN (Tabular Foundation Model)
- **STT**: Faster Whisper
- **TTS**: Kokoro TTS
- **Resume Parsing**: pdfplumber, python-docx, RapidFuzz

## 🔧 Configuration

### Environment Variables (Backend)

Create `adaptive/backend/.env`:

```env
# LLM Configuration
LLM_PROVIDER=local_qwen
LOCAL_QWEN_MODEL=Qwen/Qwen1.5-4B-Chat
LOCAL_QWEN_DEVICE=auto
LOCAL_QWEN_QUANTIZATION=int4

# TabPFN Configuration
TABPFN_MODEL_PATH=interview_training/outputs/models/tabpfn_10000.pkl

# TTS Configuration
KOKORO_MODEL_PATH=backend/kokoro-v1.0.onnx
KOKORO_VOICES_PATH=backend/voices.bin
```

### Model Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended 8GB+ VRAM)
- **Local LLM**: Qwen/Qwen1.5-4B-Chat (~4GB with INT4)
- **TabPFN**: tabpfn_10000.pkl (~220MB)
- **Kokoro TTS**: kokoro-v1.0.onnx (~325MB), voices.bin (~28MB)

## 📊 Model Files

### Included in Repository (Small Files)

- `interview_training/outputs/models/catboost.pkl` (~427KB)
- `interview_training/outputs/models/xgboost.pkl` (~1.6MB)

### Not Included (Large Files - Download Separately)

- `interview_training/outputs/models/random_forest.pkl` (~34MB)
- `interview_training/outputs/models/tabpfn_*.pkl` (~216-228MB each)
- `adaptive/backend/kokoro-v1.0.onnx` (~325MB)
- `adaptive/backend/voices.bin` (~28MB)

**Download Instructions:**
1. Contact project maintainer for large model files
2. Or train models yourself using `interview_training/train.py`
3. Download Kokoro models from official repository

## 🧪 Testing

### Backend Tests

```bash
cd adaptive/backend
pytest tests/
```

### End-to-End Test

```bash
cd adaptive/backend
python test_end_to_end_new_prompts.py
```

## 📝 Documentation

- **Adaptive System**: `adaptive/README.md`
- **Interview Training**: `interview_training/README.md`
- **Research Papers**: `ResearchPapers/` directory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This is a research project. Please contact the maintainers for usage permissions.

## 🐛 Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size or use CPU by setting `LOCAL_QWEN_DEVICE=cpu`

**Issue**: Whisper returns empty transcript
- **Solution**: Check microphone permissions and ensure audio is being recorded

**Issue**: TabPFN model not found
- **Solution**: Download the model and place it in `interview_training/outputs/models/`

**Issue**: Kokoro TTS not working
- **Solution**: Ensure ONNX model and voices.bin are in the correct directory

## 📞 Support

For issues and questions, please contact the project maintainers or open an issue in the repository.

## 🔗 Related Resources

- [TabPFN Paper](https://arxiv.org/abs/2207.01848)
- [Qwen LLM](https://huggingface.co/Qwen/Qwen1.5-4B-Chat)
- [Kokoro TTS](https://github.com/remsky/Kokoro-FastAPI)
- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper)
