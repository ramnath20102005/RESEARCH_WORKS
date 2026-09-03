# Adaptive AI Interview System - Resume Parsing & Skill Extraction Module

An independent, local, deterministic **Resume Intelligence & Skill Extraction Service** built for the research project:

> **Adaptive AI Interview System Using Tabular Foundation Models (TabPFN)**

---

## 🎯 Scope & Responsibilities

This module is **strictly non-LLM based** and purely responsible for:
1. Parsing **PDF** and **DOCX** Resumes.
2. Cleaning raw text and identifying sections.
3. Extracting **Candidate Information** (Name, Email, Phone, Social links, Education, Projects, Experience, Certifications).
4. Extracting and categorizing **Technical Skills** normalized against a **Master Skills Dictionary (~1,000+ skills)** using fuzzy matching (`RapidFuzz`).
5. Calculating **Skill Confidence Scores (0.0 – 1.0)** and **Frequencies**.
6. Performing **Heuristic Experience Level Detection** (`Beginner`, `Intermediate`, `Advanced`).
7. Returning structured **JSON** ready for direct downstream consumption by the Question Generator / TabPFN module.

It **does NOT** call any LLMs (OpenAI, Gemini, Claude, etc.), **does NOT** generate interview questions, and **does NOT** perform interview evaluations.

---

## 🏗 System Architecture & Folder Structure

```
AdaptiveInterviewSystem/
├── frontend/
│   ├── src/
│   │   ├── components/       # FileUpload, Navbar, Cards
│   │   ├── pages/            # Upload, ExtractedResume, ExtractedSkills, Statistics
│   │   ├── services/         # Axios API service
│   │   ├── styles/           # Global CSS (Dark Glassmorphism styling)
│   │   ├── App.jsx           # Main React Router setup
│   │   └── main.jsx
│   ├── index.html
│   └── package.json
│
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI endpoints (/upload-resume, /skills, /resume-summary, /health)
│   │   ├── services/         # Core parsing orchestrator
│   │   ├── parsers/          # pdfplumber / PyMuPDF & python-docx parsers
│   │   ├── extractors/       # Candidate, Skill, and Experience extractors
│   │   ├── models/           # Pydantic schema definitions
│   │   ├── utils/            # Master skills dictionary (JSON), RapidFuzz normalizer, text cleaner
│   │   ├── uploads/          # Temporary raw uploads
│   │   ├── output/           # Parsed JSON output files
│   │   ├── tests/            # Pytest test cases
│   │   └── main.py           # FastAPI entrypoint
│   └── requirements.txt
│
├── sample_resumes/           # Sample test resumes
├── docs/                     # Documentation
└── README.md
```

---

## 🚀 Getting Started

### 1. Backend Setup (FastAPI)

```bash
cd backend

# Create virtual environment (optional)
python -m venv venv
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI dev server
python app/main.py
```
Backend will run at: `http://127.0.0.1:8000`  
Interactive API Docs: `http://127.0.0.1:8000/docs`

---

### 2. Frontend Setup (React + Vite)

```bash
cd frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev
```
Frontend will run at: `http://localhost:5173`

---

## 📡 REST API Specifications

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **POST** | `/upload-resume` | Upload PDF/DOCX file and returns structured resume JSON |
| **GET** | `/skills` | Returns categorized skills, confidence scores, and frequencies |
| **GET** | `/resume-summary` | Returns candidate profile, education, projects, and statistics |
| **GET** | `/health` | Health check endpoint |

---

## 📊 Sample Output JSON Schema

```json
{
  "candidate": {
    "name": "John Doe",
    "email": "john.doe@example.com",
    "phone": "+1 555-019-2831",
    "linkedin": "linkedin.com/in/johndoe",
    "github": "github.com/johndoe",
    "portfolio": "https://johndoe.dev",
    "location": "San Francisco, CA"
  },
  "education": {
    "college": "State University",
    "degree": "Bachelor of Technology",
    "branch": "Computer Science",
    "cgpa": "3.85",
    "graduation_year": "2022"
  },
  "projects": [
    {
      "title": "Adaptive Interview System",
      "description": "Developed a local resume intelligence service with Python, FastAPI, and RapidFuzz."
    }
  ],
  "experience": [],
  "certifications": [],
  "skills": {
    "programming_languages": ["Python", "Java", "JavaScript", "C++"],
    "frontend": ["React", "HTML", "CSS", "Redux"],
    "backend": ["Node.js", "Express.js", "FastAPI", "Django"],
    "databases": ["MongoDB", "PostgreSQL", "Redis"],
    "devops": ["Docker", "Kubernetes", "Git"],
    "cloud": ["AWS"],
    "ai_ml": ["PyTorch", "TensorFlow", "scikit-learn", "Pandas"]
  },
  "skill_details": [
    {
      "skill": "Python",
      "confidence": 0.98,
      "category": "programming_languages",
      "frequency": 3
    }
  ],
  "statistics": {
    "total_skills": 18,
    "category_counts": {
      "programming_languages": 4,
      "frontend": 4,
      "backend": 4,
      "databases": 3,
      "devops": 3,
      "cloud": 1,
      "ai_ml": 4
    },
    "top_skills": ["Python", "React", "Node.js", "Docker", "PyTorch"],
    "average_confidence": 0.96,
    "experience_level": "Intermediate"
  }
}
```
