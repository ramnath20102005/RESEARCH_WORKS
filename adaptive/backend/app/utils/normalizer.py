import json
from pathlib import Path
from typing import Any

# Fallback simple ratio matching if rapidfuzz is not yet installed
try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

ALIASES = {
    "js": "JavaScript",
    "java script": "JavaScript",
    "javascript": "JavaScript",
    "ts": "TypeScript",
    "type script": "TypeScript",
    "typescript": "TypeScript",
    "py": "Python",
    "python3": "Python",
    "cpp": "C++",
    "c plus plus": "C++",
    "c#": "C#",
    "c sharp": "C#",
    "node": "Node.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "express": "Express.js",
    "expressjs": "Express.js",
    "express.js": "Express.js",
    "reactjs": "React",
    "react js": "React",
    "react.js": "React",
    "react": "React",
    "vue": "Vue.js",
    "vuejs": "Vue.js",
    "angularjs": "Angular",
    "ng": "Angular",
    "nextjs": "Next.js",
    "nuxt": "Nuxt.js",
    "nuxtjs": "Nuxt.js",
    "spring": "Spring Boot",
    "springboot": "Spring Boot",
    "django": "Django",
    "fastapi": "FastAPI",
    "flask": "Flask",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "mongo": "MongoDB",
    "mongodb": "MongoDB",
    "elastic": "Elasticsearch",
    "aws": "AWS",
    "amazon web services": "AWS",
    "gcp": "Google Cloud Platform",
    "k8s": "Kubernetes",
    "k8": "Kubernetes",
    "docker": "Docker",
    "git": "Git",
    "github": "GitHub",
    "pytorch": "PyTorch",
    "torch": "PyTorch",
    "tensorflow": "TensorFlow",
    "tf": "TensorFlow",
    "sklearn": "scikit-learn",
    "scikitlearn": "scikit-learn",
    "visual studio code": "VS Code",
    "vscode": "VS Code"
}

_MASTER_SKILLS_CACHE = None

def get_master_skills():
    global _MASTER_SKILLS_CACHE
    if _MASTER_SKILLS_CACHE is None:
        json_path = Path(__file__).parent / "master_skills.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                _MASTER_SKILLS_CACHE = json.load(f)
        else:
            _MASTER_SKILLS_CACHE = {}
    return _MASTER_SKILLS_CACHE

def normalize_skill(token: str) -> tuple[str, float]:
    """
    Normalizes a skill token to standard canonical name and returns a confidence ratio (0.0 to 1.0).
    """
    cleaned = token.strip().lower()
    if cleaned in ALIASES:
        return ALIASES[cleaned], 1.0

    master = get_master_skills()
    all_canonical = []
    for cat_skills in master.values():
        all_canonical.extend(cat_skills)

    # Exact case-insensitive check against master list
    for skill in all_canonical:
        if skill.lower() == cleaned:
            return skill, 1.0

    # Fuzzy match using RapidFuzz (or fallback substring match)
    if len(cleaned) > 2:
        if HAS_RAPIDFUZZ:
            match = process.extractOne(token, all_canonical, scorer=fuzz.WRatio)
            if match and match[1] >= 85:
                return match[0], round(match[1] / 100.0, 2)
        else:
            for skill in all_canonical:
                if cleaned in skill.lower() or skill.lower() in cleaned:
                    return skill, 0.85

    return token.strip().title(), 0.70
