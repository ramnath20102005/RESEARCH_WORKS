import json
from pathlib import Path

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
