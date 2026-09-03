import re
from typing import Any, Dict, List, Tuple
from app.utils.master_skills import get_master_skills
from app.utils.normalizer import normalize_skill, ALIASES

def extract_skills(text: str) -> Tuple[Dict[str, List[str]], List[Dict[str, Any]], Dict[str, int]]:
    """
    Intelligently detects technical skills from text (skills section, project descriptions, certification names).
    Normalizes skills (e.g. ReactJS -> React, NodeJS -> Node.js, Javascript -> JavaScript).
    Categorizes skills into the 12 standard technical groups:
    - programming_languages
    - frontend
    - backend
    - frameworks
    - libraries
    - databases
    - cloud
    - devops
    - ai_machine_learning
    - tools
    - version_control
    - operating_systems
    """
    master = get_master_skills()
    found_skills_detail: Dict[str, Dict[str, Any]] = {}

    # 1. Exact alias & keyword detection
    for alias_raw, canonical in ALIASES.items():
        pattern = r'(?i)\b' + re.escape(alias_raw) + r'\b'
        matches = list(re.finditer(pattern, text))
        if matches:
            count = len(matches)
            norm_name, conf = normalize_skill(canonical)
            
            # Find category for canonical skill in master skills
            cat_found = "tools"
            for cat, s_list in master.items():
                if any(s.lower() == norm_name.lower() or s.lower() == canonical.lower() for s in s_list):
                    cat_found = cat
                    break

            if norm_name not in found_skills_detail:
                found_skills_detail[norm_name] = {
                    "skill": norm_name,
                    "confidence": conf,
                    "category": cat_found,
                    "frequency": count
                }
            else:
                found_skills_detail[norm_name]["frequency"] += count

    # 2. Master skills direct check
    for category, skill_list in master.items():
        for master_skill in skill_list:
            escaped = re.escape(master_skill)
            pattern = r'(?i)\b' + escaped + r'\b'
            matches = list(re.finditer(pattern, text))
            count = len(matches)

            if count > 0:
                normalized_name, base_conf = normalize_skill(master_skill)
                conf = min(1.0, round(base_conf + (0.02 * (count - 1)), 2))

                if normalized_name not in found_skills_detail:
                    found_skills_detail[normalized_name] = {
                        "skill": normalized_name,
                        "confidence": conf,
                        "category": category,
                        "frequency": count
                    }
                else:
                    found_skills_detail[normalized_name]["frequency"] += count

    # Build categorized map ensuring all 12 categories exist
    categorized: Dict[str, List[str]] = {cat: [] for cat in master.keys()}
    detailed_list = []
    freq_map = {}

    for norm_name, detail in found_skills_detail.items():
        cat = detail["category"]
        if cat not in categorized:
            categorized[cat] = []
        if norm_name not in categorized[cat]:
            categorized[cat].append(norm_name)
        detailed_list.append(detail)
        freq_map[norm_name] = detail["frequency"]

    return categorized, detailed_list, freq_map

