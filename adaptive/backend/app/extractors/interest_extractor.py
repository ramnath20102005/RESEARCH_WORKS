import re
from typing import List, Dict
from app.utils.master_skills import get_master_skills

INTEREST_SECTION_KEYS = [
    "areas of interest", "area of interest", "interests",
    "domain of interest", "domains of interest", "interested technologies"
]

ACHIEVEMENT_NOISE_KEYWORDS = [
    "won", "place", "prize", "paper presentation", "mad club", "kongu engineering",
    "soft skills", "time management", "problem solving", "adaptability", "team collabration",
    "hackathon", "smart-indian-hackathon", "pre-final round", "achievements", "dbms achievements"
]

SECTION_HEADER_NOISE = [
    "certification", "certifications", "projects", "academic projects", "technical skills",
    "skills", "education", "experience", "work experience", "achievements", "summary"
]

VERB_IMPLEMENTATION_NOISE = [
    "implemented", "developed", "built", "created", "businesses to sell", "directly without",
    "using", "implemented using", "github:", "link:"
]

def extract_areas_of_interest(text: str, sections: Dict[str, str]) -> List[str]:
    """
    Extracts high-level technical domain areas representing candidate interests.
    Strictly filters out URLs, GitHub links, implementation sentences, specific tools, and section headers.
    """
    interest_text = ""
    for key, val in sections.items():
        if any(ik == key.lower().strip() or ik in key.lower() for ik in INTEREST_SECTION_KEYS):
            if val.strip():
                interest_text = val.strip()
                break

    if not interest_text:
        # Fallback regex search ONLY if explicit section title exists on its own line
        match = re.search(
            r'(?i)^\s*(?:areas?\s*of\s*interest|domains?\s*of\s*interest|interested\s*technologies)\s*[\:\-]?\s*\n([^\n]+(?:\n[^\n]+){0,6})',
            text, re.MULTILINE
        )
        if match:
            interest_text = match.group(1).strip()

    if not interest_text:
        return []

    # Get master skills list to filter out standalone tools/languages from interests
    master_skills_dict = get_master_skills()
    all_master_skills_lower = set()
    for cat_skills in master_skills_dict.values():
        for s in cat_skills:
            all_master_skills_lower.add(s.strip().lower())

    lines = [l.strip('-*• ') for l in interest_text.split('\n') if l.strip()]
    interests = []

    for line in lines:
        # Split by comma or semicolon if listed inline
        items = re.split(r'[,;]\s*', line)
        for item in items:
            clean_item = item.strip()
            clean_item = re.sub(r'^(?:and|or)\s+', '', clean_item, flags=re.I).strip()
            clean_item_lower = clean_item.lower()

            # 1. Reject URLs, GitHub links, .git
            if any(url_kw in clean_item_lower for url_kw in ["http", "github", ".git", "url", "/"]):
                continue

            # 2. Reject section headers
            if clean_item_lower in SECTION_HEADER_NOISE or any(clean_item_lower == sh for sh in SECTION_HEADER_NOISE):
                continue

            # 3. Reject implementation verbs / descriptive sentences
            if any(vk in clean_item_lower for vk in VERB_IMPLEMENTATION_NOISE):
                continue

            # 4. Reject competition awards / soft skills
            if any(ak in clean_item_lower for ak in ACHIEVEMENT_NOISE_KEYWORDS):
                continue

            # (Removed restriction blocking master skills from Area of Interest)

            # 6. Reject long sentences (> 45 chars)
            if len(clean_item) > 45:
                continue

            if clean_item and len(clean_item) >= 3:
                if clean_item not in interests:
                    interests.append(clean_item)

    return interests
