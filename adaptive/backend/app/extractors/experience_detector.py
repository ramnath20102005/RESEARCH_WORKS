import re

def detect_experience_level(text: str, projects: list, experience: list, total_skills: int) -> str:
    """
    Determines Beginner, Intermediate, or Advanced experience based on heuristics.
    No LLMs used.
    """
    lower_text = text.lower()
    
    # Keyword indicators
    senior_keywords = ["senior", "lead", "architect", "principal", "head of", "manager", "5+ years", "6+ years", "7+ years"]
    inter_keywords = ["intermediate", "mid-level", "2+ years", "3+ years", "4+ years", "associate"]
    
    for kw in senior_keywords:
        if kw in lower_text:
            return "Advanced"

    for kw in inter_keywords:
        if kw in lower_text:
            return "Intermediate"

    # Count projects and experience length
    num_projects = len(projects)
    has_exp = len(experience) > 0

    if has_exp and (num_projects >= 3 or total_skills >= 15):
        return "Intermediate"
    elif num_projects >= 4 and total_skills >= 12:
        return "Intermediate"
    
    return "Beginner"
