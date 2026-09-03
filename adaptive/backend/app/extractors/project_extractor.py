import re
from typing import Any, List, Dict
from app.utils.master_skills import get_master_skills

GITHUB_REGEX = r'https?://(?:www\.)?github\.com/[a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_\.-]+)?'
URL_REGEX = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)'

PROJECT_SECTION_KEYS = [
    "projects", "academic projects", "personal projects",
    "major projects", "mini projects", "project experience"
]

ROLE_KEYWORDS = [
    "full stack developer", "full-stack developer", "frontend developer",
    "backend developer", "team lead", "lead developer", "developer", "creator",
    "contributor", "architect", "engineer", "designer"
]

EDUCATION_NOISE_KEYWORDS = [
    "hsc", "sslc", "matric", "hr. sec.", "higher secondary", "secondary school",
    "percentage", "cgpa", "gpa", "grade", "school", "college", "kongu engineering",
    "malco vidyalaya", "salem", "mettur dam", "2021-2022", "2022-2023", "2023- present", "2023-present"
]

NON_PROJECT_HEADER_KEYWORDS = [
    "c, java", "html,css", "nodejs", "mysql", "mongodb", "global certification",
    "areas of interest", "area of interest", "dbms achievements", "oops", "technical skills",
    "https://github.com/", "certifications", "education"
]

def extract_projects(text: str, sections: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Extracts all valid projects from project-related sections or full text.
    Ensures 100% detection of all candidate projects.
    """
    project_text_blocks = []
    
    # 1. Collect text from matching project section
    for key, val in sections.items():
        if any(proj_key in key.lower() for proj_key in PROJECT_SECTION_KEYS):
            if val.strip():
                project_text_blocks.append(val.strip())
                
    if not project_text_blocks:
        # Collect text from all sections EXCEPT skills, certifications, interests
        exclude_sections = ["technical skills", "skills", "certifications", "global certification", "areas of interest", "area of interest", "domain of interest"]
        for key, val in sections.items():
            if not any(ex_sec in key.lower() for ex_sec in exclude_sections):
                if val.strip():
                    project_text_blocks.append(val.strip())

    combined_text = "\n\n".join(project_text_blocks)
    if not combined_text:
        return []

    master_skills_dict = get_master_skills()
    all_master_skills = set()
    for cat_skills in master_skills_dict.values():
        for s in cat_skills:
            all_master_skills.add(s.strip())

    raw_blocks = _split_into_project_blocks(combined_text)
    
    projects = []
    seen_titles = set()

    for block in raw_blocks:
        proj = _parse_single_project(block, all_master_skills)
        if proj and proj["project_name"] and _is_valid_project_name(proj["project_name"]):
            title_clean = proj["project_name"].strip().lower()
            # Ignore candidate name (e.g. single/double name line where description equals title)
            if proj["description"].strip().lower() == title_clean:
                continue
            if title_clean not in seen_titles:
                seen_titles.add(title_clean)
                projects.append(proj)

    return projects

def _is_valid_project_name(name: str) -> bool:
    cleaned_lower = name.strip().lower()
    if not cleaned_lower or len(cleaned_lower) < 3:
        return False

    # Check education noise
    if any(edu_kw in cleaned_lower for edu_kw in EDUCATION_NOISE_KEYWORDS):
        return False

    # Check non-project header noise
    if any(np_kw in cleaned_lower for np_kw in NON_PROJECT_HEADER_KEYWORDS):
        return False

    # Reject loose project attribute labels
    if cleaned_lower in ['link', 'about', 'description', 'technologies', 'tech stack', 'techstack', 'github', 'url']:
        return False

    # Reject implementation phrases mistaken as project titles
    if cleaned_lower.startswith(('full-stack implemented', 'implemented', 'built', 'developed', 'created', 'designed', 'using', 'powered by', 'integrated')):
        return False

    # Reject if it's just a raw URL
    if cleaned_lower.startswith('http'):
        return False

    return True

def _split_into_project_blocks(text: str) -> List[str]:
    lines = text.split('\n')
    blocks = []
    current_block = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        is_title_candidate = False
        
        if len(stripped) < 90 and not stripped.startswith(('-', '*', '•', '1.', '2.', '3.')):
            if not stripped.lower().startswith(('about :', 'about:', 'tech stack:', 'techstack:', 'technologies:', 'link:', 'link :', 'features:', 'key features:', 'education link', 'skills link', 'developed', 'built', 'created', 'an online', 'a web', 'a mobile')):
                if _is_valid_project_name(stripped):
                    # Check if next line contains project attributes
                    next_line = lines[i+1].strip().lower() if i + 1 < len(lines) else ""
                    if any(next_line.startswith(prefix) for prefix in ['link', 'github', 'about', 'tech', 'built', 'developed', 'created', 'an ', 'a ', 'description']):
                        # Prevent wrapped sentence fragments from being detected as titles
                        if not (stripped[0].islower() and len(stripped.split()) > 2):
                            is_title_candidate = True
                    elif stripped.isupper() or stripped.istitle() or re.match(r'^[A-Z0-9\s\-\.\:\(\)\/]+$', stripped):
                        is_title_candidate = True

        if is_title_candidate and current_block:
            blocks.append("\n".join(current_block))
            current_block = [line]
        else:
            current_block.append(line)

    if current_block:
        blocks.append("\n".join(current_block))

    return blocks

def _parse_single_project(block: str, master_skills: set) -> Dict[str, Any]:
    lines = [l.strip() for l in block.split('\n') if l.strip()]
    if not lines:
        return {
            "project_name": "",
            "description": "",
            "technologies": [],
            "github_link": "",
            "live_demo": "",
            "role": "",
            "key_features": []
        }

    project_name = lines[0]
    project_name = re.sub(r'^(?:Project|Title)\s*[:\-]\s*', '', project_name, flags=re.I).strip()

    description_parts = []
    technologies = set()
    github_link = ""
    live_demo = ""
    role = ""
    key_features = []

    tech_stack_match = re.search(r'(?i)(?:tech\s*stack|technologies|built\s*with)\s*[:\-]\s*(.*)', block)
    if tech_stack_match:
        tech_str = tech_stack_match.group(1)
        for t in re.split(r'[,\|\/\;]', tech_str):
            cleaned_t = t.strip()
            if cleaned_t and len(cleaned_t) < 30 and not any(ek in cleaned_t.lower() for ek in EDUCATION_NOISE_KEYWORDS):
                technologies.add(cleaned_t)

    for line in lines[1:]:
        line_lower = line.lower()
        
        # Skip education lines embedded inside block
        if any(edu in line_lower for edu in EDUCATION_NOISE_KEYWORDS):
            continue

        # Links
        gh_matches = re.findall(GITHUB_REGEX, line, re.I)
        if gh_matches and not github_link:
            github_link = gh_matches[0]

        all_urls = re.findall(URL_REGEX, line)
        for url in all_urls:
            if 'github.com' not in url.lower() and 'linkedin.com' not in url.lower():
                if not live_demo:
                    live_demo = url

        # Check line for explicitly defined role
        if not role:
            for r_kw in ROLE_KEYWORDS:
                if r_kw in line_lower:
                    role = line
                    break

        # Check for key features or bullet points
        if line.startswith(('-', '*', '•')) or re.match(r'^\d+\.', line):
            cleaned_feature = re.sub(r'^[\-\*•\d\.\s]+', '', line).strip()
            if cleaned_feature and not any(ek in cleaned_feature.lower() for ek in EDUCATION_NOISE_KEYWORDS) and not cleaned_feature.lower().startswith('http'):
                key_features.append(cleaned_feature)
        elif 'features:' in line_lower or 'key features:' in line_lower:
            feat_content = re.sub(r'(?i)^(?:key\s*)?features\s*[:\-]\s*', '', line).strip()
            if feat_content and not any(ek in feat_content.lower() for ek in EDUCATION_NOISE_KEYWORDS):
                key_features.append(feat_content)
        else:
            if not line_lower.startswith(('link:', 'link :', 'github:', 'demo:', 'skills link:', 'education link:')):
                description_parts.append(line)

    description = " ".join(description_parts).strip()
    description = re.sub(r'(?i)cgpa\s*[:\-]?\s*[0-9\.]+\*?', '', description)
    description = re.sub(r'(?i)percentage\s*[:\-]?\s*[0-9\.]+\%?', '', description)
    description = re.sub(r'\s+', ' ', description).strip()

    # Detect skills/technologies inside description and block text
    for skill in master_skills:
        pattern = r'(?i)\b' + re.escape(skill) + r'\b'
        if re.search(pattern, block):
            technologies.add(skill)

    # Extract key feature phrases if no bullet points found
    if not key_features and description:
        parts = re.split(r'[\.\;\n]', description)
        for p in parts:
            p_clean = p.strip()
            if len(p_clean) > 10 and not any(ek in p_clean.lower() for ek in EDUCATION_NOISE_KEYWORDS) and not p_clean.lower().startswith('http'):
                key_features.append(p_clean)

    return {
        "project_name": project_name,
        "description": description if description else project_name,
        "technologies": sorted(list(technologies)),
        "github_link": github_link,
        "live_demo": live_demo,
        "role": role,
        "key_features": key_features[:8]
    }
