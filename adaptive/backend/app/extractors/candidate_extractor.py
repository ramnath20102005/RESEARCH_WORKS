import re
from typing import Optional, Any

EMAIL_REGEX = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
PHONE_REGEX = r'(?:\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}'
LINKEDIN_REGEX = r'(?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+'
GITHUB_REGEX = r'(?:https?://)?(?:www\.)?github\.com/[a-zA-Z0-9_-]+'
URL_REGEX = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
CGPA_REGEX = r'(?:CGPA|GPA|Grade)\s*[:\-]?\s*([0-9]\.[0-9]{1,2}|10(?:\.0)?)(?:\s*/\s*(?:10|4))?'

DEGREE_KEYWORDS = [
    "Bachelor of Technology", "B.Tech", "BTech", "Bachelor of Engineering", "B.E.", "BE",
    "Master of Technology", "M.Tech", "MTech", "Master of Computer Applications", "MCA",
    "Bachelor of Computer Applications", "BCA", "Bachelor of Science", "B.Sc", "BSc",
    "Master of Science", "M.Sc", "MSc", "Ph.D", "Doctor of Philosophy"
]

BRANCH_KEYWORDS = [
    "Computer Science", "Information Technology", "Artificial Intelligence",
    "Data Science", "Software Engineering", "Electronics and Communication",
    "Electrical Engineering", "Mechanical Engineering", "Civil Engineering"
]

def extract_candidate_info(text: str, sections: dict[str, str]) -> dict[str, Any]:
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Candidate Name
    name: Optional[str] = None
    for line in lines[:5]:
        if not re.search(EMAIL_REGEX, line) and not re.search(PHONE_REGEX, line) and not re.search(r'http', line, re.I):
            if len(line.split()) <= 4 and re.match(r'^[A-Za-z\s\.\-]+$', line):
                name = line.strip()
                break

    # Contact details
    emails = re.findall(EMAIL_REGEX, text)
    email = emails[0] if emails else None

    phones = re.findall(PHONE_REGEX, text)
    phone = phones[0] if phones else None

    linkedins = re.findall(LINKEDIN_REGEX, text, re.I)
    linkedin = linkedins[0] if linkedins else None

    githubs = re.findall(GITHUB_REGEX, text, re.I)
    github = githubs[0] if githubs else None

    all_urls = re.findall(URL_REGEX, text)
    portfolio = None
    for url in all_urls:
        if 'linkedin.com' not in url.lower() and 'github.com' not in url.lower():
            portfolio = url
            break

    location_match = re.search(r'\b([A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z\s]+)\b', text)
    location = location_match.group(1) if location_match else None

    # Education Extraction
    edu_section = sections.get("education", text)
    
    degree = None
    for deg in DEGREE_KEYWORDS:
        if re.search(r'\b' + re.escape(deg) + r'\b', edu_section, re.I):
            degree = deg
            break

    branch = None
    for br in BRANCH_KEYWORDS:
        if re.search(r'\b' + re.escape(br) + r'\b', edu_section, re.I):
            branch = br
            break

    cgpa_match = re.search(CGPA_REGEX, edu_section, re.I)
    cgpa = cgpa_match.group(1) if cgpa_match else None

    year_match = re.search(r'\b(20[0-9]{2})\b', edu_section)
    grad_year = year_match.group(1) if year_match else None

    college_name = None
    college_keywords = ["Institute", "University", "College", "School of", "IIT", "NIT", "IIIT", "BITS"]
    for line in edu_section.split('\n'):
        if any(ck.lower() in line.lower() for ck in college_keywords):
            college_name = line.strip()
            break

    education = {
        "college": college_name,
        "degree": degree,
        "branch": branch,
        "cgpa": cgpa,
        "graduation_year": grad_year
    }

    # Projects Extraction
    proj_section = sections.get("projects", sections.get("personal projects", ""))
    projects = []
    if proj_section:
        proj_lines = [p.strip() for p in proj_section.split('\n') if p.strip()]
        curr_proj = None
        for p_line in proj_lines:
            if len(p_line) < 60 and not p_line.startswith('-') and not p_line.startswith('*'):
                if curr_proj:
                    projects.append(curr_proj)
                curr_proj = {"title": p_line, "description": ""}
            else:
                if curr_proj:
                    curr_proj["description"] += " " + p_line
                else:
                    curr_proj = {"title": "Project", "description": p_line}
        if curr_proj:
            projects.append(curr_proj)

    # Certifications Extraction
    cert_section = sections.get("certifications", "")
    certifications = []
    if cert_section:
        certifications = [c.strip('-* ') for c in cert_section.split('\n') if c.strip()]

    # Experience Section
    exp_section = sections.get("experience", sections.get("work experience", ""))
    experience = []
    if exp_section:
        exp_lines = [e.strip() for e in exp_section.split('\n') if e.strip()]
        experience = [{"details": "\n".join(exp_lines)}]

    return {
        "candidate": {
            "name": name,
            "email": email,
            "phone": phone,
            "linkedin": linkedin,
            "github": github,
            "portfolio": portfolio,
            "location": location
        },
        "education": education,
        "projects": projects,
        "experience": experience,
        "certifications": certifications
    }
