import re

def clean_text(text: str) -> str:
    """
    Cleans raw resume text by removing non-standard characters, repeated whitespace, and noise.
    """
    if not text:
        return ""
    # Normalize newline sequences
    text = re.sub(r'\r\n|\r', '\n', text)
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Replace multiple spaces/tabs with single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Reduce consecutive newlines to max 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def split_sections(text: str) -> dict[str, str]:
    """
    Identifies common resume sections using regex rules.
    """
    headers = [
        "academic projects", "personal projects", "major projects", "mini projects",
        "project experience", "projects",
        "technical skills", "skills", "tech stack", "technical competencies", "soft skills",
        "areas of interest", "area of interest", "domain of interest", "domains of interest",
        "interested technologies", "interests",
        "global / professional certifications", "global certifications",
        "professional certifications", "global certification", "certifications", "certification", "licenses & certifications", "certificates",
        "education", "sslc", "hsc", "academic details", "qualifications",
        "experience", "work experience", "employment", "achievements", "dbms achievements", "awards", "summary", "objective", "career objective", "extra-curricular", "co-curricular"
    ]
    
    # Sort headers by length descending to match longest phrase first
    headers_sorted = sorted(headers, key=len, reverse=True)
    pattern = r'(?i)^\s*(?:' + '|'.join(re.escape(h) for h in headers_sorted) + r')\s*[\:\-]?\s*$'
    lines = text.split('\n')
    sections = {}
    current_section = "general"
    sections[current_section] = []

    for line in lines:
        stripped = line.strip()
        if re.match(pattern, stripped):
            current_section = stripped.lower().replace(':', '').replace('-', '').strip()
            if current_section not in sections:
                sections[current_section] = []
        else:
            sections[current_section].append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items() if v}
