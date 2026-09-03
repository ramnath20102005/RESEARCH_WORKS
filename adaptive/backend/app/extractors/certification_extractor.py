import re
from typing import List, Dict, Any

CERT_SECTION_KEYS = [
    "certifications", "global certifications", "professional certifications",
    "certificates", "global / professional certifications", "licenses & certifications",
    "global certification", "professional certification", "certification"
]

KNOWN_ISSUERS = [
    "Oracle", "AWS", "Amazon Web Services", "Google Cloud", "Google",
    "Microsoft", "Cisco", "IBM", "Coursera", "Udemy", "edX", "Meta",
    "Red Hat", "CNCF", "CompTIA", "freeCodeCamp", "HackerRank", "LinkedIn", "NVIDIA"
]

VALID_CERT_KEYWORDS = [
    "certified", "certificate", "certification", "licence", "license",
    "credential", "specialization", "course", "degree", "diploma", "proficient in",
    "developer", "architect", "administrator", "associate", "professional", "expert"
]

def extract_certifications(text: str, sections: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Extracts professional or global certifications into structured objects:
    {
        "certificate_name": "",
        "issuer": "",
        "year": "",
        "credential_id": ""
    }
    Filters out raw technology skill names (MySQL, MongoDB) listed without certification context.
    """
    cert_text = ""
    for key, val in sections.items():
        if any(ck == key.lower().strip() or ck in key.lower() for ck in CERT_SECTION_KEYS):
            if val.strip():
                cert_text = val.strip()
                break

    if not cert_text:
        # Fallback regex search in raw text
        match = re.search(
            r'(?i)(?:global\s*\/?\s*professional\s*certifications|professional\s*certifications|global\s*certifications|certifications|certificates|certification)\s*[\:\-]?\s*\n?([^\n]+(?:\n[^\n]+){0,8})',
            text
        )
        if match:
            cert_text = match.group(1).strip()

    if not cert_text:
        return []

    lines = [l.strip('-*• ') for l in cert_text.split('\n') if l.strip()]
    certifications = []

    for line in lines:
        line_lower = line.lower()
        if len(line) > 150 or line_lower.startswith(('description', 'about', 'learned', 'covered')):
            continue

        # Reject plain skill names listed without certification context
        has_issuer = any(re.search(r'\b' + re.escape(ki) + r'\b', line, re.I) for ki in KNOWN_ISSUERS)
        has_cert_kw = any(ckw in line_lower for ckw in VALID_CERT_KEYWORDS)

        if not (has_issuer or has_cert_kw):
            # Skip standalone skill names like "MySQL", "MongoDB"
            continue

        cert_obj = _parse_certification_line(line)
        if cert_obj["certificate_name"]:
            certifications.append(cert_obj)

    return certifications

def _parse_certification_line(line: str) -> Dict[str, str]:
    year_match = re.search(r'\b(20[0-9]{2}|19[0-9]{2})\b', line)
    year = year_match.group(1) if year_match else ""

    id_match = re.search(r'(?i)\b(?:credential\s*id|id|license)\b\s*[:\-]?\s*([a-zA-Z0-9_\-]+)', line)
    credential_id = id_match.group(1) if id_match else ""

    issuer = ""
    for known in KNOWN_ISSUERS:
        if re.search(r'\b' + re.escape(known) + r'\b', line, re.I):
            issuer = known
            break

    if not issuer:
        issuer_match = re.search(r'(?i)(?:issued\s*by|by|from)\s+([A-Z][a-zA-Z0-9\s]+)', line)
        if issuer_match:
            issuer = issuer_match.group(1).strip()

    clean_name = line
    if year:
        clean_name = clean_name.replace(year, '')
    if id_match:
        clean_name = clean_name.replace(id_match.group(0), '')
    
    clean_name = re.sub(r'(?i)^\s*(?:certified|certification)\s*[:\-]\s*', '', clean_name)
    clean_name = re.sub(r'[\(\)]', ' ', clean_name)
    clean_name = re.sub(r'\s+', ' ', clean_name).strip(' -:,')

    return {
        "certificate_name": clean_name if clean_name else line.strip(),
        "issuer": issuer,
        "year": year,
        "credential_id": credential_id
    }
