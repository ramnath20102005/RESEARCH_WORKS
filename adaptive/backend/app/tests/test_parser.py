import sys
from pathlib import Path

app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from utils.text_cleaner import clean_text, split_sections
from utils.normalizer import normalize_skill
from extractors.skill_extractor import extract_skills
from extractors.experience_detector import detect_experience_level

def test_text_cleaner():
    raw = "  John   Doe \r\n\r\n  Software   Engineer  "
    cleaned = clean_text(raw)
    assert "John Doe" in cleaned
    assert "Software Engineer" in cleaned
    print("[PASS] test_text_cleaner")

def test_skill_normalizer():
    name, conf = normalize_skill("ReactJS")
    assert name == "React"
    assert conf == 1.0

    name2, conf2 = normalize_skill("python3")
    assert name2 == "Python"
    assert conf2 == 1.0
    print("[PASS] test_skill_normalizer")

def test_skill_extractor():
    sample_text = "Experienced in Python, React, Node.js, MongoDB, Docker, and PyTorch. Built apps using Python and React."
    cat_skills, details, freq_map = extract_skills(sample_text)
    
    assert "Python" in cat_skills["programming_languages"]
    assert "React" in cat_skills["frontend"]
    assert "MongoDB" in cat_skills["databases"]
    assert freq_map["Python"] == 2
    assert freq_map["React"] == 2
    print("[PASS] test_skill_extractor")

def test_experience_detector():
    exp = detect_experience_level("Senior Software Architect with 6+ years experience", [], [], 20)
    assert exp == "Advanced"
    print("[PASS] test_experience_detector")

if __name__ == "__main__":
    test_text_cleaner()
    test_skill_normalizer()
    test_skill_extractor()
    test_experience_detector()
    print("\nALL TESTS PASSED SUCCESSFULLY!")
