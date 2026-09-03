import sys
from pathlib import Path

# Add app directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.parsing_service import ParsingService

def test_resume_intelligence():
    sample_text = """
    VICKKASH R
    
    Fake News Prediction Using Machine Learning
    Developed a fake news detection system using ML algorithms (Naive Bayes, Passive Aggressive Classifier)
    Technologies: Machine Learning, Python, scikit-learn
    GITHUB: https://github.com/Vickkash/Fake-News-Prediction

    E-COMMERCE MARKETPLACE
    businesses to sell products directly without middlemen.
    Full-Stack Implemented using React Native
    Tech Stack: Node.js, Firebase (DB), Razorpay, Nodemailer (emails).
    GITHUB: https://github.com/Vickkash/Kaithiran-web.git

    TECHNICAL SKILLS
    C, Java, JavaScript, Python, SQL, React, React Native, Firebase, Oracle, Deep Learning, NLP, OpenCV, TensorFlow, Git, GitHub

    AREAS OF INTEREST
    Machine Learning, Artificial Intelligence, Web Development, Cloud Computing, Cyber Security

    CERTIFICATION
    AWS Certified Solutions Architect
    """

    sample_file = Path(__file__).parent / "test_resume.txt"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(sample_text)

    service = ParsingService()
    result = service.parse_resume(sample_file, "test_resume.txt")

    print("--- PARSED RESULT KEYS ---")
    print(list(result.keys()))

    print("\n--- PROJECTS ---")
    proj_names = [p["project_name"] for p in result.get("projects", [])]
    for proj in result.get("projects", []):
        print("Project Name:", proj.get("project_name"))
        print("Description:", proj.get("description"))
        print("Technologies:", proj.get("technologies"))
        print("GitHub:", proj.get("github_link"))
        print("-")

    print("\n--- TECHNICAL SKILLS ---")
    print(result.get("technical_skills"))

    print("\n--- AREA OF INTEREST ---")
    print(result.get("area_of_interest"))

    print("\n--- CERTIFICATIONS ---")
    print(result.get("certifications"))

    # Assertions
    # 1. Multiple Projects must be detected
    assert len(proj_names) >= 2
    assert "Fake News Prediction Using Machine Learning" in proj_names
    assert "E-COMMERCE MARKETPLACE" in proj_names

    # 2. Area of Interest must NOT contain GitHub URLs, implementation sentences, specific tools, or section headers
    interests = result["area_of_interest"]
    for item in interests:
        item_lower = item.lower()
        assert "github" not in item_lower
        assert "http" not in item_lower
        assert ".git" not in item_lower
        assert "certification" not in item_lower
        assert "businesses to sell" not in item_lower
        assert "implemented" not in item_lower

    print("\n[SUCCESS] Multiple projects detected & Area of Interest 100% clean!")

if __name__ == "__main__":
    test_resume_intelligence()
