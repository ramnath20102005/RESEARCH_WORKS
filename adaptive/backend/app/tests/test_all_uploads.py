import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.parsing_service import ParsingService

def test_all_uploads():
    p = ParsingService()
    uploads_dir = Path(__file__).parent.parent / "uploads"
    for pdf_file in uploads_dir.glob("*.pdf"):
        res = p.parse_resume(pdf_file, pdf_file.name)
        print("==========================================")
        print("FILE:", pdf_file.name)
        print("==========================================")
        print("PROJECTS COUNT:", len(res["projects"]))
        for proj in res["projects"]:
            print("  - Title:", proj["project_name"])
            print("    GitHub:", proj["github_link"])
            print("    Tech:", proj["technologies"])
        print("\nAREA OF INTEREST:", res["area_of_interest"])
        print("\nCERTIFICATIONS:", res["certifications"])
        print("\nTECHNICAL SKILLS:")
        for cat, skills in res["technical_skills"].items():
            if skills:
                print(f"  {cat}: {skills}")
        print("\n")

if __name__ == "__main__":
    test_all_uploads()
