import json
from pathlib import Path
from typing import Any, Union
from app.parsers.pdf_parser import extract_text_from_pdf
from app.parsers.docx_parser import extract_text_from_docx
from app.utils.text_cleaner import clean_text, split_sections
from app.extractors.project_extractor import extract_projects
from app.extractors.skill_extractor import extract_skills
from app.extractors.interest_extractor import extract_areas_of_interest
from app.extractors.certification_extractor import extract_certifications

class ParsingService:
    def parse_resume(self, file_path: Union[str, Path], filename: str) -> dict[str, Any]:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            raw_text = extract_text_from_pdf(path)
        elif ext in [".docx", ".doc"]:
            raw_text = extract_text_from_docx(path)
        elif ext == ".txt":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        cleaned = clean_text(raw_text)
        sections = split_sections(cleaned)

        # 1. Project Extraction
        projects = extract_projects(cleaned, sections)
        project_names_lower = {p["project_name"].strip().lower() for p in projects if p.get("project_name")}

        # 2. Area of Interest Extraction
        area_of_interest = extract_areas_of_interest(cleaned, sections)
        interest_items_lower = {aoi.strip().lower() for aoi in area_of_interest}

        # 3. Technical Skills Extraction
        categorized_skills, skill_details, freq_map = extract_skills(cleaned)
        
        # Ensure project names & interest domain items are NEVER classified as technical skills
        filtered_technical_skills = {}
        for cat, skills_list in categorized_skills.items():
            filtered = []
            for s in skills_list:
                s_lower = s.strip().lower()
                if s_lower not in project_names_lower and s_lower not in interest_items_lower:
                    filtered.append(s)
            if filtered:
                filtered_technical_skills[cat] = filtered

        # 4. Global / Professional Certifications Extraction
        certifications = extract_certifications(cleaned, sections)

        # Standardize technical_skills category keys to match expected output schema
        key_mapping = {
            "databases": "database",
            "ai_machine_learning": "ai_ml"
        }
        
        expected_skill_categories = [
            "programming_languages", "frontend", "backend", "frameworks",
            "libraries", "database", "cloud", "devops", "ai_ml",
            "tools", "version_control", "operating_systems"
        ]

        formatted_technical_skills = {cat: [] for cat in expected_skill_categories}
        for cat, skills_list in filtered_technical_skills.items():
            mapped_key = key_mapping.get(cat, cat)
            if mapped_key in formatted_technical_skills:
                formatted_technical_skills[mapped_key] = sorted(list(set(skills_list)))

        # 5. Compute Statistics
        total_projects = len(projects)
        all_extracted_skills = [s for sublist in formatted_technical_skills.values() for s in sublist]
        total_skills = len(all_extracted_skills)
        category_counts = {cat: len(skills) for cat, skills in formatted_technical_skills.items()}

        sorted_skills = sorted(skill_details, key=lambda x: (x["frequency"], x["confidence"]), reverse=True)
        top_skills = [s["skill"] for s in sorted_skills if s["skill"] in all_extracted_skills][:5]

        statistics = {
            "total_projects": total_projects,
            "total_skills": total_skills,
            "category_counts": category_counts,
            "top_skills": top_skills
        }

        output_data = {
            "projects": projects,
            "technical_skills": formatted_technical_skills,
            "area_of_interest": area_of_interest,
            "certifications": certifications,
            "statistics": statistics
        }

        # Save result JSON to backend/app/output/
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"{path.stem}_parsed.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        return output_data

