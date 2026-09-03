from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ProjectItem(BaseModel):
    project_name: str
    description: str
    technologies: List[str] = []
    github_link: Optional[str] = ""
    live_demo: Optional[str] = ""
    role: Optional[str] = ""
    key_features: List[str] = []

class TechnicalSkillsSchema(BaseModel):
    programming_languages: List[str] = []
    frontend: List[str] = []
    backend: List[str] = []
    frameworks: List[str] = []
    libraries: List[str] = []
    database: List[str] = []
    cloud: List[str] = []
    devops: List[str] = []
    ai_ml: List[str] = []
    tools: List[str] = []
    version_control: List[str] = []
    operating_systems: List[str] = []

class CertificationItem(BaseModel):
    certificate_name: str
    issuer: Optional[str] = ""
    year: Optional[str] = ""
    credential_id: Optional[str] = ""

class StatisticsSchema(BaseModel):
    total_projects: int = 0
    total_skills: int = 0
    category_counts: Dict[str, int] = {}
    top_skills: List[str] = []

class ResumeIntelligenceResponse(BaseModel):
    projects: List[ProjectItem]
    technical_skills: TechnicalSkillsSchema
    area_of_interest: List[str]
    certifications: List[CertificationItem]
    statistics: StatisticsSchema


