import React from 'react';

export default function SkillSidebar({ skills = {} }) {
  const allSkills = Object.entries(skills).filter(([_, skillList]) => skillList.length > 0);
  
  return (
    <div className="skill-sidebar">
      <div className="skill-sidebar-header">
        <h3>💻 Resume Skills</h3>
      </div>
      <div className="skill-sidebar-content">
        {allSkills.length === 0 ? (
          <p className="no-skills">No skills available</p>
        ) : (
          allSkills.map(([category, skillList]) => (
            <div key={category} className="skill-category">
              <h4 className="skill-category-title">{category.replace('_', ' ')}</h4>
              <div className="skill-list">
                {skillList.map((skill, index) => (
                  <span key={index} className="skill-pill">
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
