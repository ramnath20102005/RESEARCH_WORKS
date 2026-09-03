import React from 'react';

export default function ExtractedSkillsPage({ parsedData }) {
  if (!parsedData) {
    return (
      <div className="glass-card" style={{ textAlign: 'center' }}>
        <h2>No Skills Data Available</h2>
        <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>
          Please upload a resume first to extract technical skills.
        </p>
      </div>
    );
  }

  const technical_skills = parsedData.technical_skills || parsedData.skills || {};

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Technical Skills Intelligence</h1>
        <p className="page-subtitle">
          Normalized technical competencies extracted for interview preparation
        </p>
      </div>

      <div className="grid-2">
        {Object.entries(technical_skills).map(([category, skillList]) => {
          if (!skillList || skillList.length === 0) return null;
          return (
            <div key={category} className="glass-card">
              <h3 className="category-title">{category.replace('_', ' ')}</h3>
              <div className="skills-wrapper">
                {skillList.map((skill, idx) => (
                  <div key={idx} className="skill-tag">
                    <span>{skill}</span>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

