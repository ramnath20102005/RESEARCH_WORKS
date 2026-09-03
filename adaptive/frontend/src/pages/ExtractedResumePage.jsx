import React from 'react';
import { useNavigate } from 'react-router-dom';

export default function ExtractedResumePage({ parsedData }) {
  const navigate = useNavigate();

  if (!parsedData) {
    return (
      <div className="glass-card" style={{ textAlign: 'center' }}>
        <h2>No Resume Intelligence Data</h2>
        <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>
          Please upload a resume on the Upload page to view structured section intelligence.
        </p>
      </div>
    );
  }

  const { projects = [], technical_skills = {}, area_of_interest = [], certifications = [] } = parsedData;

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Resume Intelligence Engine</h1>
        <p className="page-subtitle">Structured section extraction tailored for AI Interview System</p>
      </div>

      {/* 1. Projects Section */}
      <div className="glass-card">
        <h2 style={{ marginBottom: '1.25rem', color: 'var(--accent-cyan)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          📂 Projects ({projects.length})
        </h2>
        {projects && projects.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            {projects.map((proj, idx) => (
              <div
                key={idx}
                style={{
                  background: 'rgba(255, 255, 255, 0.03)',
                  border: '1px solid var(--border-card)',
                  borderRadius: '10px',
                  padding: '1.25rem'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '0.5rem' }}>
                  <h3 style={{ color: '#fff', margin: 0, fontSize: '1.2rem' }}>{proj.project_name || 'Unnamed Project'}</h3>
                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    {proj.github_link && (
                      <a
                        href={proj.github_link}
                        target="_blank"
                        rel="noreferrer"
                        style={{
                          fontSize: '0.85rem',
                          background: 'rgba(56, 189, 248, 0.15)',
                          color: 'var(--accent-cyan)',
                          padding: '0.25rem 0.6rem',
                          borderRadius: '6px',
                          textDecoration: 'none',
                          border: '1px solid rgba(56, 189, 248, 0.3)'
                        }}
                      >
                        GitHub Link ↗
                      </a>
                    )}
                    {proj.live_demo && (
                      <a
                        href={proj.live_demo}
                        target="_blank"
                        rel="noreferrer"
                        style={{
                          fontSize: '0.85rem',
                          background: 'rgba(168, 85, 247, 0.15)',
                          color: '#c084fc',
                          padding: '0.25rem 0.6rem',
                          borderRadius: '6px',
                          textDecoration: 'none',
                          border: '1px solid rgba(168, 85, 247, 0.3)'
                        }}
                      >
                        Live Demo ↗
                      </a>
                    )}
                  </div>
                </div>

                {proj.role && (
                  <p style={{ color: 'var(--accent-cyan)', fontSize: '0.9rem', marginTop: '0.4rem', fontWeight: '500' }}>
                    Role: {proj.role}
                  </p>
                )}

                <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', marginTop: '0.6rem', lineHeight: '1.5' }}>
                  {proj.description}
                </p>

                {proj.technologies && proj.technologies.length > 0 && (
                  <div style={{ marginTop: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                    <strong style={{ fontSize: '0.85rem', color: '#cbd5e1' }}>Technologies:</strong>
                    {proj.technologies.map((tech, tIdx) => (
                      <span
                        key={tIdx}
                        style={{
                          fontSize: '0.8rem',
                          background: 'rgba(255, 255, 255, 0.08)',
                          color: '#f1f5f9',
                          padding: '0.2rem 0.5rem',
                          borderRadius: '4px'
                        }}
                      >
                        {tech}
                      </span>
                    ))}
                  </div>
                )}

                {proj.key_features && proj.key_features.length > 0 && (
                  <div style={{ marginTop: '0.75rem' }}>
                    <strong style={{ fontSize: '0.85rem', color: '#cbd5e1' }}>Key Features:</strong>
                    <ul style={{ marginTop: '0.4rem', paddingLeft: '1.25rem', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                      {proj.key_features.map((feat, fIdx) => (
                        <li key={fIdx} style={{ marginBottom: '0.2rem' }}>{feat}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <p style={{ color: 'var(--text-muted)' }}>No projects extracted.</p>
        )}
      </div>

      {/* 2. Technical Skills Section */}
      <div className="glass-card">
        <h2 style={{ marginBottom: '1.25rem', color: 'var(--accent-cyan)' }}>💻 Technical Skills</h2>
        {technical_skills && Object.keys(technical_skills).length > 0 ? (
          <div className="grid-2">
            {Object.entries(technical_skills).map(([cat, skills]) => (
              <div key={cat} style={{ background: 'rgba(255, 255, 255, 0.02)', padding: '1rem', borderRadius: '8px' }}>
                <h4 style={{ color: '#fff', textTransform: 'capitalize', marginBottom: '0.5rem' }}>{cat.replace('_', ' ')}</h4>
                <div className="skills-wrapper">
                  {skills.map((skill, sIdx) => (
                    <span key={sIdx} className="skill-tag">
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p style={{ color: 'var(--text-muted)' }}>No technical skills extracted.</p>
        )}
      </div>

      {/* 3. Area of Interest Section */}
      <div className="glass-card">
        <h2 style={{ marginBottom: '1rem', color: 'var(--accent-cyan)' }}>🎯 Area of Interest</h2>
        {area_of_interest && area_of_interest.length > 0 ? (
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            {area_of_interest.map((interest, iIdx) => (
              <span
                key={iIdx}
                style={{
                  background: 'rgba(56, 189, 248, 0.1)',
                  border: '1px solid rgba(56, 189, 248, 0.25)',
                  color: '#e0f2fe',
                  padding: '0.4rem 0.8rem',
                  borderRadius: '20px',
                  fontSize: '0.9rem'
                }}
              >
                {interest}
              </span>
            ))}
          </div>
        ) : (
          <p style={{ color: 'var(--text-muted)' }}>No areas of interest extracted.</p>
        )}
      </div>

      {/* 4. Global / Professional Certifications Section */}
      <div className="glass-card">
        <h2 style={{ marginBottom: '1rem', color: 'var(--accent-cyan)' }}>📜 Global / Professional Certifications</h2>
        {certifications && certifications.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {certifications.map((cert, cIdx) => (
              <div
                key={cIdx}
                style={{
                  background: 'rgba(255, 255, 255, 0.02)',
                  border: '1px solid var(--border-card)',
                  borderRadius: '8px',
                  padding: '0.85rem 1rem',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  flexWrap: 'wrap',
                  gap: '0.5rem'
                }}
              >
                <div>
                  <h4 style={{ color: '#fff', margin: 0 }}>
                    {typeof cert === 'string' ? cert : cert.certificate_name}
                  </h4>
                  {typeof cert === 'object' && cert.issuer && (
                    <span style={{ fontSize: '0.85rem', color: 'var(--accent-cyan)', marginTop: '0.2rem', display: 'inline-block' }}>
                      Issuer: {cert.issuer}
                    </span>
                  )}
                </div>
                {typeof cert === 'object' && (
                  <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    {cert.year && (
                      <span style={{ fontSize: '0.8rem', background: 'rgba(255, 255, 255, 0.08)', color: '#cbd5e1', padding: '0.2rem 0.5rem', borderRadius: '4px' }}>
                        {cert.year}
                      </span>
                    )}
                    {cert.credential_id && (
                      <span style={{ fontSize: '0.8rem', background: 'rgba(56, 189, 248, 0.12)', color: 'var(--accent-cyan)', padding: '0.2rem 0.5rem', borderRadius: '4px' }}>
                        ID: {cert.credential_id}
                      </span>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <p style={{ color: 'var(--text-muted)' }}>No certifications extracted.</p>
        )}
      </div>

      {/* Start Interview Button */}
      <div className="glass-card" style={{ textAlign: 'center', padding: '2rem' }}>
        <h2 style={{ marginBottom: '0.75rem', color: '#fff' }}>Ready for Interview?</h2>
        <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
          Start the adaptive AI interview based on your resume profile
        </p>
        <button className="btn" onClick={() => navigate('/interview-dashboard')} style={{ fontSize: '1rem', padding: '0.75rem 2rem' }}>
          🎤 Go to Interview Dashboard
        </button>
      </div>
    </div>
  );
}

