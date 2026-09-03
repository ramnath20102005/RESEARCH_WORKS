import React from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/interview.css';

export default function InterviewDashboard({ parsedData }) {
  const navigate = useNavigate();

  if (!parsedData) {
    return (
      <div className="glass-card" style={{ textAlign: 'center' }}>
        <h2>No Resume Data</h2>
        <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>
          Please upload and parse a resume first to start the interview.
        </p>
      </div>
    );
  }

  const { projects = [], technical_skills = {}, area_of_interest = [], statistics = {} } = parsedData;

  const handleStartInterview = () => {
    navigate('/interview');
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Interview Dashboard</h1>
        <p className="page-subtitle">
          Review your resume profile and start the adaptive AI interview
        </p>
      </div>

      {/* Candidate Summary */}
      <div className="glass-card">
        <h2 style={{ marginBottom: '1.5rem', color: 'var(--accent-cyan)' }}>👤 Candidate Profile</h2>
        <div className="grid-2">
          <div>
            <h3 style={{ color: '#fff', marginBottom: '1rem' }}>Resume Summary</h3>
            <div style={{ color: 'var(--text-muted)', lineHeight: '1.8' }}>
              <p><strong>Total Projects:</strong> {statistics.total_projects || 0}</p>
              <p><strong>Total Skills:</strong> {statistics.total_skills || 0}</p>
              <p><strong>Experience Level:</strong> {statistics.experience_level || 'Beginner'}</p>
              <p><strong>Areas of Interest:</strong> {area_of_interest.length || 0}</p>
            </div>
          </div>
          <div>
            <h3 style={{ color: '#fff', marginBottom: '1rem' }}>Top Skills</h3>
            <div className="skills-wrapper">
              {(statistics.top_skills || []).slice(0, 5).map((skill, index) => (
                <span key={index} className="skill-tag">
                  {skill}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Skills Overview */}
      <div className="glass-card">
        <h2 style={{ marginBottom: '1.5rem', color: 'var(--accent-cyan)' }}>💻 Technical Skills</h2>
        <div className="grid-3">
          {Object.entries(technical_skills).slice(0, 6).map(([category, skills]) => (
            skills.length > 0 && (
              <div key={category} style={{ background: 'rgba(255, 255, 255, 0.02)', padding: '1rem', borderRadius: '8px' }}>
                <h4 style={{ color: '#fff', textTransform: 'capitalize', marginBottom: '0.5rem', fontSize: '0.95rem' }}>
                  {category.replace('_', ' ')}
                </h4>
                <div className="skills-wrapper">
                  {skills.slice(0, 4).map((skill, idx) => (
                    <span key={idx} className="skill-tag" style={{ fontSize: '0.75rem', padding: '0.3rem 0.6rem' }}>
                      {skill}
                    </span>
                  ))}
                  {skills.length > 4 && (
                    <span className="skill-tag" style={{ fontSize: '0.75rem', padding: '0.3rem 0.6rem', background: 'rgba(255, 255, 255, 0.03)' }}>
                      +{skills.length - 4} more
                    </span>
                  )}
                </div>
              </div>
            )
          ))}
        </div>
      </div>

      {/* Projects Overview */}
      <div className="glass-card">
        <h2 style={{ marginBottom: '1.5rem', color: 'var(--accent-cyan)' }}>📂 Projects Overview</h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {projects.slice(0, 3).map((project, index) => (
            <div key={index} style={{ background: 'rgba(255, 255, 255, 0.02)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-card)' }}>
              <h4 style={{ color: '#fff', marginBottom: '0.5rem' }}>{project.project_name || 'Unnamed Project'}</h4>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '0.5rem' }}>
                {project.description || 'No description available'}
              </p>
              {project.technologies && project.technologies.length > 0 && (
                <div className="skills-wrapper">
                  {project.technologies.slice(0, 4).map((tech, idx) => (
                    <span key={idx} className="skill-tag" style={{ fontSize: '0.75rem', padding: '0.3rem 0.6rem' }}>
                      {tech}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Start Interview Button */}
      <div className="glass-card" style={{ textAlign: 'center', padding: '3rem' }}>
        <h2 style={{ marginBottom: '1rem', color: '#fff' }}>Ready to Begin?</h2>
        <p style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>
          The AI interviewer will ask adaptive questions based on your skills and experience.
        </p>
        <button className="btn" onClick={handleStartInterview} style={{ fontSize: '1.1rem', padding: '1rem 2.5rem' }}>
          🎤 Start Interview
        </button>
      </div>
    </div>
  );
}
