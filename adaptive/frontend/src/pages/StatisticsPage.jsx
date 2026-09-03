import React from 'react';

export default function StatisticsPage({ parsedData }) {
  if (!parsedData || !parsedData.statistics) {
    return (
      <div className="glass-card" style={{ textAlign: 'center' }}>
        <h2>No Statistics Available</h2>
        <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>
          Upload a resume to generate skill analytics and experience levels.
        </p>
      </div>
    );
  }

  const { statistics } = parsedData;

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Skill Analytics & Statistics</h1>
        <p className="page-subtitle">Heuristic experience evaluation and metrics</p>
      </div>

      <div className="grid-3">
        <div className="glass-card" style={{ textAlign: 'center' }}>
          <div style={{ color: 'var(--text-muted)' }}>Total Skills Detected</div>
          <div className="stat-value">{statistics.total_skills}</div>
        </div>

        <div className="glass-card" style={{ textAlign: 'center' }}>
          <div style={{ color: 'var(--text-muted)' }}>Experience Level</div>
          <div className="stat-value" style={{ fontSize: '2rem' }}>{statistics.experience_level}</div>
        </div>

        <div className="glass-card" style={{ textAlign: 'center' }}>
          <div style={{ color: 'var(--text-muted)' }}>Average Confidence</div>
          <div className="stat-value">{(statistics.average_confidence * 100).toFixed(0)}%</div>
        </div>
      </div>

      <div className="glass-card">
        <h3 style={{ color: 'var(--accent-cyan)', marginBottom: '1rem' }}>Top Skills</h3>
        <div className="skills-wrapper">
          {statistics.top_skills.map((skill, idx) => (
            <div key={idx} className="skill-tag" style={{ border: '1px solid var(--accent-cyan)' }}>
              <span>#{idx + 1} {skill}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="glass-card">
        <h3 style={{ color: 'var(--accent-cyan)', marginBottom: '1rem' }}>Category Distribution</h3>
        <div className="grid-2">
          {Object.entries(statistics.category_counts).map(([cat, count]) => (
            <div key={cat} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid var(--border-card)' }}>
              <span style={{ textTransform: 'capitalize' }}>{cat.replace('_', ' ')}</span>
              <span style={{ fontWeight: 'bold', color: 'var(--accent-green)' }}>{count}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
