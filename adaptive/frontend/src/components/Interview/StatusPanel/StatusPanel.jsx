import React from 'react';

export default function StatusPanel({ questionNumber = 1, difficulty = 'Medium', elapsedTime = '00:00', currentTopic = 'Python' }) {
  return (
    <div className="status-panel">
      <div className="status-header">
        <h3>📊 Interview Status</h3>
      </div>
      <div className="status-content">
        <div className="status-row">
          <span className="status-label">Question:</span>
          <span className="status-value">{questionNumber} / 10</span>
        </div>
        <div className="status-row">
          <span className="status-label">Difficulty:</span>
          <span className={`status-value difficulty-${difficulty.toLowerCase()}`}>{difficulty}</span>
        </div>
        <div className="status-row">
          <span className="status-label">Elapsed Time:</span>
          <span className="status-value">{elapsedTime}</span>
        </div>
        <div className="status-row">
          <span className="status-label">Current Topic:</span>
          <span className="status-value">{currentTopic}</span>
        </div>
      </div>
    </div>
  );
}
