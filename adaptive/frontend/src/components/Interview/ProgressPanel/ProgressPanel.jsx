import React from 'react';

export default function ProgressPanel({ currentQuestion = 1, totalQuestions = 10, progress = 10 }) {
  return (
    <div className="progress-panel">
      <div className="progress-header">
        <h3>📈 Interview Progress</h3>
      </div>
      <div className="progress-content">
        <div className="progress-bar-container">
          <div className="progress-bar" style={{ width: `${progress}%` }}></div>
        </div>
        <div className="progress-text">
          <span>{currentQuestion} of {totalQuestions} questions</span>
          <span>{progress}% complete</span>
        </div>
      </div>
    </div>
  );
}
