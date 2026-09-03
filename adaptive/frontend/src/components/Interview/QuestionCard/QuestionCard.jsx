import React from 'react';

export default function QuestionCard({ question = '', questionNumber = 1 }) {
  return (
    <div className="question-card">
      <div className="question-header">
        <span className="question-number">Q{questionNumber}</span>
        <span className="question-role">AI Interviewer</span>
      </div>
      <div className="question-content">
        {question}
      </div>
    </div>
  );
}
