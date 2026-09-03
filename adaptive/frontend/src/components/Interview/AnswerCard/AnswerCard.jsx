import React from 'react';

export default function AnswerCard({ answer = '' }) {
  return (
    <div className="answer-card">
      <div className="answer-header">
        <span className="answer-role">Candidate</span>
      </div>
      <div className="answer-content">
        {answer}
      </div>
    </div>
  );
}
