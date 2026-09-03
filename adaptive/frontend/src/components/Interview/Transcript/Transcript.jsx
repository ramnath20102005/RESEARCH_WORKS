import React from 'react';

export default function Transcript({ messages = [] }) {
  return (
    <div className="transcript">
      <div className="transcript-header">
        <h3>📝 Full Transcript</h3>
      </div>
      <div className="transcript-content">
        {messages.length === 0 ? (
          <p className="transcript-empty">No transcript available yet</p>
        ) : (
          messages.map((msg, index) => (
            <div key={index} className={`transcript-entry ${msg.role}`}>
              <span className="transcript-role">{msg.role === 'ai' ? 'AI Interviewer' : 'Candidate'}</span>
              <span className="transcript-text">{msg.content}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
