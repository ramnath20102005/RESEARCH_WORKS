import React from 'react';

export default function InterviewControls({ onStartInterview = () => {}, onEndInterview = () => {}, onNextQuestion = () => {}, onSubmitAnswer = () => {}, isInterviewStarted = false }) {
  return (
    <div className="interview-controls">
      {!isInterviewStarted ? (
        <button className="btn btn-primary" onClick={onStartInterview}>
          🎤 Start Interview
        </button>
      ) : (
        <div className="control-buttons">
          <button className="btn btn-mic" onClick={onSubmitAnswer}>
            🎤 Record Answer
          </button>
          <button className="btn btn-submit" onClick={onSubmitAnswer}>
            ✅ Submit Answer
          </button>
          <button className="btn btn-next" onClick={onNextQuestion}>
            ➡️ Next Question
          </button>
          <button className="btn btn-danger" onClick={onEndInterview}>
            ⏹️ End Interview
          </button>
        </div>
      )}
    </div>
  );
}
