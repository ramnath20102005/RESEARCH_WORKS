import React, { useState, useEffect } from 'react';
import { getResearchStateApi } from '../../../services/api';
import './ResearchStateTable.css';

export default function ResearchStateTable({ sessionId, isVisible }) {
  const [researchState, setResearchState] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchResearchState = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await getResearchStateApi(sessionId);
      setResearchState(data);
    } catch (err) {
      console.error('[ResearchState] Failed to fetch research state:', err);
      setError('Failed to load research state');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (sessionId && isVisible) {
      fetchResearchState();
    }
  }, [sessionId, isVisible]);

  if (!isVisible) {
    return null;
  }

  if (loading) {
    return (
      <div className="research-state-table">
        <h3>Research State History</h3>
        <div className="loading-message">Loading research state...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="research-state-table">
        <h3>Research State History</h3>
        <div className="error-message">{error}</div>
      </div>
    );
  }

  if (!researchState || !researchState.research_state_history || researchState.research_state_history.length === 0) {
    return (
      <div className="research-state-table">
        <h3>Research State History</h3>
        {researchState && researchState.behavioral_status && (
          <div className="behavioral-status">
            <span className="status-label">Behavioral features:</span>
            <span className="status-value">{researchState.behavioral_status}</span>
          </div>
        )}
        <div className="no-data-message">No research state data available yet. Complete a question-answer cycle to see the 11-feature vector and policy history.</div>
      </div>
    );
  }

  const history = researchState.research_state_history;

  return (
    <div className="research-state-table">
      <h3>Research State History</h3>

      {researchState.behavioral_status && (
        <div className="behavioral-status">
          <span className="status-label">Behavioral features:</span>
          <span className="status-value">{researchState.behavioral_status}</span>
        </div>
      )}

      <div className="table-container">
        <table className="research-table">
          <thead>
            <tr>
              <th>Q</th>
              <th>Corr</th>
              <th>Cov</th>
              <th>Reas</th>
              <th>Miss</th>
              <th>Eng</th>
              <th>Conf</th>
              <th>Hes</th>
              <th>Eye</th>
              <th>Diff</th>
              <th>Str</th>
              <th>WStr</th>
              <th>Policy</th>
            </tr>
          </thead>
          <tbody>
            {history.map((state, index) => {
              const f = state.features || {};
              const getVal = (key, decimals = 0) => {
                const val = f[key];
                return val !== undefined && val !== null ? val.toFixed(decimals) : '-';
              };
              return (
                <tr key={index}>
                  <td>Q{state.question_number}</td>
                  <td>{getVal("Correctness Score", 0)}</td>
                  <td>{getVal("Concept Coverage", 0)}</td>
                  <td>{getVal("Reasoning Score", 0)}</td>
                  <td>{getVal("Missing Concepts", 0)}</td>
                  <td>{getVal("Engagement Score", 2)}</td>
                  <td>{getVal("Confidence Score", 2)}</td>
                  <td>{getVal("Hesitation Score", 2)}</td>
                  <td>{getVal("Eye Contact Score", 2)}</td>
                  <td>{getVal("Difficulty", 0)}</td>
                  <td>{getVal("Correct Streak", 0)}</td>
                  <td>{getVal("Wrong Streak", 0)}</td>
                  <td className="policy-cell">{state.selected_policy || '-'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="table-legend">
        <div className="legend-title">Feature Legend:</div>
        <div className="legend-items">
          <span>Corr=Correctness Score, Cov=Concept Coverage, Reas=Reasoning Score, Miss=Missing Concepts</span>
          <span>Eng=Engagement Score, Conf=Confidence Score, Hes=Hesitation Score, Eye=Eye Contact Score</span>
          <span>Diff=Difficulty, Str=Correct Streak, WStr=Wrong Streak</span>
        </div>
      </div>
    </div>
  );
}
