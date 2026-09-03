import React, { useState } from 'react';

export default function PerformancePanel({ performanceData }) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!performanceData || performanceData.length === 0) {
    return null;
  }

  const latestCycle = performanceData[performanceData.length - 1];

  const formatTime = (ms) => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const getCycleTotal = (cycle) => {
    if (!cycle) return 0;
    return (cycle.nvidia_llm || 0) + 
           (cycle.whisper || 0) + 
           (cycle.tabpfn || 0) + 
           (cycle.kokoro || 0);
  };

  return (
    <div className="performance-panel">
      <div 
        className="performance-header"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <span className="performance-title">Performance</span>
        <span className={`performance-toggle ${isExpanded ? 'expanded' : ''}`}>
          {isExpanded ? '▼' : '▶'}
        </span>
      </div>
      
      {isExpanded && (
        <div className="performance-content">
          {/* Latest Cycle Summary */}
          <div className="performance-summary">
            <div className="performance-label">Last Question:</div>
            <div className="performance-times">
              {latestCycle.nvidia_llm && (
                <div className="performance-item">
                  <span className="performance-name">NVIDIA NIM:</span>
                  <span className="performance-value">{formatTime(latestCycle.nvidia_llm)}</span>
                </div>
              )}
              {latestCycle.whisper && (
                <div className="performance-item">
                  <span className="performance-name">Whisper:</span>
                  <span className="performance-value">{formatTime(latestCycle.whisper)}</span>
                </div>
              )}
              {latestCycle.tabpfn && (
                <div className="performance-item">
                  <span className="performance-name">TabPFN:</span>
                  <span className="performance-value">{formatTime(latestCycle.tabpfn)}</span>
                </div>
              )}
              {latestCycle.kokoro && (
                <div className="performance-item">
                  <span className="performance-name">Kokoro:</span>
                  <span className="performance-value">{formatTime(latestCycle.kokoro)}</span>
                </div>
              )}
              <div className="performance-item performance-total">
                <span className="performance-name">Total:</span>
                <span className="performance-value">{formatTime(getCycleTotal(latestCycle))}</span>
              </div>
            </div>
          </div>

          {/* All Cycles */}
          {performanceData.length > 1 && (
            <div className="performance-history">
              <div className="performance-label">History:</div>
              {performanceData.slice().reverse().map((cycle, index) => (
                <div key={index} className="performance-history-item">
                  <span className="history-cycle">Q{cycle.question_number || index + 1}:</span>
                  <span className="history-total">{formatTime(getCycleTotal(cycle))}</span>
                </div>
              ))}
            </div>
          )}

          {/* Additional Details */}
          {latestCycle.details && (
            <div className="performance-details">
              <div className="performance-label">Details:</div>
              <div className="performance-details-content">
                {latestCycle.details.model && (
                  <div className="detail-item">
                    <span className="detail-label">Model:</span>
                    <span className="detail-value">{latestCycle.details.model}</span>
                  </div>
                )}
                {latestCycle.details.voice && (
                  <div className="detail-item">
                    <span className="detail-label">Voice:</span>
                    <span className="detail-value">{latestCycle.details.voice}</span>
                  </div>
                )}
                {latestCycle.details.policy && (
                  <div className="detail-item">
                    <span className="detail-label">Policy:</span>
                    <span className="detail-value">{latestCycle.details.policy}</span>
                  </div>
                )}
                {latestCycle.details.probability && (
                  <div className="detail-item">
                    <span className="detail-label">Probability:</span>
                    <span className="detail-value">{(latestCycle.details.probability * 100).toFixed(1)}%</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
