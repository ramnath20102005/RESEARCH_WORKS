import React, { useState, useEffect, useRef } from 'react';

export default function Timer({ initialSeconds = 0, onTimeUpdate = () => {} }) {
  const [seconds, setSeconds] = useState(initialSeconds);
  const onTimeUpdateRef = useRef(onTimeUpdate);

  // Update ref when callback changes
  useEffect(() => {
    onTimeUpdateRef.current = onTimeUpdate;
  }, [onTimeUpdate]);

  useEffect(() => {
    const interval = setInterval(() => {
      setSeconds(prev => prev + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Call onTimeUpdate when seconds changes
  useEffect(() => {
    onTimeUpdateRef.current(seconds);
  }, [seconds]);

  const formatTime = (totalSeconds) => {
    const mins = Math.floor(totalSeconds / 60);
    const secs = totalSeconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="timer">
      <span className="timer-display">⏱️ {formatTime(seconds)}</span>
    </div>
  );
}
