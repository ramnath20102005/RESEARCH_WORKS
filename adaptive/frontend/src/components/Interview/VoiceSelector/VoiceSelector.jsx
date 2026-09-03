import React from 'react';

const VOICES = [
  { id: 'af_bella', name: 'Bella', gender: 'Female' },
  { id: 'af_sarah', name: 'Sarah', gender: 'Female' },
  { id: 'af_nova', name: 'Nova', gender: 'Female' },
  { id: 'am_michael', name: 'Michael', gender: 'Male' },
  { id: 'am_adam', name: 'Adam', gender: 'Male' }
];

export default function VoiceSelector({ selectedVoice, onVoiceChange, disabled = false }) {
  return (
    <div className="voice-selector">
      <label className="voice-label">
        Voice:
        <select
          className="voice-select"
          value={selectedVoice}
          onChange={(e) => onVoiceChange(e.target.value)}
          disabled={disabled}
        >
          {VOICES.map(voice => (
            <option key={voice.id} value={voice.id}>
              {voice.name} ({voice.gender})
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
