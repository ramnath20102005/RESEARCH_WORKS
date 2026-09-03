import React, { useState } from 'react';

export default function FileUpload({ onUpload, isLoading }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleSubmit = () => {
    if (selectedFile) {
      onUpload(selectedFile);
    }
  };

  return (
    <div className="glass-card">
      <div
        className={`dropzone ${dragActive ? 'active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => document.getElementById('file-upload-input').click()}
      >
        <input
          id="file-upload-input"
          type="file"
          accept=".pdf,.docx,.doc"
          onChange={handleChange}
          style={{ display: 'none' }}
        />
        <div className="upload-icon">📄</div>
        <h3>{selectedFile ? selectedFile.name : 'Drag & Drop Resume Here'}</h3>
        <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>
          Supports PDF and DOCX formats (Local NLP extraction, No LLM API calls)
        </p>
      </div>

      {selectedFile && (
        <div style={{ marginTop: '1.5rem', textAlign: 'center' }}>
          <button className="btn" onClick={handleSubmit} disabled={isLoading}>
            {isLoading ? 'Parsing Resume...' : 'Parse Resume & Extract Skills'}
          </button>
        </div>
      )}
    </div>
  );
}
