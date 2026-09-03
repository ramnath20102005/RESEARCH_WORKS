import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import FileUpload from '../components/FileUpload';
import { uploadResumeApi } from '../services/api';

export default function UploadPage({ setParsedData }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleUpload = async (file) => {
    setLoading(true);
    setError(null);
    try {
      const data = await uploadResumeApi(file);
      setParsedData(data);
      navigate('/extracted');
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to upload and parse resume.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Resume Parsing & Skill Extraction</h1>
        <p className="page-subtitle">
          Local, deterministic NLP intelligence service for candidate resumes.
        </p>
      </div>

      {error && (
        <div className="glass-card" style={{ borderLeft: '4px solid #ef4444', color: '#f87171' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      <FileUpload onUpload={handleUpload} isLoading={loading} />
    </div>
  );
}
