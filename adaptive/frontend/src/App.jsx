import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import UploadPage from './pages/UploadPage';
import ExtractedResumePage from './pages/ExtractedResumePage';
import ExtractedSkillsPage from './pages/ExtractedSkillsPage';
import StatisticsPage from './pages/StatisticsPage';
import InterviewDashboard from './pages/InterviewDashboard';
import InterviewPage from './pages/InterviewPage';
import './styles/global.css';

export default function App() {
  const [parsedData, setParsedData] = useState(null);

  return (
    <Router>
      <div className="app-container">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<UploadPage setParsedData={setParsedData} />} />
            <Route path="/extracted" element={<ExtractedResumePage parsedData={parsedData} />} />
            <Route path="/skills" element={<ExtractedSkillsPage parsedData={parsedData} />} />
            <Route path="/statistics" element={<StatisticsPage parsedData={parsedData} />} />
            <Route path="/interview-dashboard" element={<InterviewDashboard parsedData={parsedData} />} />
            <Route path="/interview" element={<InterviewPage parsedData={parsedData} />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}
