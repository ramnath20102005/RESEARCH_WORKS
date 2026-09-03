import axios from 'axios';

const API_BASE = 'http://127.0.0.1:8000';

export const uploadResumeApi = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await axios.post(`${API_BASE}/upload-resume`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return response.data;
};

export const getSkillsApi = async () => {
  const response = await axios.get(`${API_BASE}/skills`);
  return response.data;
};

export const getResumeSummaryApi = async () => {
  const response = await axios.get(`${API_BASE}/resume-summary`);
  return response.data;
};

export const getHealthApi = async () => {
  const response = await axios.get(`${API_BASE}/health`);
  return response.data;
};

export const startInterviewApi = async (resumeData) => {
  const response = await axios.post(`${API_BASE}/interview/start`, {
    resume_data: resumeData
  });
  return response.data;
};

export const evaluateAnswerApi = async (question, topic, difficulty, answer) => {
  const response = await axios.post(`${API_BASE}/interview/evaluate`, {
    question,
    topic,
    difficulty,
    answer
  });
  return response.data;
};

export const transcribeAudioApi = async (audioBlob) => {
  const formData = new FormData();
  formData.append('file', audioBlob, 'recording.webm');

  const response = await axios.post(`${API_BASE}/interview/transcribe`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return response.data;
};

export const textToSpeechApi = async (text, voice = 'af_bella', speed = 1.0) => {
  const response = await axios.post(
    `${API_BASE}/interview/tts`,
    null,
    {
      params: { text, voice, speed },
      responseType: 'blob'
    }
  );
  return response.data;
};

// Adaptive Interview API Functions

export const startAdaptiveInterviewApi = async (sessionId, resumeData, voice = 'af_bella') => {
  const response = await axios.post(
    `${API_BASE}/interview/adaptive/start`,
    {
      session_id: sessionId,
      resume_data: resumeData,
      voice: voice
    },
    {
      responseType: 'blob'
    }
  );
  
  // Extract metadata from response headers
  const metadata = {
    questionNumber: response.headers['x-question-number'],
    question: response.headers['x-question'],
    topic: response.headers['x-topic'],
    difficulty: response.headers['x-difficulty'],
    source: response.headers['x-source'],
    sessionId: response.headers['x-session-id']
  };
  
  return {
    audio: response.data,
    metadata
  };
};

export const transcribeAdaptiveAudioApi = async (sessionId, audioBlob) => {
  const formData = new FormData();
  formData.append('file', audioBlob, 'recording.webm');

  const response = await axios.post(
    `${API_BASE}/interview/adaptive/audio?session_id=${sessionId}`,
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' }
    }
  );
  return response.data;
};

export const submitAdaptiveAnswerApi = async (sessionId, transcript, audioDuration = 0, voice = 'af_bella') => {
  const response = await axios.post(
    `${API_BASE}/interview/adaptive/answer`,
    {
      session_id: sessionId,
      transcript: transcript,
      audio_duration: audioDuration,
      voice: voice
    },
    {
      responseType: 'blob'
    }
  );
  
  // Check if interview ended
  const interviewEnded = response.headers['x-interview-ended'] === 'true';
  
  if (interviewEnded) {
    // Return JSON data for ended interview
    return {
      interviewEnded: true,
      data: response.data
    };
  }
  
  // Extract metadata from response headers
  const metadata = {
    questionNumber: response.headers['x-question-number'],
    question: response.headers['x-question'],
    topic: response.headers['x-topic'],
    difficulty: response.headers['x-difficulty'],
    source: response.headers['x-source'],
    sessionId: response.headers['x-session-id'],
    policy: response.headers['x-policy']
  };
  
  return {
    audio: response.data,
    metadata,
    interviewEnded: false
  };
};

export const getAdaptiveStateApi = async (sessionId) => {
  const response = await axios.get(
    `${API_BASE}/interview/adaptive/state?session_id=${sessionId}`
  );
  return response.data;
};

export const endAdaptiveInterviewApi = async (sessionId) => {
  const response = await axios.post(
    `${API_BASE}/interview/adaptive/end?session_id=${sessionId}`
  );
  return response.data;
};
