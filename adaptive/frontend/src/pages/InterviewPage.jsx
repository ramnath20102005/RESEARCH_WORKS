import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import CameraPanel from '../components/Interview/CameraPanel/CameraPanel';
import ChatWindow from '../components/Interview/ChatWindow/ChatWindow';
import StatusPanel from '../components/Interview/StatusPanel/StatusPanel';
import ProgressPanel from '../components/Interview/ProgressPanel/ProgressPanel';
import InterviewControls from '../components/Interview/InterviewControls/InterviewControls';
import Timer from '../components/Interview/Timer/Timer';
import SkillSidebar from '../components/Interview/SkillSidebar/SkillSidebar';
import Transcript from '../components/Interview/Transcript/Transcript';
import AudioRecorder from '../components/Interview/AudioRecorder/AudioRecorder';
import VoiceSelector from '../components/Interview/VoiceSelector/VoiceSelector';
import PerformancePanel from '../components/Interview/PerformancePanel/PerformancePanel';
import { startAdaptiveInterviewApi, transcribeAdaptiveAudioApi, submitAdaptiveAnswerApi, getAdaptiveStateApi, endAdaptiveInterviewApi } from '../services/api';
import '../styles/interview.css';

export default function InterviewPage({ parsedData }) {
  const navigate = useNavigate();
  const [isInterviewStarted, setIsInterviewStarted] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [selectedVoice, setSelectedVoice] = useState('af_bella');
  const [currentQuestion, setCurrentQuestion] = useState(1);
  const [totalQuestions] = useState(10);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [messages, setMessages] = useState([]);
  const [currentAnswer, setCurrentAnswer] = useState('');
  const [difficulty, setDifficulty] = useState('Easy');
  const [currentTopic, setCurrentTopic] = useState('');
  const [isGeneratingQuestion, setIsGeneratingQuestion] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evaluation, setEvaluation] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [performanceData, setPerformanceData] = useState([]);
  const [recordingStartTime, setRecordingStartTime] = useState(null);
  const [error, setError] = useState(null);
  const audioRef = useRef(null);

  const technicalSkills = parsedData?.technical_skills || {};

  const playAudio = async (audioBlob) => {
    return new Promise((resolve, reject) => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }

      try {
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audioRef.current = audio;

        audio.onended = () => {
          setIsPlayingAudio(false);
          URL.revokeObjectURL(audioUrl);
          resolve();
        };

        audio.onerror = (error) => {
          setIsPlayingAudio(false);
          URL.revokeObjectURL(audioUrl);
          console.error('[Audio] Playback error:', error);
          setError('Failed to play audio. Question text is displayed below.');
          reject(error);
        };

        audio.play().catch(playError => {
          setIsPlayingAudio(false);
          URL.revokeObjectURL(audioUrl);
          console.error('[Audio] Play error:', playError);
          setError('Failed to play audio. Question text is displayed below.');
          reject(playError);
        });
        
        setIsPlayingAudio(true);
      } catch (error) {
        console.error('[Audio] Audio setup error:', error);
        setError('Failed to play audio. Question text is displayed below.');
        reject(error);
      }
    });
  };

  const stopAudio = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
      setIsPlayingAudio(false);
    }
  };

  const handleStartInterview = async () => {
    setIsInterviewStarted(true);
    setIsGeneratingQuestion(true);
    
    // Generate unique session ID
    const newSessionId = `session_${Date.now()}`;
    setSessionId(newSessionId);
    
    const requestStartTime = performance.now();
    console.log('[PERF] start_request_sent: 0.000ms');
    
    try {
      console.log('[Interview] Starting adaptive interview...');
      console.log('[Interview] Session ID:', newSessionId);
      console.log('[Interview] Voice:', selectedVoice);
      
      const response = await startAdaptiveInterviewApi(newSessionId, parsedData, selectedVoice);
      
      const responseTime = performance.now() - requestStartTime;
      console.log('[PERF] backend_response_received:', responseTime.toFixed(2), 'ms');
      
      console.log('[Interview] Question metadata:', response.metadata);
      setCurrentTopic(response.metadata.topic);
      setDifficulty(response.metadata.difficulty);
      setCurrentQuestion(parseInt(response.metadata.questionNumber));
      
      // Add AI question to messages
      setMessages([{
        id: Date.now(),
        role: 'ai',
        content: response.metadata.question,
        questionNumber: parseInt(response.metadata.questionNumber)
      }]);
      
      // Play the audio
      console.log('[Interview] Playing audio...');
      const audioStartTime = performance.now();
      await playAudio(response.audio);
      const audioPlayTime = performance.now() - audioStartTime;
      console.log('[PERF] audio_playback_started:', audioPlayTime.toFixed(2), 'ms');
      
      const totalTime = performance.now() - requestStartTime;
      console.log('[PERF] total_frontend_startup:', totalTime.toFixed(2), 'ms');
      console.log('[Interview] Interview start total time:', totalTime.toFixed(2), 'ms');
      
      // Log performance
      setPerformanceData([{
        question_number: 1,
        nvidia_llm: totalTime,
        kokoro: 0, // Will be updated when we get timing from backend
        total: totalTime
      }]);
      
    } catch (error) {
      console.error('[Interview] Failed to start interview:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to start interview. Please try again.';
      setError(errorMessage);
      setIsInterviewStarted(false);
    } finally {
      setIsGeneratingQuestion(false);
    }
  };

  const handleEndInterview = async () => {
    // Stop any playing audio
    stopAudio();
    
    if (confirm('Are you sure you want to end the interview?')) {
      try {
        // End session on backend
        if (sessionId) {
          console.log('[Interview] Ending session:', sessionId);
          await endAdaptiveInterviewApi(sessionId);
        }
      } catch (error) {
        console.error('[Interview] Failed to end session:', error);
      }
      
      // Reset state
      setSessionId(null);
      setIsInterviewStarted(false);
      setMessages([]);
      setEvaluation(null);
      setPerformanceData([]);
      
      navigate('/interview-dashboard');
    }
  };

  const handleRecordingStart = () => {
    setRecordingStartTime(Date.now());
  };

  const handleRecordingComplete = async (audioBlob, recordingDuration) => {
    if (!sessionId) {
      alert('Session not found. Please start a new interview.');
      return;
    }
    
    console.log('[VOICE] onRecordingComplete received audio blob');
    console.log('[VOICE] audio blob size:', audioBlob.size, 'bytes');
    console.log('[VOICE] audio blob type:', audioBlob.type);
    console.log('[VOICE] recording duration from recorder:', recordingDuration, 's');
    
    setIsTranscribing(true);
    setIsRecording(false);
    
    const cycleStartTime = performance.now();
    const audioDuration = recordingDuration || (recordingStartTime ? (Date.now() - recordingStartTime) / 1000 : 0);
    
    try {
      console.log('[VOICE] uploading audio to STT');
      const whisperStart = performance.now();
      
      const response = await transcribeAdaptiveAudioApi(sessionId, audioBlob);
      const transcript = response.transcript;
      
      const whisperTime = performance.now() - whisperStart;
      console.log('[PERF][STT] Whisper completed:', whisperTime.toFixed(2), 'ms');
      console.log('[VOICE] transcript:', transcript);
      console.log('[VOICE] transcript length:', transcript.length);
      
      // Check for empty transcript
      if (!transcript || transcript.trim() === '') {
        console.warn('[VOICE] Empty transcript detected from Whisper');
        setError('No speech detected. Please try again.');
        setIsTranscribing(false);
        setRecordingStartTime(null);
        return;
      }
      
      // Add candidate answer to messages
      console.log('[VOICE] adding transcript to conversation');
      setMessages(prev => [...prev, {
        id: Date.now(),
        role: 'candidate',
        content: transcript
      }]);
      console.log('[VOICE] transcript added to conversation');
      
      // Process answer through adaptive pipeline
      await processAdaptiveAnswer(transcript, audioDuration, cycleStartTime, whisperTime);
      
    } catch (error) {
      console.error('[VOICE] Transcription failed:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to transcribe audio. Please try again.';
      setError(errorMessage);
    } finally {
      setIsTranscribing(false);
      setRecordingStartTime(null);
    }
  };

  const processAdaptiveAnswer = async (transcript, audioDuration, cycleStartTime, whisperTime) => {
    setIsEvaluating(true);
    
    try {
      console.log('[VOICE] processing answer through adaptive pipeline');
      console.log('[VOICE] transcript:', transcript);
      console.log('[VOICE] audio duration:', audioDuration, 's');
      
      const llmStart = performance.now();
      console.log('[VOICE] sending answer to backend for evaluation');
      
      const response = await submitAdaptiveAnswerApi(sessionId, transcript, audioDuration, selectedVoice);
      
      const llmTime = performance.now() - llmStart;
      console.log('[PERF][ANSWER] LocalLLM + TabPFN time:', llmTime.toFixed(2), 'ms');
      
      // Check if interview ended
      if (response.interviewEnded) {
        console.log('[VOICE] interview ended by backend');
        setEvaluation(response.data.evaluation);
        
        // Show summary
        alert(`Interview ended!\n\nSummary:\nTotal Questions: ${response.data.session_summary.total_questions}\nAverage Correctness: ${response.data.session_summary.average_correctness.toFixed(1)}%\nAccuracy: ${(response.data.session_summary.accuracy * 100).toFixed(1)}%`);
        
        await handleEndInterview();
        return;
      }
      
      console.log('[VOICE] next question metadata:', response.metadata);
      
      // Update state
      setEvaluation(response.metadata.evaluation);
      setCurrentTopic(response.metadata.topic);
      setDifficulty(response.metadata.difficulty);
      setCurrentQuestion(parseInt(response.metadata.questionNumber));
      
      // Add next AI question to messages
      console.log('[VOICE] adding next question to conversation');
      setMessages(prev => [...prev, {
        id: Date.now(),
        role: 'ai',
        content: response.metadata.question,
        questionNumber: parseInt(response.metadata.questionNumber)
      }]);
      
      // Play next question audio
      console.log('[VOICE] playing next question audio');
      await playAudio(response.audio);
      
      const totalTime = performance.now() - cycleStartTime;
      console.log('[PERF][ANSWER_PIPELINE] Complete answer cycle time:', totalTime.toFixed(2), 'ms');
      console.log('[PERF][ANSWER_PIPELINE] Recording:', (audioDuration * 1000).toFixed(2), 'ms');
      console.log('[PERF][ANSWER_PIPELINE] Whisper:', whisperTime.toFixed(2), 'ms');
      console.log('[PERF][ANSWER_PIPELINE] LLM+TabPFN:', llmTime.toFixed(2), 'ms');
      console.log('[PERF][ANSWER_PIPELINE] TOTAL:', totalTime.toFixed(2), 'ms');
      
      // Log performance
      setPerformanceData(prev => [...prev, {
        question_number: parseInt(response.metadata.questionNumber) - 1,
        whisper: whisperTime,
        nvidia_llm: llmTime,
        kokoro: 0, // Will be updated when we get timing from backend
        tabpfn: 0, // Will be updated when we get timing from backend
        total: totalTime,
        details: {
          model: 'meta/llama-3.1-8b-instruct',
          voice: selectedVoice,
          policy: response.metadata.policy
        }
      }]);
      
    } catch (error) {
      console.error('[VOICE] Failed to process answer:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to process answer. Please try again.';
      setError(errorMessage);
    } finally {
      setIsEvaluating(false);
    }
  };

  const handleTimeUpdate = (seconds) => {
    setElapsedTime(seconds);
  };

  const progress = Math.round((currentQuestion / totalQuestions) * 100);

  const formatTime = (totalSeconds) => {
    const mins = Math.floor(totalSeconds / 60);
    const secs = totalSeconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const getScoreClass = (score) => {
    if (score >= 80) return 'high';
    if (score >= 60) return 'medium';
    return 'low';
  };

  return (
    <div className="interview-page">
      <div className="page-header">
        <h1 className="page-title">Adaptive AI Interview</h1>
        <p className="page-subtitle">
          Real-time interview with AI-powered semantic evaluation
        </p>
        {error && (
          <div className="error-banner">
            <span className="error-message">{error}</span>
            <button 
              className="error-dismiss"
              onClick={() => setError(null)}
            >
              ✕
            </button>
          </div>
        )}
      </div>

      <div className="interview-layout">
        {/* Left Side - Camera Panel */}
        <div className="interview-left">
          <CameraPanel
            cameraStatus={isInterviewStarted ? 'connected' : 'disconnected'}
            faceDetected={false}
            micStatus={isInterviewStarted ? 'connected' : 'disconnected'}
            isActive={isInterviewStarted}
          />
          <SkillSidebar skills={technicalSkills} />
        </div>

        {/* Center - Chat Window */}
        <div className="interview-center">
          <ChatWindow messages={messages} />
          
          {/* Answer Box */}
          {isInterviewStarted && (
            <div className="answer-box">
              {isGeneratingQuestion && (
                <div className="loading-message">Generating question...</div>
              )}
              
              {!isGeneratingQuestion && !isTranscribing && !isEvaluating && !evaluation && (
                <>
                  <AudioRecorder 
                    onRecordingComplete={handleRecordingComplete}
                    onRecordingStart={handleRecordingStart}
                    isRecording={isRecording}
                  />
                  <p className="recording-hint">Click the microphone to record your answer</p>
                </>
              )}
              
              {isTranscribing && (
                <div className="loading-message">Transcribing your answer...</div>
              )}
              
              {isEvaluating && (
                <div className="loading-message">Evaluating your answer...</div>
              )}
              
              {evaluation && (
                <div className="semantic-evaluation">
                  <div className="evaluation-header">
                    <h3>Answer Evaluation</h3>
                    <span className={`evaluation-status ${evaluation.is_correct ? 'correct' : 'incorrect'}`}>
                      {evaluation.is_correct ? 'Correct' : 'Incorrect'}
                    </span>
                  </div>
                  
                  <div className="evaluation-scores">
                    <div className="score-item">
                      <div className="score-label">Correctness</div>
                      <div className={`score-value ${getScoreClass(evaluation.correctness_score)}`}>
                        {evaluation.correctness_score}%
                      </div>
                    </div>
                    <div className="score-item">
                      <div className="score-label">Concept Coverage</div>
                      <div className={`score-value ${getScoreClass(evaluation.concept_coverage)}`}>
                        {evaluation.concept_coverage}%
                      </div>
                    </div>
                    <div className="score-item">
                      <div className="score-label">Reasoning</div>
                      <div className={`score-value ${getScoreClass(evaluation.reasoning_score)}`}>
                        {evaluation.reasoning_score}%
                      </div>
                    </div>
                    <div className="score-item">
                      <div className="score-label">Missing Concepts</div>
                      <div className="score-value">
                        {evaluation.missing_concepts}
                      </div>
                    </div>
                  </div>
                  
                  <div className="evaluation-feedback">
                    <div className="feedback-label">Feedback</div>
                    <div className="feedback-text">{evaluation.feedback}</div>
                  </div>
                </div>
              )}
              
              <InterviewControls
                isInterviewStarted={isInterviewStarted}
                onStartInterview={handleStartInterview}
                onEndInterview={handleEndInterview}
                onNextQuestion={() => {}}
                onSubmitAnswer={() => {}}
              />
            </div>
          )}
          
          {!isInterviewStarted && (
            <div className="interview-start-prompt">
              <VoiceSelector
                selectedVoice={selectedVoice}
                onVoiceChange={setSelectedVoice}
                disabled={isInterviewStarted}
              />
              <InterviewControls
                isInterviewStarted={isInterviewStarted}
                onStartInterview={handleStartInterview}
                onEndInterview={handleEndInterview}
                onNextQuestion={() => {}}
                onSubmitAnswer={() => {}}
              />
            </div>
          )}
        </div>

        {/* Right Side - Status Panel */}
        <div className="interview-right">
          <Timer initialSeconds={elapsedTime} onTimeUpdate={handleTimeUpdate} />
          <StatusPanel
            questionNumber={currentQuestion}
            difficulty={difficulty}
            elapsedTime={formatTime(elapsedTime)}
            currentTopic={currentTopic || 'Not started'}
          />
          <ProgressPanel
            currentQuestion={currentQuestion}
            totalQuestions={totalQuestions}
            progress={progress}
          />
          <PerformancePanel performanceData={performanceData} />
          <Transcript messages={messages} />
        </div>
      </div>
    </div>
  );
}
