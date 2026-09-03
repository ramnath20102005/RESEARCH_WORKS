import React, { useState, useRef, useEffect } from 'react';

export default function AudioRecorder({ onRecordingComplete, onRecordingStart, isRecording: externalIsRecording }) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const recordingStartTimeRef = useRef(null);

  useEffect(() => {
    // Sync with external recording state if provided
    if (externalIsRecording !== undefined && externalIsRecording !== isRecording) {
      if (externalIsRecording) {
        startRecording();
      } else {
        stopRecording();
      }
    }
  }, [externalIsRecording]);

  const startRecording = async () => {
    try {
      console.log('[VOICE] microphone permission requested');
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      console.log('[VOICE] microphone permission granted');
      console.log('[VOICE] audio tracks:', stream.getAudioTracks().length);
      const audioTrack = stream.getAudioTracks()[0];
      console.log('[VOICE] track state:', audioTrack?.readyState);
      console.log('[VOICE] track settings:', audioTrack?.getSettings());
      
      recordingStartTimeRef.current = Date.now();
      
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];
      
      console.log('[VOICE] MediaRecorder created');
      console.log('[VOICE] MediaRecorder MIME type:', mediaRecorderRef.current.mimeType);
      console.log('[VOICE] MediaRecorder state:', mediaRecorderRef.current.state);

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
          console.log('[VOICE] chunk received, size:', event.data.size);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        const recordingDuration = recordingStartTimeRef.current ? (Date.now() - recordingStartTimeRef.current) / 1000 : 0;
        console.log('[VOICE] recording stopped');
        console.log('[VOICE][DEBUG] recording_duration_ms:', recordingDuration * 1000);
        console.log('[VOICE] chunks collected:', audioChunksRef.current.length);
        
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        
        console.log('[VOICE] audio blob created');
        console.log('[VOICE][DEBUG] audio_blob_size:', audioBlob.size);
        console.log('[VOICE][DEBUG] audio_mime_type:', audioBlob.type);
        console.log('[VOICE][DEBUG] audio_track_state:', audioTrack?.readyState);
        console.log('[VOICE][DEBUG] audio_track_settings:', audioTrack?.getSettings());
        
        // Stop all audio tracks
        stream.getTracks().forEach(track => track.stop());
        
        // Pass the audio blob and duration to parent
        if (onRecordingComplete) {
          console.log('[VOICE] calling onRecordingComplete');
          onRecordingComplete(audioBlob, recordingDuration);
        }
      };

      mediaRecorderRef.current.start();
      console.log('[VOICE] MediaRecorder.start() called');
      console.log('[VOICE] MediaRecorder state after start:', mediaRecorderRef.current.state);
      
      setIsRecording(true);
      setRecordingTime(0);

      // Notify parent that recording started
      if (onRecordingStart) {
        console.log('[VOICE] calling onRecordingStart');
        onRecordingStart();
      }

      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

      console.log('[VOICE] recording started');

    } catch (err) {
      console.error('[VOICE] Error accessing microphone:', err);
      alert('Failed to access microphone. Please check permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      console.log('[VOICE] stopping recording...');
      console.log('[VOICE] MediaRecorder state before stop:', mediaRecorderRef.current.state);
      mediaRecorderRef.current.stop();
      setIsRecording(false);

      // Clear timer
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    } else {
      console.log('[VOICE] stopRecording called but not recording');
    }
  };

  const handleToggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="audio-recorder">
      <button
        className={`record-button ${isRecording ? 'recording' : ''}`}
        onClick={handleToggleRecording}
        disabled={externalIsRecording === true}
      >
        {isRecording ? (
          <>
            <span className="record-icon">⏹️</span>
            <span className="record-text">Stop Recording</span>
            <span className="record-time">{formatTime(recordingTime)}</span>
          </>
        ) : (
          <>
            <span className="record-icon">🎤</span>
            <span className="record-text">Start Recording</span>
          </>
        )}
      </button>
    </div>
  );
}
