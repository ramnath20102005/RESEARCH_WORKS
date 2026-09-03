import React, { useEffect, useRef, useState } from 'react';

export default function CameraPanel({ cameraStatus = 'disconnected', faceDetected = false, micStatus = 'disconnected', isActive = false }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [error, setError] = useState(null);
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const isStartingRef = useRef(false);

  // Cleanup function - stops all media tracks
  const cleanupMediaStream = (streamToCleanup) => {
    if (streamToCleanup) {
      console.log('[Camera] Releasing media stream...');
      streamToCleanup.getTracks().forEach(track => {
        console.log(`[Camera] Stopping track: ${track.kind}`);
        track.stop();
      });
    }
  };

  useEffect(() => {
    if (!isActive) {
      // Stop camera if not active
      console.log('[Camera] Interview not active, stopping camera');
      cleanupMediaStream(streamRef.current);
      streamRef.current = null;
      setIsConnected(false);
      setIsVideoReady(false);
      isStartingRef.current = false;
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      return;
    }

    // Guard against multiple simultaneous getUserMedia calls
    if (isStartingRef.current || streamRef.current) {
      console.log('[Camera] Camera already starting or active, skipping');
      return;
    }

    isStartingRef.current = true;

    // Start camera when active
    async function startCamera() {
      try {
        console.log('[Camera] Requesting camera access...');
        const mediaStream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: { ideal: 1280 },
            height: { ideal: 720 }
          }, 
          audio: false 
        });
        
        console.log('[Camera] Camera stream received');
        streamRef.current = mediaStream;
        
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
          console.log('[Camera] Video attached to element');
          
          // Wait for video metadata to load before marking as ready
          videoRef.current.onloadedmetadata = () => {
            console.log('[Camera] Video metadata loaded, stream ready');
            setIsVideoReady(true);
            setIsConnected(true);
            isStartingRef.current = false;
          };
          
          // Handle video load errors
          videoRef.current.onerror = (err) => {
            console.error('[Camera] Video element error:', err);
            setError('Failed to load video stream');
            setIsVideoReady(false);
            setIsConnected(false);
            isStartingRef.current = false;
          };
        }
      } catch (err) {
        console.error('[Camera] Error accessing camera:', err);
        setError('Camera access denied or not available');
        setIsVideoReady(false);
        setIsConnected(false);
        isStartingRef.current = false;
      }
    }

    startCamera();

    // Cleanup function - runs on unmount or when isActive changes
    return () => {
      console.log('[Camera] Cleanup triggered');
      cleanupMediaStream(streamRef.current);
      streamRef.current = null;
      setIsConnected(false);
      setIsVideoReady(false);
      isStartingRef.current = false;
      if (videoRef.current) {
        videoRef.current.srcObject = null;
        videoRef.current.onloadedmetadata = null;
        videoRef.current.onerror = null;
      }
    };
  }, [isActive]);

  // Additional cleanup on component unmount (safety net)
  useEffect(() => {
    return () => {
      console.log('[Camera] Component unmounting - final cleanup');
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject;
        cleanupMediaStream(stream);
        videoRef.current.srcObject = null;
      }
    };
  }, []);

  return (
    <div className="camera-panel">
      <div className="camera-header">
        <h3>📹 Live Camera</h3>
      </div>
      <div className="camera-video-container">
        {error && (
          <div className="camera-error">
            <p>{error}</p>
          </div>
        )}
        
        {/* Video element always exists - no conditional rendering */}
        <video 
          ref={videoRef} 
          autoPlay 
          playsInline 
          muted 
          className="camera-video"
        />
        
        {/* Placeholder overlay when not connected */}
        {!isConnected && (
          <div className="camera-placeholder">
            <div className="camera-icon">📷</div>
            <p className="camera-placeholder-text">
              {isActive ? 'Camera Starting...' : 'Camera Off - Start Interview to begin'}
            </p>
          </div>
        )}
      </div>
      <div className="camera-status">
        <div className={`status-item ${isConnected && isActive ? 'connected' : 'disconnected'}`}>
          <span className="status-dot"></span>
          <span className="status-text">Camera: {isConnected && isActive ? 'Connected' : 'Disconnected'}</span>
        </div>
        <div className={`status-item ${faceDetected ? 'detected' : 'not-detected'}`}>
          <span className="status-dot"></span>
          <span className="status-text">Face Detection: Not Started</span>
        </div>
        <div className={`status-item ${micStatus}`}>
          <span className="status-dot"></span>
          <span className="status-text">Microphone: {micStatus === 'connected' ? 'Connected' : 'Disconnected'}</span>
        </div>
      </div>
    </div>
  );
}
