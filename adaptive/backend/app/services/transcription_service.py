"""
Speech-to-Text service using faster-whisper.
Transcribes audio files to text using local Whisper model.
"""

import os
import tempfile
from typing import Optional
from faster_whisper import WhisperModel


class TranscriptionService:
    """Service for transcribing audio using faster-whisper."""
    
    def __init__(self, model_size: str = "base", device: str = "cpu"):
        """
        Initialize the transcription service.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cpu or cuda)
        """
        self.model_size = model_size
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        try:
            print(f"[Transcription] Loading Whisper model: {self.model_size}")
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="int8" if self.device == "cpu" else "float16"
            )
            print(f"[Transcription] Model loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load Whisper model: {str(e)}")
    
    def transcribe_audio_file(self, audio_path: str, language: str = "en") -> str:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file.
            language: Language code (default: 'en' for English).
        
        Returns:
            Transcribed text.
        """
        if not self.model:
            self._load_model()
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            print(f"[Transcription] Transcribing audio: {audio_path}")
            
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    max_speech_duration_s=30,
                    min_silence_duration_ms=2000,
                    speech_pad_ms=30
                )
            )
            
            # Combine all segments
            transcript = " ".join([segment.text for segment in segments])
            transcript = transcript.strip()
            
            print(f"[Transcription] Transcription complete: {len(transcript)} chars")
            return transcript
            
        except Exception as e:
            raise Exception(f"Failed to transcribe audio: {str(e)}")
    
    def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        file_extension: str = "wav",
        language: str = "en"
    ) -> str:
        """
        Transcribe audio from bytes.
        
        Args:
            audio_bytes: Audio data as bytes.
            file_extension: File extension (wav, mp3, etc.).
            language: Language code (default: 'en' for English).
        
        Returns:
            Transcribed text.
        """
        # Create a temporary file
        with tempfile.NamedTemporaryFile(
            suffix=f".{file_extension}",
            delete=False
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        
        try:
            # Transcribe the temporary file
            transcript = self.transcribe_audio_file(temp_path, language)
            return transcript
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"[Transcription] Cleaned up temporary file: {temp_path}")
