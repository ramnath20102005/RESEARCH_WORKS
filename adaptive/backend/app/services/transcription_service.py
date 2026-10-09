"""
Speech-to-Text service using faster-whisper.
Transcribes audio files to text using local Whisper model.
"""

import os
import tempfile
import logging
from typing import Optional
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriptionService:
    """Service for transcribing audio using faster-whisper."""
    
    def __init__(self, model_size: str = "base", device: str = "cuda"):
        """
        Initialize the transcription service.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cpu or cuda). Default is cuda.
        """
        self.model_size = model_size
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Whisper model with GPU fallback to CPU."""
        try:
            logger.info(f"[Transcription] Loading Whisper model: {self.model_size}")
            logger.info(f"[Transcription] Requested device: {self.device}")

            # Try requested device first
            try:
                compute_type = "int8" if self.device == "cpu" else "float16"
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=compute_type
                )
                logger.info(f"[Transcription] Model loaded successfully on {self.device} with compute_type={compute_type}")
                return
            except Exception as cuda_error:
                if self.device == "cuda":
                    logger.warning(f"[Transcription] CUDA initialization failed: {cuda_error}")
                    logger.warning(f"[Transcription] Falling back to CPU")
                    self.device = "cpu"
                    self.model = WhisperModel(
                        self.model_size,
                        device="cpu",
                        compute_type="int8"
                    )
                    logger.info(f"[Transcription] Model loaded successfully on CPU (fallback)")
                else:
                    raise Exception(f"Failed to load Whisper model on CPU: {str(cuda_error)}")

        except Exception as e:
            raise Exception(f"Failed to load Whisper model: {str(e)}")
    
    def _convert_webm_to_wav(self, input_path: str) -> str:
        """
        Convert webm audio to wav using pydub.
        
        Args:
            input_path: Path to input webm file.
        
        Returns:
            Path to output wav file.
        """
        output_path = input_path.rsplit('.', 1)[0] + '.wav'
        
        try:
            logger.info(f"[Transcription] Converting {input_path} to {output_path} using pydub")
            
            from pydub import AudioSegment
            
            # Configure pydub to use specific FFmpeg path (not in system PATH)
            ffmpeg_path = r"C:\Users\RamnathBaskar\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe"
            ffprobe_path = r"C:\Users\RamnathBaskar\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffprobe.exe"
            
            AudioSegment.converter = ffmpeg_path
            AudioSegment.ffprobe = ffprobe_path
            
            logger.info(f"[Transcription] FFmpeg path configured: {ffmpeg_path}")
            logger.info(f"[Transcription] FFprobe path configured: {ffprobe_path}")
            
            # Load audio
            audio = AudioSegment.from_file(input_path)
            
            # Convert to WAV with 16kHz mono
            audio = audio.set_frame_rate(16000)
            audio = audio.set_channels(1)
            
            # Export as WAV
            audio.export(output_path, format="wav")
            
            logger.info(f"[Transcription] Conversion successful")
            return output_path
            
        except Exception as e:
            logger.error(f"[Transcription] pydub conversion failed: {str(e)}")
            logger.error(f"[Transcription] This is a fatal error - transcription cannot proceed without audio conversion")
            raise Exception(f"Audio conversion failed: {str(e)}. FFmpeg may not be properly configured.")
    
    def transcribe_audio_file(self, audio_path: str, language: str = "en", file_extension: str = None) -> str:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file.
            language: Language code (default: 'en' for English).
            file_extension: Original file extension (for logging).
        
        Returns:
            Transcribed text.
        """
        if not self.model:
            self._load_model()
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Convert webm to wav if necessary
            actual_audio_path = audio_path
            if audio_path.endswith('.webm'):
                actual_audio_path = self._convert_webm_to_wav(audio_path)
            
            logger.info(f"[Transcription] Transcribing audio: {actual_audio_path}")
            logger.info(f"[Transcription] File size: {os.path.getsize(actual_audio_path)} bytes")
            logger.info(f"[Transcription] Language: {language}")
            if file_extension:
                logger.info(f"[Transcription] Original format: {file_extension}")
            
            # Verify WAV file header if converted
            if actual_audio_path.endswith('.wav'):
                with open(actual_audio_path, 'rb') as f:
                    header = f.read(44)
                    logger.info(f"[Transcription] WAV header (first 44 bytes): {header[:12]}...")
                    if header[:4] == b'RIFF':
                        logger.info(f"[Transcription] Valid RIFF WAV header detected")
                    else:
                        logger.warning(f"[Transcription] WARNING: Invalid WAV header")
            
            # Disable VAD filter to avoid filtering out short recordings
            # VAD parameters were too aggressive (min_speech_duration_ms=250, min_silence_duration_ms=2000)
            # Try without forcing language to allow auto-detection
            # Lower beam_size for faster processing
            segments, info = self.model.transcribe(
                actual_audio_path,
                language=None,  # Auto-detect language
                beam_size=1,  # Lower beam size for faster processing
                vad_filter=False,
                word_timestamps=True  # Enable word timestamps for debugging
            )
            
            logger.info(f"[Transcription] Detected language: {info.language}")
            logger.info(f"[Transcription] Audio duration: {info.duration:.2f}s")
            
            # Convert segments iterator to list to avoid consuming it twice
            segments_list = list(segments)
            logger.info(f"[Transcription] Number of segments: {len(segments_list)}")
            
            # Log segment details for debugging
            for i, segment in enumerate(segments_list):
                logger.info(f"[Transcription] Segment {i}: {segment.text[:50] if segment.text else '(empty)'}...")
            
            # Combine all segments
            transcript = " ".join([segment.text for segment in segments_list])
            transcript = transcript.strip()
            
            logger.info(f"[Transcription] Transcription complete: {len(transcript)} chars")
            if transcript:
                logger.info(f"[Transcription] Transcript preview: {transcript[:100]}...")
            else:
                logger.warning(f"[Transcription] WARNING: Empty transcript returned")
                logger.warning(f"[Transcription] This may indicate the audio file contains no speech or the format is not supported")
            
            # Clean up converted file if we created one
            if actual_audio_path != audio_path and os.path.exists(actual_audio_path):
                os.remove(actual_audio_path)
                logger.info(f"[Transcription] Cleaned up converted file: {actual_audio_path}")
            
            return transcript
            
        except Exception as e:
            logger.error(f"[Transcription] Transcription failed: {str(e)}")
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
        logger.info(f"[Transcription] Received audio bytes: {len(audio_bytes)} bytes")
        logger.info(f"[Transcription] File extension: {file_extension}")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(
            suffix=f".{file_extension}",
            delete=False
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        
        try:
            # Transcribe the temporary file
            transcript = self.transcribe_audio_file(temp_path, language, file_extension)
            return transcript
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info(f"[Transcription] Cleaned up temporary file: {temp_path}")
