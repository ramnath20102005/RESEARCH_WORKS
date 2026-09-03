"""
Text-to-Speech service using Kokoro TTS.
Generates natural-sounding speech from text using local Kokoro ONNX models.
"""

import hashlib
import io
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTSService:
    """Service for text-to-speech using Kokoro TTS."""
    
    _instance = None
    _model = None
    _voices = None
    _cache = {}  # Simple cache: {hash: audio_bytes}
    
    def __new__(cls):
        """Singleton pattern to ensure model is loaded only once."""
        if cls._instance is None:
            cls._instance = super(TTSService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, voice: str = "af_bella"):
        """
        Initialize the TTS service.
        
        Args:
            voice: Voice to use for synthesis (default: 'af_bella').
                   Available voices: af_bella, af_nicole, am_michael, etc.
        """
        # Lazy initialization - model will be loaded on first synthesis request
        self.voice = voice
        self._model_loaded = False
    
    def _load_model(self):
        """Load the Kokoro model (cached for singleton)."""
        import time
        load_start = time.perf_counter()
        logger.info("[PERF] kokoro_model_loading: 0.000s")
        
        try:
            logger.info("[TTS] Loading Kokoro TTS model...")
            
            # Import Kokoro here to avoid import errors if not installed
            from kokoro_onnx import Kokoro
            import onnxruntime as ort
            
            # Log available execution providers
            logger.info(f"[TTS] Available ONNX Runtime providers: {ort.get_available_providers()}")
            logger.info(f"[TTS] Using CPUExecutionProvider (GPU requires CUDA 13.x and cuDNN 9.x)")
            
            # Get absolute paths for model files
            backend_dir = Path(__file__).parent.parent.parent
            model_path = backend_dir / "kokoro-v1.0.onnx"
            voices_path = backend_dir / "voices.bin"
            
            # Check if model files exist
            if not model_path.exists():
                logger.warning(f"[TTS] Kokoro model file not found at {model_path}")
                logger.warning("[TTS] TTS will be disabled. Please download model files to enable TTS.")
                self._model = None
                self._voices = []
                self._model_loaded = False
                return
            
            if not voices_path.exists():
                logger.warning(f"[TTS] Kokoro voices file not found at {voices_path}")
                logger.warning("[TTS] TTS will be disabled. Please download model files to enable TTS.")
                self._model = None
                self._voices = []
                self._model_loaded = False
                return
            
            # Initialize Kokoro with default model and voices
            self._model = Kokoro(str(model_path), str(voices_path))
            
            # Log the actual execution provider being used
            logger.info(f"[TTS] ONNX Runtime session providers: {self._model.sess.get_providers()}")
            
            # Get available voices
            self._voices = self._model.voices
            self._model_loaded = True
            
            load_time = time.perf_counter() - load_start
            logger.info(f"[PERF] kokoro_model_loading: {load_time:.3f}s")
            logger.info(f"[TTS] ✓ Model loaded successfully")
            logger.info(f"[TTS] ✓ Voices loaded: {len(self._voices)} voices available")
            logger.info(f"[TTS] ✓ Available voice names: {self._model.get_voices()}")
            logger.info(f"[TTS] ✓ Ready for synthesis")
            
        except Exception as e:
            logger.error(f"[TTS] Failed to load Kokoro model: {str(e)}")
            logger.warning("[TTS] TTS will be disabled due to model loading error.")
            self._model = None
            self._voices = []
            self._model_loaded = False
    
    def _get_cache_key(self, text: str, voice: str, speed: float) -> str:
        """Generate cache key from text, voice, and speed."""
        key = f"{text}|{voice}|{speed}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0
    ) -> bytes:
        """
        Synthesize speech from text.
        
        Args:
            text: The text to synthesize.
            voice: Voice to use (overrides default voice).
            speed: Speech speed (0.5 to 2.0, default: 1.0).
        
        Returns:
            Audio data as bytes (16kHz mono WAV).
        
        Raises:
            RuntimeError: If TTS model is not available.
        """
        import time
        if not text or text.strip() == "":
            raise ValueError("Text cannot be empty")
        
        if self._model is None or not self._model_loaded:
            self._load_model()
        
        # Check if model is available after loading attempt
        if self._model is None:
            raise RuntimeError(
                "TTS model is not available. "
                "Please download the required model files (kokoro-v1.0.onnx and voices.bin) "
                "from the official kokoro-onnx repository and place them in the backend directory."
            )
        
        # Use provided voice or default
        selected_voice = voice or self.voice
        
        # Check cache
        cache_key = self._get_cache_key(text, selected_voice, speed)
        if cache_key in self._cache:
            logger.info(f"[TTS] Cache hit for: {text[:30]}...")
            return self._cache[cache_key]
        
        try:
            logger.info(f"[TTS] Synthesizing text: {text[:50]}...")
            synth_start = time.perf_counter()
            
            # Validate voice exists
            if selected_voice not in self._voices:
                logger.warning(f"[TTS] Voice '{selected_voice}' not found, using default")
                selected_voice = self._voices[0] if self._voices else "af_bella"
            
            # Generate audio
            audio, sample_rate = self._model.create(
                text,
                voice=selected_voice,
                speed=speed
            )
            
            synthesis_time = time.perf_counter() - synth_start
            logger.info(f"[PERF] kokoro_synthesis: {synthesis_time:.3f}s")
            logger.info(f"[TTS] Kokoro synthesis time: {synthesis_time:.2f}s")
            
            # Convert to numpy array if needed
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            
            # Resample to 16kHz mono if needed
            resample_start = time.time()
            if sample_rate != 16000:
                # Simple resampling using numpy (for production, use scipy.signal.resample)
                from scipy import signal
                num_samples = int(len(audio) * 16000 / sample_rate)
                audio = signal.resample(audio, num_samples)
                sample_rate = 16000
            
            # Ensure mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            resample_time = time.time() - resample_start
            logger.info(f"[TTS] Resample/encode time: {resample_time:.2f}s")
            
            # Convert to bytes directly (no temporary file)
            buffer = io.BytesIO()
            sf.write(buffer, audio, sample_rate, format='WAV', subtype='PCM_16')
            buffer.seek(0)
            audio_bytes = buffer.getvalue()
            buffer.close()
            
            # Cache the result
            self._cache[cache_key] = audio_bytes
            
            total_time = time.time() - synth_start
            logger.info(f"[TTS] Total TTS time: {total_time:.2f}s (synthesis: {synthesis_time:.2f}s, encoding: {resample_time:.2f}s)")
            logger.info(f"[TTS] Synthesis complete: {len(audio_bytes)} bytes (16kHz mono WAV)")
            return audio_bytes
            
        except Exception as e:
            logger.error(f"[TTS] Failed to synthesize speech: {str(e)}")
            raise Exception(f"TTS synthesis failed: {str(e)}")
    
    def get_available_voices(self) -> list:
        """
        Get list of available voices.
        
        Returns:
            List of voice names.
        """
        if self._model is None:
            self._load_model()
        return self._voices or []
    
    def set_voice(self, voice: str):
        """
        Set the default voice.
        
        Args:
            voice: Voice name to use.
        """
        if self._model is None:
            self._load_model()
        
        if voice in self._voices:
            self.voice = voice
            logger.info(f"[TTS] Voice set to: {voice}")
        else:
            logger.warning(f"[TTS] Voice '{voice}' not available")
    
    def clear_cache(self):
        """Clear the audio cache."""
        self._cache.clear()
        logger.info("[TTS] Cache cleared")
