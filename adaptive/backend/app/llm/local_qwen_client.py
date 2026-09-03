"""
Local Qwen client for the Semantic Interview Engine.
Provides local GPU-accelerated inference using Qwen models.
"""

import os
import json
import logging
import time
import torch
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class LocalQwenClient:
    """Client for local Qwen inference with GPU acceleration."""
    
    _instance = None
    _model = None
    _tokenizer = None
    _device = None
    _gpu_info = {}
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure model is loaded only once."""
        if cls._instance is None:
            cls._instance = super(LocalQwenClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: Optional[str] = None,
        quantization: str = "int4"
    ):
        """
        Initialize the local Qwen client.
        
        Args:
            model_name: The Qwen model to use. Default is Qwen2.5-7B-Instruct.
            device: Device to use (cuda, cpu, or auto-detect).
            quantization: Quantization mode (fp16, int8, int4, or none).
                         int4 is recommended for 8GB VRAM.
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.model_name = model_name
        self.quantization = quantization
        
        # Verify GPU availability
        self._verify_gpu()
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model and tokenizer (lazy load on first use)
        self._model_loaded = False
        self._initialized = True
        
        logger.info(f"[LocalQwenClient] Initialized with model: {self.model_name}")
        logger.info(f"[LocalQwenClient] Device: {self.device}")
        logger.info(f"[LocalQwenClient] Quantization: {self.quantization}")
        logger.info(f"[LocalQwenClient] GPU: {self._gpu_info.get('name', 'N/A')}")
        logger.info(f"[LocalQwenClient] VRAM: {self._gpu_info.get('vram_gb', 'N/A')} GB")
    
    def _verify_gpu(self):
        """Verify GPU availability and log details."""
        if not torch.cuda.is_available():
            logger.warning("[LocalQwenClient] WARNING: CUDA not available, will use CPU")
            self._gpu_info = {
                'available': False,
                'name': 'N/A',
                'vram_gb': 0,
                'device': 'cpu'
            }
            return
        
        gpu_count = torch.cuda.device_count()
        logger.info(f"[LocalQwenClient] CUDA available: {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_props = torch.cuda.get_device_properties(i)
            vram_gb = gpu_props.total_memory / (1024**3)
            
            logger.info(f"[LocalQwenClient] GPU {i}: {gpu_name}")
            logger.info(f"[LocalQwenClient] GPU {i} VRAM: {vram_gb:.2f} GB")
            logger.info(f"[LocalQwenClient] GPU {i} Compute Capability: {gpu_props.major}.{gpu_props.minor}")
            
            self._gpu_info = {
                'available': True,
                'name': gpu_name,
                'vram_gb': vram_gb,
                'device': f'cuda:{i}'
            }
            
            # Warn if VRAM is limited
            if vram_gb < 8:
                logger.warning(f"[LocalQwenClient] WARNING: GPU VRAM ({vram_gb:.2f} GB) is below recommended 8GB")
                logger.warning(f"[LocalQwenClient] Consider using int4 quantization or a smaller model")
    
    def _get_quantization_config(self):
        """Get quantization config based on settings."""
        if self.quantization == "int4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.quantization == "int8":
            return BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantization == "fp16":
            return {"torch_dtype": torch.float16}
        else:
            return {"torch_dtype": torch.float32}
    
    def _load_model(self):
        """Load the model and tokenizer (lazy loading)."""
        if self._model_loaded:
            return
        
        logger.info(f"[LocalQwenClient] Loading model: {self.model_name}")
        load_start = time.perf_counter()
        
        try:
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Determine load kwargs based on quantization
            load_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto" if self.device == "cuda" else None,
                "low_cpu_mem_usage": True,
            }
            
            # Add quantization config if using bitsandbytes
            if self.quantization in ["int4", "int8"]:
                load_kwargs["quantization_config"] = self._get_quantization_config()
            else:
                load_kwargs.update(self._get_quantization_config())
            
            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            
            # Set to eval mode
            self._model.eval()
            
            load_time = time.perf_counter() - load_start
            logger.info(f"[LocalQwenClient] Model loaded in {load_time:.2f}s")
            logger.info(f"[LocalQwenClient] Using device: {self.device}")
            
            # Log actual device being used
            if hasattr(self._model, 'device'):
                logger.info(f"[LocalQwenClient] Model device: {self._model.device}")
            
            if self.device == "cpu":
                logger.warning("[LocalQwenClient] WARNING: Running on CPU - this will be slow!")
            
            self._model_loaded = True
            
        except Exception as e:
            logger.error(f"[LocalQwenClient] Failed to load model: {str(e)}")
            import traceback
            logger.error(f"[LocalQwenClient] Traceback:\n{traceback.format_exc()}")
            raise Exception(f"Failed to load Qwen model: {str(e)}")
    
    def generate_content(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        response_format: str = "text"
    ) -> str:
        """
        Generate content using local Qwen.
        
        Args:
            prompt: The user prompt to send.
            system_instruction: Optional system instruction.
            temperature: Sampling temperature (0.0 to 1.0).
            response_format: Expected response format ("text" or "json").
        
        Returns:
            The generated content as a string.
        """
        # Ensure model is loaded
        self._load_model()
        
        try:
            logger.info(f"[PERF] local_qwen_request_start: 0.000s")
            request_start = time.perf_counter()
            
            logger.info(f"======== LOCAL QWEN Request ========")
            logger.info(f"Model: {self.model_name}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Quantization: {self.quantization}")
            logger.info(f"Temperature: {temperature}")
            logger.info(f"Response Format: {response_format}")
            logger.info(f"System Instruction: {system_instruction[:100] if system_instruction else 'None'}...")
            logger.info(f"Prompt: {prompt[:200]}...")
            logger.info(f"================================")
            
            # Build messages
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})
            
            # Apply chat template
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Add JSON format instruction if requested
            if response_format == "json":
                text += "\n\nRespond with valid JSON only."
            
            # Tokenize
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self._model.device)
            
            # Generate
            inference_start = time.perf_counter()
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            
            inference_time = time.perf_counter() - inference_start
            
            # Decode
            generated_text = self._tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            total_time = time.perf_counter() - request_start
            logger.info(f"[PERF] local_qwen_inference: {inference_time:.3f}s")
            logger.info(f"[PERF] local_qwen_total: {total_time:.3f}s")
            
            logger.info(f"======== LOCAL QWEN Response ========")
            logger.info(f"Latency: {total_time:.2f}s")
            logger.info(f"Raw Response: {generated_text[:500]}...")
            logger.info(f"================================")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"[LocalQwenClient] Exception type: {type(e).__name__}")
            logger.error(f"[LocalQwenClient] Exception message: {str(e)}")
            import traceback
            logger.error(f"[LocalQwenClient] Traceback:\n{traceback.format_exc()}")
            raise Exception(f"Local Qwen inference error: {str(e)}")
    
    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate JSON content using local Qwen.
        
        Args:
            prompt: The user prompt to send.
            system_instruction: Optional system instruction.
            temperature: Sampling temperature (0.0 to 1.0).
        
        Returns:
            The generated content as a dictionary.
        """
        response_text = self.generate_content(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            response_format="json"
        )
        
        # Log raw response for debugging
        logger.info("======================== RAW LOCAL QWEN TEXT RESPONSE ========================")
        logger.info(f"Type: {type(response_text)}")
        logger.info(f"Repr: {repr(response_text)}")
        logger.info(f"Content: {response_text}")
        logger.info("========================================================================")
        
        # Try to extract JSON from response (handle markdown code fences)
        json_text = response_text.strip()
        
        # Remove markdown code fences if present
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.startswith("```"):
            json_text = json_text[3:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        json_text = json_text.strip()
        
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Qwen response as JSON: {e}")
            logger.error(f"Response was: {response_text}")
            raise ValueError(
                f"Failed to parse Qwen response as JSON: {e}\n"
                f"Response was: {response_text}"
            )
