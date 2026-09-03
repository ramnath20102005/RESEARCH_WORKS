"""
Local LLM client for the Semantic Interview Engine.
Provides local GPU-accelerated inference using Qwen1.5-4B-Chat with 4-bit quantization.
"""

import os
import json
import logging
import time
import torch
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class LocalLLMClient:
    """Client for local LLM inference with GPU acceleration using Qwen1.5-4B-Chat."""
    
    _instance = None
    _model = None
    _tokenizer = None
    _device = None
    _gpu_info = {}
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure model is loaded only once."""
        if cls._instance is None:
            cls._instance = super(LocalLLMClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen1.5-4B-Chat",
        device: Optional[str] = None,
        quantization: str = "int4"
    ):
        """
        Initialize the local LLM client.
        
        Args:
            model_name: The Qwen model to use. Default is Qwen1.5-4B-Chat.
            device: Device to use (cuda, cpu, or auto-detect).
            quantization: Quantization mode (int4, int8, fp16, or none).
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
        
        logger.info(f"[LocalLLM] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"[LocalLLM] GPU: {self._gpu_info.get('name', 'N/A')}")
        logger.info(f"[LocalLLM] Model: {self.model_name}")
        logger.info(f"[LocalLLM] Quantization: 4-bit")
        logger.info(f"[LocalLLM] Device: {self.device}")
    
    def _verify_gpu(self):
        """Verify GPU availability and log details."""
        if not torch.cuda.is_available():
            logger.warning("[LocalLLM] WARNING: CUDA unavailable - using CPU")
            self._gpu_info = {
                'available': False,
                'name': 'N/A',
                'vram_gb': 0,
                'device': 'cpu'
            }
            return
        
        gpu_count = torch.cuda.device_count()
        logger.info(f"[LocalLLM] CUDA available: {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_props = torch.cuda.get_device_properties(i)
            vram_gb = gpu_props.total_memory / (1024**3)
            
            logger.info(f"[LocalLLM] GPU {i}: {gpu_name}")
            logger.info(f"[LocalLLM] GPU {i} VRAM: {vram_gb:.2f} GB")
            logger.info(f"[LocalLLM] GPU {i} Compute Capability: {gpu_props.major}.{gpu_props.minor}")
            
            self._gpu_info = {
                'available': True,
                'name': gpu_name,
                'vram_gb': vram_gb,
                'device': f'cuda:{i}'
            }
            
            # Warn if VRAM is limited
            if vram_gb < 8:
                logger.warning(f"[LocalLLM] WARNING: GPU VRAM ({vram_gb:.2f} GB) is below recommended 8GB")
    
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
        
        logger.info(f"[LocalLLM] Loading model: {self.model_name}")
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
            logger.info(f"[PERF][STARTUP] LocalLLM model load: {load_time*1000:.0f} ms")
            logger.info(f"[LocalLLM] Using device: {self.device}")
            logger.info("[ARCH] LocalLLM model loaded and cached")
            
            # Log actual device being used
            if hasattr(self._model, 'device'):
                logger.info(f"[GPU][LLM] device={self._model.device}")
                logger.info(f"[GPU][LLM] model_device={self._model.device}")
            
            if self.device == "cpu":
                logger.warning("[LocalLLM] WARNING: CUDA unavailable - using CPU")
            
            self._model_loaded = True
            logger.info(f"[LocalLLM] Model loaded successfully")
            
        except Exception as e:
            logger.error(f"[LocalLLM] Failed to load model: {str(e)}")
            import traceback
            logger.error(f"[LocalLLM] Traceback:\n{traceback.format_exc()}")
            raise Exception(f"Failed to load LLM model: {str(e)}")
    
    def generate_content(
        self,
        prompt: str,
        system_instruction: str = None,
        temperature: float = 0.7,
        response_format: str = "text"
    ) -> str:
        """
        Generate content from the LLM.
        
        Args:
            prompt: The input prompt
            system_instruction: Optional system instruction
            temperature: Sampling temperature
            response_format: Response format ("text" or "json")
        
        Returns:
            Generated text content
        """
        try:
            import torch
            request_start = time.perf_counter()
            
            # Ensure model is loaded
            self._load_model()
            
            # Log GPU usage before inference
            if self.device == "cuda":
                if torch.cuda.is_available():
                    allocated_before = torch.cuda.memory_allocated(0) / 1024**2
                    reserved_before = torch.cuda.memory_reserved(0) / 1024**2
                    logger.info(f"[GPU][LLM] VRAM before inference={allocated_before:.0f} MB")
                    logger.info(f"[GPU][LLM] VRAM reserved before={reserved_before:.0f} MB")
            
            logger.info(f"[PERF][LOCAL_LLM] Request start")
            logger.info(f"======== LOCAL LLM Request ========")
            logger.info(f"Model: {self.model_name}")
            logger.info(f"Device: {self.device}")
            if hasattr(self._model, 'device'):
                logger.info(f"Model device: {self._model.device}")
            logger.info(f"Quantization: {self.quantization}")
            logger.info(f"Temperature: {temperature}")
            logger.info(f"Response Format: {response_format}")
            logger.info(f"System Instruction: {system_instruction[:200] if system_instruction else 'None'}...")
            logger.info(f"Prompt: {prompt[:200]}...")
            logger.info(f"================================")
            
            # Build messages
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})
            
            # Apply chat template for both text and JSON
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self._model.device)
            
            # Generate
            inference_start = time.perf_counter()
            
            # Adjust max_new_tokens for JSON to ensure enough space
            max_tokens = 512 if response_format == "json" else 256
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id
                )
            
            inference_time = time.perf_counter() - inference_start
            
            # Decode only newly generated tokens
            input_length = inputs['input_ids'].shape[1]
            generated_ids = outputs[0][input_length:]
            
            # Log token counts
            logger.info(f"[LOCAL_LLM][TOKENS] Input tokens: {input_length}")
            logger.info(f"[LOCAL_LLM][TOKENS] Generated tokens: {len(generated_ids)}")
            
            # Decode
            generated_text = self._tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            # Log raw generated text before any processing
            logger.info(f"[LOCAL_LLM][RAW_GENERATED] '{generated_text}'")
            
            # Log GPU usage after inference
            if self.device == "cuda":
                if torch.cuda.is_available():
                    allocated_after = torch.cuda.memory_allocated(0) / 1024**2
                    reserved_after = torch.cuda.memory_reserved(0) / 1024**2
                    logger.info(f"[GPU][LLM] VRAM after inference={allocated_after:.0f} MB")
                    logger.info(f"[GPU][LLM] VRAM reserved after={reserved_after:.0f} MB")
            
            total_time = time.perf_counter() - request_start
            logger.info(f"[PERF][LOCAL_LLM] Inference: {inference_time*1000:.0f} ms")
            logger.info(f"[PERF][LOCAL_LLM] Generation: {total_time*1000:.0f} ms")
            
            logger.info(f"======== LOCAL LLM Response ========")
            logger.info(f"Latency: {total_time:.2f}s")
            logger.info(f"Raw Response: {generated_text[:500]}...")
            logger.info(f"================================")
            
            return generated_text

        except Exception as e:
            logger.error(f"[LocalLLM] Exception type: {type(e).__name__}")
            logger.error(f"[LocalLLM] Exception message: {str(e)}")
            import traceback
            logger.error(f"[LocalLLM] Traceback:\n{traceback.format_exc()}")
            raise Exception(f"Local LLM inference error: {str(e)}")
    
    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate JSON content using local LLM.
        
        Args:
            prompt: The user prompt to send.
            system_instruction: Optional system instruction.
            temperature: Sampling temperature (0.0 to 1.0).
        
        Returns:
            The generated content as a dictionary.
        """
        # Default JSON system instruction if not provided
        if system_instruction is None:
            system_instruction = """You are an interview evaluation engine. Return ONLY valid JSON. Do not include markdown. Do not include ```json. Do not include explanations outside the JSON."""
        
        # Try generation with retry for empty responses
        max_retries = 2
        for attempt in range(max_retries):
            if attempt > 0:
                # Retry with stronger JSON instruction but preserve the full prompt
                logger.warning(f"[LocalLLM][JSON] Retry {attempt + 1}/{max_retries} with stronger JSON instruction")
                system_instruction = "Return ONLY the JSON object. No markdown. No explanation. Follow the supplied policy and difficulty exactly. Do not simplify or remove any instructions from the prompt."
                # Keep the full prompt - do NOT truncate it
            
            response_text = self.generate_content(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=temperature,
                response_format="json"
            )
            
            # Log raw response for debugging
            logger.info("======================== RAW LOCAL LLM TEXT RESPONSE ========================")
            logger.info(f"Type: {type(response_text)}")
            logger.info(f"Repr: {repr(response_text)}")
            logger.info(f"Content: {response_text}")
            logger.info("========================================================================")
            
            # Check for empty response
            if not response_text or response_text.strip() == "":
                logger.warning(f"[LOCAL_LLM][JSON] Empty response on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    continue
                else:
                    raise ValueError("LLM returned empty response after retries")
            
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
            
            # Find first '{' and last '}' to extract JSON substring
            first_brace = json_text.find('{')
            last_brace = json_text.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_text = json_text[first_brace:last_brace + 1]
                logger.info(f"[LOCAL_LLM][JSON] Extracted JSON substring from positions {first_brace} to {last_brace}")
            
            try:
                parsed_json = json.loads(json_text)
                logger.info(f"[LOCAL_LLM][JSON] Parsed successfully: True")
                logger.info(f"[LOCAL_LLM][JSON] Parsed object: {parsed_json}")
                
                # Validate required fields for semantic evaluation (only if present)
                required_fields = ["correctness_score", "concept_coverage", "reasoning_score", "missing_concepts", "difficulty"]
                missing_fields = [f for f in required_fields if f not in parsed_json]
                if missing_fields and all(field in parsed_json for field in ["question", "topic"]):
                    # This is a question generation response, not semantic evaluation
                    logger.info(f"[LOCAL_LLM][JSON] Question generation response detected (has 'question' and 'topic' fields)")
                elif missing_fields:
                    logger.warning(f"[LOCAL_LLM][JSON] Missing required semantic fields: {missing_fields}")
                
                # Validate numerical ranges only if fields are present
                if "correctness_score" in parsed_json:
                    if not 0 <= parsed_json["correctness_score"] <= 100:
                        logger.warning(f"[LOCAL_LLM][JSON] correctness_score out of range: {parsed_json['correctness_score']}")
                if "concept_coverage" in parsed_json:
                    if not 0 <= parsed_json["concept_coverage"] <= 100:
                        logger.warning(f"[LOCAL_LLM][JSON] concept_coverage out of range: {parsed_json['concept_coverage']}")
                if "reasoning_score" in parsed_json:
                    if not 0 <= parsed_json["reasoning_score"] <= 100:
                        logger.warning(f"[LOCAL_LLM][JSON] reasoning_score out of range: {parsed_json['reasoning_score']}")
                if "missing_concepts" in parsed_json:
                    if not 0 <= parsed_json["missing_concepts"] <= 8:
                        logger.warning(f"[LOCAL_LLM][JSON] missing_concepts out of range: {parsed_json['missing_concepts']}")
                
                return parsed_json
            except json.JSONDecodeError as e:
                logger.error(f"[LOCAL_LLM][JSON] Failed to parse JSON: {e}")
                logger.error(f"[LOCAL_LLM][JSON] Cleaned JSON text: {json_text}")
                if attempt < max_retries - 1:
                    continue
                else:
                    raise ValueError(
                        f"Failed to parse LLM response as JSON: {e}\n"
                        f"Response was: {response_text}"
                    )
        
        raise ValueError("Failed to generate valid JSON after retries")
