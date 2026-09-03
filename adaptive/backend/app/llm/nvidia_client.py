"""
NVIDIA NIM client for the Semantic Interview Engine.
Provides a reusable interface for interacting with NVIDIA's NIM API using OpenAI-compatible client.
"""

import os
import json
import logging
import time
from pathlib import Path
from openai import OpenAI
from typing import Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NVIDIAClient:
    """Client for interacting with NVIDIA NIM API using OpenAI-compatible interface."""
    
    def __init__(self, model_name: str = "meta/llama-3.1-8b-instruct"):
        """
        Initialize the NVIDIA NIM client.
        
        Args:
            model_name: The NVIDIA NIM model to use. Default is meta/llama-3.1-8b-instruct.
        """
        self.model_name = model_name
        self.api_key = os.getenv("NVIDIA_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "NVIDIA_API_KEY environment variable not set. "
                "Please set it in .env file or as environment variable."
            )
        
        # Initialize OpenAI client configured for NVIDIA NIM
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key
        )
        logger.info(f"[NVIDIAClient] Initialized with model: {self.model_name}")
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def generate_content(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        response_format: str = "text"
    ) -> str:
        """
        Generate content using NVIDIA NIM.
        
        Args:
            prompt: The user prompt to send.
            system_instruction: Optional system instruction.
            temperature: Sampling temperature (0.0 to 1.0).
            response_format: Expected response format ("text" or "json").
        
        Returns:
            The generated content as a string.
        """
        try:
            logger.info(f"[PERF] llm_request_start: 0.000s")
            request_start = time.perf_counter()
            
            logger.info(f"======== NVIDIA NIM Request ========")
            logger.info(f"Model: {self.model_name}")
            logger.info(f"Temperature: {temperature}")
            logger.info(f"Response Format: {response_format}")
            logger.info(f"System Instruction: {system_instruction[:100] if system_instruction else 'None'}...")
            logger.info(f"Prompt: {prompt[:200]}...")
            logger.info(f"================================")
            
            # Build messages array
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})
            
            api_start = time.perf_counter()
            
            # Generate content with NVIDIA NIM
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=0.7,
                max_tokens=1024,
                stream=False
            )
            
            api_time = time.perf_counter() - api_start
            logger.info(f"[PERF] llm_api_call: {api_time:.3f}s")
            
            # Extract text from response
            result_text = completion.choices[0].message.content
            
            total_time = time.perf_counter() - request_start
            logger.info(f"[PERF] llm_total: {total_time:.3f}s")
            
            logger.info(f"======== NVIDIA NIM Response ========")
            logger.info(f"Latency: {total_time:.2f}s")
            logger.info(f"Raw Response: {result_text[:500]}...")
            logger.info(f"================================")
            
            return result_text
            
        except Exception as e:
            logger.error(f"[NVIDIA] Exception type: {type(e).__name__}")
            logger.error(f"[NVIDIA] Exception message: {str(e)}")
            import traceback
            logger.error(f"[NVIDIA] Traceback:\n{traceback.format_exc()}")
            raise Exception(f"NVIDIA NIM API error: {str(e)}")
    
    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate JSON content using NVIDIA NIM.
        
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
        logger.info("======================== RAW NVIDIA TEXT RESPONSE ========================")
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
            logger.error(f"Failed to parse NVIDIA response as JSON: {e}")
            logger.error(f"Response was: {response_text}")
            raise ValueError(
                f"Failed to parse NVIDIA response as JSON: {e}\n"
                f"Response was: {response_text}"
            )
