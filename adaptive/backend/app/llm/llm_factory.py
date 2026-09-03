"""
LLM Factory for provider abstraction.
Creates LLM client instances based on configuration.
"""

import os
import logging
from typing import Protocol
from .nvidia_client import NVIDIAClient
from .local_llm_client import LocalLLMClient

logger = logging.getLogger(__name__)


class LLMClientProtocol(Protocol):
    """Protocol for LLM clients to ensure interface compatibility."""
    
    def generate_json(
        self,
        prompt: str,
        system_instruction: str = None,
        temperature: float = 0.7
    ) -> dict:
        """Generate JSON response from LLM."""
        ...
    
    def generate_content(
        self,
        prompt: str,
        system_instruction: str = None,
        temperature: float = 0.7,
        response_format: str = "text"
    ) -> str:
        """Generate content from LLM."""
        ...


def create_llm_client(provider: str = None) -> LLMClientProtocol:
    """
    Create an LLM client based on the specified provider.
    
    Args:
        provider: The LLM provider to use. Options: "local_qwen", "nvidia_nim".
                  If None, reads from LLM_PROVIDER environment variable.
    
    Returns:
        An LLM client instance implementing LLMClientProtocol.
    
    Raises:
        ValueError: If provider is invalid or not configured.
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "local_qwen")
    
    provider = provider.lower()
    
    logger.info(f"[LLMFactory] Creating LLM client with provider: {provider}")
    
    if provider == "local_qwen":
        model_name = os.getenv("LOCAL_QWEN_MODEL", "Qwen/Qwen1.5-4B-Chat")
        device = os.getenv("LOCAL_QWEN_DEVICE", None)
        quantization = os.getenv("LOCAL_QWEN_QUANTIZATION", "int4")
        
        logger.info(f"[LLMFactory] Local LLM configuration:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Device: {device if device else 'auto-detect'}")
        logger.info(f"  Quantization: {quantization}")
        
        return LocalLLMClient(
            model_name=model_name,
            device=device,
            quantization=quantization
        )
    
    elif provider == "nvidia_nim":
        model_name = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
        
        logger.info(f"[LLMFactory] NVIDIA NIM configuration:")
        logger.info(f"  Model: {model_name}")
        
        return NVIDIAClient(model_name=model_name)
    
    else:
        raise ValueError(
            f"Invalid LLM provider: {provider}. "
            f"Supported providers: local_qwen, nvidia_nim"
        )
