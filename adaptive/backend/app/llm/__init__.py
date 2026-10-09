"""
LLM module for the Semantic Interview Engine.
"""

import os
from typing import Any
from .interview_engine import InterviewEngine
from .nvidia_client import NVIDIAClient, NVIDIANIMReasoningClient, NVIDIANIMQuestionClient
from .local_llm_client import LocalLLMClient
from .question_generator import QuestionGenerator
from .semantic_evaluator import SemanticEvaluator
from .schemas import (
    FirstQuestionRequest,
    FirstQuestionResponse,
    AnswerEvaluationRequest,
    AnswerEvaluationResponse,
    SemanticEvaluation
)

__all__ = [
    'InterviewEngine',
    'NVIDIAClient',
    'NVIDIANIMReasoningClient',
    'NVIDIANIMQuestionClient',
    'LocalLLMClient',
    'QuestionGenerator',
    'SemanticEvaluator',
    'FirstQuestionRequest',
    'FirstQuestionResponse',
    'AnswerEvaluationRequest',
    'AnswerEvaluationResponse',
    'SemanticEvaluation',
    'get_llm_client'
]


def get_llm_client(task: str = "semantic") -> Any:
    """
    Factory function to get the appropriate LLM client based on configuration.
    
    Supports task-specific providers via SEMANTIC_LLM_PROVIDER and QUESTION_LLM_PROVIDER.
    Falls back to global LLM_PROVIDER if task-specific not set.
    For local_qwen provider, supports task-specific models via SEMANTIC_MODEL and QUESTION_MODEL.
    
    Args:
        task: Task type - "semantic" for evaluation, "question" for generation
    
    Returns:
        Appropriate LLM client instance
    """
    # Check for task-specific provider first
    if task == "semantic":
        provider = os.getenv("SEMANTIC_LLM_PROVIDER") or os.getenv("LLM_PROVIDER", "local_qwen")
    elif task == "question":
        provider = os.getenv("QUESTION_LLM_PROVIDER") or os.getenv("LLM_PROVIDER", "local_qwen")
    else:
        raise ValueError(f"Unknown task type: {task}")
    
    provider = provider.lower()
    
    if provider == "nvidia_nim":
        if task == "semantic":
            return NVIDIANIMReasoningClient()
        elif task == "question":
            return NVIDIANIMQuestionClient()
        else:
            raise ValueError(f"Unknown task type: {task}")
    elif provider == "local_qwen":
        # Import create_llm_client here to avoid circular dependency
        from .llm_factory import create_llm_client
        return create_llm_client(provider="local_qwen", task=task)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
