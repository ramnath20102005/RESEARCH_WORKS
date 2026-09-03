"""
Interview engine orchestrator for the Semantic Interview Engine.
Coordinates question generation and answer evaluation.
"""

import json
import logging
from .nvidia_client import NVIDIAClient
from .question_generator import QuestionGenerator
from .semantic_evaluator import SemanticEvaluator
from .schemas import FirstQuestionResponse, AnswerEvaluationResponse
from .prompts import FIRST_QUESTION_SYSTEM_PROMPT, FIRST_QUESTION_USER_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterviewEngine:
    """
    Main orchestrator for the semantic interview engine.
    Coordinates LLM interactions for question generation and evaluation.
    """
    
    def __init__(self, model_name: str = "meta/llama-3.1-8b-instruct"):
        """
        Initialize the interview engine.
        
        Args:
            model_name: The NVIDIA NIM model to use.
        """
        self.llm_client = NVIDIAClient(model_name=model_name)
        self.question_generator = QuestionGenerator(self.llm_client)
        self.semantic_evaluator = SemanticEvaluator(self.llm_client)
    
    def generate_first_question(self, resume_data: dict) -> FirstQuestionResponse:
        """
        Generate the first interview question based on resume data.
        
        Args:
            resume_data: Parsed resume JSON containing skills, projects, education, etc.
        
        Returns:
            FirstQuestionResponse with the generated question, topic, and difficulty.
        """
        try:
            import time
            logger.info("[Interview] Generating first question from resume data")
            start_time = time.time()
            
            # Use the question generator service
            question_response = self.question_generator.generate_first_question(resume_data)
            
            total_time = time.time() - start_time
            logger.info(f"[Interview] Total question generation time: {total_time:.2f}s")
            logger.info(f"[Interview] Generated question: {question_response.question[:50]}...")
            logger.info(f"[Interview] Topic: {question_response.topic}, Difficulty: {question_response.difficulty}, Source: {question_response.source}")
            
            return question_response
            
        except Exception as e:
            logger.error(f"[Interview] Failed to generate first question: {str(e)}")
            raise Exception(f"Failed to generate first question: {str(e)}")
    
    def evaluate_answer(
        self,
        question: str,
        topic: str,
        difficulty: str,
        answer: str
    ):
        """
        Evaluate a candidate's answer.
        
        Args:
            question: The interview question.
            topic: The question topic.
            difficulty: The question difficulty.
            answer: The candidate's answer.
        
        Returns:
            AnswerEvaluationResponse with semantic evaluation.
        """
        return self.semantic_evaluator.evaluate_answer(
            question=question,
            topic=topic,
            difficulty=difficulty,
            answer=answer
        )
