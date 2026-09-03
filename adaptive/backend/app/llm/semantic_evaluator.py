"""
Semantic evaluator for the Semantic Interview Engine.
Evaluates candidate answers using LLM and returns semantic features.
"""

import logging
from typing import Dict, Any, Protocol, Optional
from .prompts import SEMANTIC_EVAL_SYSTEM_PROMPT, SEMANTIC_EVAL_USER_PROMPT
from .schemas import AnswerEvaluationResponse, SemanticEvaluation

logger = logging.getLogger(__name__)


class LLMClientProtocol(Protocol):
    """Protocol for LLM clients to ensure interface compatibility."""
    
    def generate_json(
        self,
        prompt: str,
        system_instruction: str = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate JSON response from LLM."""
        ...


class SemanticEvaluator:
    """Evaluates candidate answers semantically using an LLM client."""
    
    def __init__(self, llm_client: LLMClientProtocol):
        """
        Initialize the semantic evaluator.
        
        Args:
            llm_client: An instance of an LLM client (NVIDIA, Gemini, etc.).
        """
        self.client = llm_client
    
    def evaluate_answer(
        self,
        question: str,
        topic: str,
        current_difficulty: str,
        answer: str,
        question_number: int = 1,
        correct_streak: int = 0,
        wrong_streak: int = 0
    ) -> Dict[str, Any]:
        """
        Evaluate a candidate's answer to an interview question.
        
        Args:
            question: The interview question asked.
            topic: The topic of the question.
            current_difficulty: The current difficulty level of the question.
            answer: The candidate's transcribed answer.
            question_number: Current question number in the interview.
            correct_streak: Current correct answer streak.
            wrong_streak: Current wrong answer streak.
        
        Returns:
            Dictionary containing semantic features, question difficulty, and feedback.
        """
        # Validate inputs
        if not answer or answer.strip() == "":
            raise ValueError("Answer cannot be empty")
        
        if not question or question.strip() == "":
            raise ValueError("Question cannot be empty")
        
        # Build the prompt with context
        prompt = SEMANTIC_EVAL_USER_PROMPT.format(
            question=question,
            answer=answer,
            difficulty=current_difficulty
        )
        
        # Generate evaluation from LLM
        try:
            response_json = self.client.generate_json(
                prompt=prompt,
                system_instruction=SEMANTIC_EVAL_SYSTEM_PROMPT,
                temperature=0.5  # Lower temperature for more consistent evaluation
            )
            
            # Log raw LLM response
            logger.info("")
            logger.info("=" * 60)
            logger.info("[LLM RAW RESPONSE]")
            logger.info("=" * 60)
            logger.info(f"{response_json}")
            logger.info("=" * 60)
            logger.info("")
            
            # Extract semantic features from flat JSON schema
            # New schema: {correctness_score, concept_coverage, reasoning_score, missing_concepts, difficulty, is_correct}
            correctness_score = int(response_json.get('correctness_score', 50))
            concept_coverage = int(response_json.get('concept_coverage', 50))
            reasoning_score = int(response_json.get('reasoning_score', 50))
            missing_concepts = int(response_json.get('missing_concepts', 3))
            is_correct = bool(response_json.get('is_correct', False))
            question_difficulty = response_json.get('difficulty', current_difficulty)
            
            # Log LLM to semantic features mapping
            logger.info("")
            logger.info("=" * 60)
            logger.info("[LLM → SEMANTIC FEATURES]")
            logger.info("=" * 60)
            logger.info(f"Correctness Score = {correctness_score}")
            logger.info(f"Concept Coverage = {concept_coverage}")
            logger.info(f"Reasoning Score = {reasoning_score}")
            logger.info(f"Missing Concepts = {missing_concepts}")
            logger.info(f"Is Correct = {is_correct}")
            logger.info(f"Question Difficulty = {question_difficulty}")
            logger.info("=" * 60)
            logger.info("")
            
            # Ensure scores are within valid dataset ranges
            correctness_score = max(0, min(100, correctness_score))
            concept_coverage = max(0, min(100, concept_coverage))
            reasoning_score = max(0, min(100, reasoning_score))
            missing_concepts = max(0, min(8, missing_concepts))
            
            # Validate correlation consistency (dataset rule)
            if correctness_score > 95 and missing_concepts > 3:
                # Force consistency if LLM violates dataset rule
                missing_concepts = min(missing_concepts, 3)
            
            # Validate difficulty
            valid_difficulties = ['Easy', 'Medium', 'Hard']
            if question_difficulty not in valid_difficulties:
                question_difficulty = current_difficulty
            
            return {
                'semantic': {
                    'correctness_score': correctness_score,
                    'concept_coverage': concept_coverage,
                    'reasoning_score': reasoning_score,
                    'missing_concepts': missing_concepts,
                    'is_correct': is_correct
                },
                'question_assessment': {
                    'question_difficulty': question_difficulty
                }
            }
            
        except Exception as e:
            raise Exception(f"Failed to evaluate answer: {str(e)}")
