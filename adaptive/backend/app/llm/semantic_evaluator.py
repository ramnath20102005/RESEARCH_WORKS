"""
Semantic evaluator for the Semantic Interview Engine.
Evaluates candidate answers using LLM and returns semantic features.
"""

import json
import logging
from typing import Dict, Any, Protocol, Optional, List
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
        required_concepts: list = None,
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
            required_concepts: List of required concepts for this question.
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
        
        # Default required concepts if not provided
        if required_concepts is None:
            required_concepts = []
        
        # Deterministic handling for explicit no-answer patterns
        # Only apply fallback for genuine non-answers, not partial answers with substantive content
        normalized_answer = answer.strip().lower()
        no_answer_patterns = [
            "i don't know",
            "i do not know",
            "i don't know the answer",
            "not sure",
            "no idea",
            "cannot answer",
            "i can't answer",
            "i don't remember",
            "i'm not sure",
            "i have no idea",
            "i have no clue",
            "i can't recall",
            "i'm unable to answer",
            "i don't know how to answer this"
        ]
        
        has_no_answer_pattern = any(pattern in normalized_answer for pattern in no_answer_patterns)
        
        # Only treat as no-answer if:
        # 1. Contains no-answer pattern AND
        # 2. Answer is very short (< 60 chars) OR consists mostly of uncertainty phrases
        # This allows partial answers with substantive content to be evaluated by the LLM
        if has_no_answer_pattern:
            answer_length = len(answer.strip())
            # Calculate proportion of uncertainty phrases in the answer
            uncertainty_phrase_chars = sum(len(pattern) for pattern in no_answer_patterns if pattern in normalized_answer)
            uncertainty_ratio = uncertainty_phrase_chars / max(answer_length, 1)
            
            # Treat as genuine no-answer if very short or mostly uncertainty phrases
            is_genuine_no_answer = (answer_length < 60) or (uncertainty_ratio > 0.5)
            
            if is_genuine_no_answer:
                # Return deterministic low scores for explicit no-answer
                # Still pass through normal TabPFN pipeline - do NOT hardcode policy
                # Use actual count of required concepts as missing (all are missing)
                missing_concepts_count = len(required_concepts) if required_concepts else 8
                
                logger.info("")
                logger.info("=" * 60)
                logger.info("[DETECTED GENUINE NO-ANSWER]")
                logger.info("=" * 60)
                logger.info(f"Answer: {answer}")
                logger.info(f"Answer length: {answer_length} chars")
                logger.info(f"Uncertainty ratio: {uncertainty_ratio:.2f}")
                logger.info("Applying deterministic low scores for explicit no-answer")
                logger.info(f"Returning: correctness=5, coverage=5, reasoning=5, missing={missing_concepts_count}")
                logger.info("=" * 60)
                logger.info("")
                return {
                    'semantic': {
                        'correctness_score': 5,  # Very low (0-100 scale)
                        'concept_coverage': 5,  # Very low (0-100 scale)
                        'reasoning_score': 5,  # Very low (0-100 scale)
                        'missing_concepts': missing_concepts_count,  # Count of required concepts
                        'missing_concept_list': required_concepts.copy() if required_concepts else [],
                        'is_correct': False
                    },
                    'question_assessment': {
                        'question_difficulty': current_difficulty
                    }
                }
            else:
                logger.info("")
                logger.info("=" * 60)
                logger.info("[NO-ANSWER PATTERN DETECTED BUT ANSWER HAS SUBSTANTIVE CONTENT]")
                logger.info("=" * 60)
                logger.info(f"Answer: {answer}")
                logger.info(f"Answer length: {answer_length} chars")
                logger.info(f"Uncertainty ratio: {uncertainty_ratio:.2f}")
                logger.info("Proceeding to LLM evaluation for partial answer")
                logger.info("=" * 60)
                logger.info("")
        
        # Build the prompt with context
        prompt = SEMANTIC_EVAL_USER_PROMPT.format(
            question=question,
            topic=topic,
            difficulty=current_difficulty,
            required_concepts=json.dumps(required_concepts) if required_concepts else "[]",
            answer=answer
        )
        
        # Generate evaluation from LLM with deterministic settings
        try:
            response_json = self.client.generate_json(
                prompt=prompt,
                system_instruction=SEMANTIC_EVAL_SYSTEM_PROMPT,
                temperature=0.0  # Deterministic evaluation
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
            # Required schema: {correctness_score, concept_coverage, reasoning_score, missing_concepts, missing_concept_list, difficulty, is_correct}
            
            # Extract scores (these are always required)
            correctness_score = int(response_json.get('correctness_score', 50))
            concept_coverage = int(response_json.get('concept_coverage', 50))
            reasoning_score = int(response_json.get('reasoning_score', 50))
            
            # Handle missing_concepts - may be count or list from LLM
            missing_concepts_raw = response_json.get('missing_concepts', 3)
            if isinstance(missing_concepts_raw, list):
                # LLM returned a list instead of count
                missing_concept_list = missing_concepts_raw
                missing_concepts = len(missing_concept_list)
                logger.warning(f"[SEMANTIC_EVAL] LLM returned list for missing_concepts field, converted to count")
            else:
                # LLM returned count
                missing_concepts = int(missing_concepts_raw)
                missing_concept_list = response_json.get('missing_concept_list', [])
            
            # Derive missing_concept_list if missing but missing_concepts is provided
            if 'missing_concept_list' not in response_json and isinstance(missing_concepts_raw, list):
                missing_concept_list = missing_concepts_raw
            elif 'missing_concept_list' not in response_json and missing_concepts > 0 and required_concepts:
                # Derive from required_concepts if we have a count
                missing_concept_list = required_concepts[:missing_concepts]
                logger.warning(f"[SEMANTIC_EVAL] Derived missing_concept_list from required_concepts (first {missing_concepts} items)")
            elif 'missing_concept_list' not in response_json:
                missing_concept_list = []
            
            # Derive is_correct if missing
            if 'is_correct' not in response_json:
                # Derive from correctness_score (threshold at 60)
                is_correct = correctness_score >= 60
                logger.warning(f"[SEMANTIC_EVAL] Derived is_correct={is_correct} from correctness_score={correctness_score}")
            else:
                is_correct = bool(response_json['is_correct'])
            
            # Derive difficulty if missing
            if 'difficulty' not in response_json:
                question_difficulty = current_difficulty
                logger.warning(f"[SEMANTIC_EVAL] Derived difficulty from current_difficulty: {question_difficulty}")
            else:
                question_difficulty = response_json['difficulty']
            
            # Validate types
            if not isinstance(missing_concept_list, list):
                raise ValueError(f"missing_concept_list must be a list, got: {type(missing_concept_list)}")
            
            # Validate count matches list length
            if missing_concepts != len(missing_concept_list):
                logger.warning(f"[SEMANTIC_EVAL] Validation warning: missing_concepts count ({missing_concepts}) does not match missing_concept_list length ({len(missing_concept_list)}). Using count from list.")
                missing_concepts = len(missing_concept_list)
            
            # Log LLM to semantic features mapping
            logger.info("")
            logger.info("=" * 60)
            logger.info("[LLM → SEMANTIC FEATURES]")
            logger.info("=" * 60)
            logger.info(f"Correctness Score = {correctness_score}")
            logger.info(f"Concept Coverage = {concept_coverage}")
            logger.info(f"Reasoning Score = {reasoning_score}")
            logger.info(f"Missing Concepts = {missing_concepts}")
            logger.info(f"Missing Concept List = {missing_concept_list}")
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

            # Heuristic: Override is_correct if it contradicts correctness_score
            # If the LLM returns low scores but sets is_correct=True, correct it
            # If the LLM returns high scores but sets is_correct=False, correct it
            if correctness_score >= 60 and not is_correct:
                logger.info(f"[SEMANTIC_EVAL] Heuristic: Overriding is_correct from False to True (correctness_score={correctness_score} >= 60)")
                is_correct = True
            elif correctness_score < 40 and is_correct:
                logger.info(f"[SEMANTIC_EVAL] Heuristic: Overriding is_correct from True to False (correctness_score={correctness_score} < 40)")
                is_correct = False
            
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
                    'missing_concept_list': missing_concept_list,
                    'is_correct': is_correct
                },
                'question_assessment': {
                    'question_difficulty': question_difficulty
                }
            }
            
        except Exception as e:
            raise Exception(f"Failed to evaluate answer: {str(e)}")
