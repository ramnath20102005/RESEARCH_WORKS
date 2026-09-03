"""
Question generator for the Semantic Interview Engine.
Generates the first interview question based on parsed resume data.
Generates adaptive next questions based on TabPFN policy predictions.
"""

import json
import logging
from typing import Dict, Any, Protocol, List, Optional
from .prompts import FIRST_QUESTION_SYSTEM_PROMPT, FIRST_QUESTION_USER_PROMPT, NEXT_QUESTION_SYSTEM_PROMPT, NEXT_QUESTION_USER_PROMPT
from .schemas import FirstQuestionResponse

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


class QuestionGenerator:
    """Generates interview questions using an LLM client."""
    
    def __init__(self, llm_client: LLMClientProtocol):
        """
        Initialize the question generator.
        
        Args:
            llm_client: An instance of an LLM client (NVIDIA, Gemini, etc.).
        """
        self.client = llm_client
    
    def generate_first_question(self, resume_data: Dict[str, Any]) -> FirstQuestionResponse:
        """
        Generate the first interview question based on resume data.
        
        Args:
            resume_data: Parsed resume JSON containing skills, projects, etc.
        
        Returns:
            FirstQuestionResponse with the generated question, topic, difficulty, and source.
        """
        # Format resume data for the prompt
        resume_text = json.dumps(resume_data, indent=2)
        
        # Build the prompt
        prompt = FIRST_QUESTION_USER_PROMPT.format(resume_data=resume_text)
        
        # Generate response from Gemini
        try:
            response_json = self.client.generate_json(
                prompt=prompt,
                system_instruction=FIRST_QUESTION_SYSTEM_PROMPT,
                temperature=0.7
            )
            
            # Validate required fields
            required_fields = ["question", "topic", "difficulty", "source"]
            for field in required_fields:
                if field not in response_json:
                    raise ValueError(f"Missing required field in response: {field}")
            
            # Validate source field
            valid_sources = ["Project", "Internship", "Skill"]
            if response_json["source"] not in valid_sources:
                raise ValueError(f"Invalid source field: {response_json['source']}. Must be one of {valid_sources}")
            
            # Validate difficulty
            if response_json["difficulty"] != "Easy":
                raise ValueError(f"First question must be Easy difficulty, got: {response_json['difficulty']}")
            
            # Validate and return response
            return FirstQuestionResponse(**response_json)
            
        except Exception as e:
            raise Exception(f"Failed to generate first question: {str(e)}")
    
    def generate_next_question(
        self,
        policy: str,
        topic: str,
        current_difficulty: str,
        previous_question: str,
        candidate_answer: str,
        correctness_score: int,
        concept_coverage: int,
        reasoning_score: int,
        missing_concepts: int,
        correct_streak: int,
        wrong_streak: int,
        resume_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate the next interview question based on TabPFN policy prediction.
        
        Args:
            policy: The predicted policy from TabPFN
            topic: The current interview topic
            current_difficulty: The current difficulty level
            previous_question: The previous question asked
            candidate_answer: The candidate's previous answer
            correctness_score: Semantic evaluation correctness score
            concept_coverage: Semantic evaluation concept coverage
            reasoning_score: Semantic evaluation reasoning score
            missing_concepts: Number of missing concepts from semantic evaluation
            correct_streak: Current correct answer streak
            wrong_streak: Current wrong answer streak
            resume_data: Optional resume data for topic switching
        
        Returns:
            Dictionary containing the next question, topic, difficulty, and policy
        """
        # Build the prompt with new parameter structure
        prompt = NEXT_QUESTION_USER_PROMPT.format(
            topic=topic,
            previous_question=previous_question,
            candidate_answer=candidate_answer,
            correctness_score=correctness_score,
            concept_coverage=concept_coverage,
            reasoning_score=reasoning_score,
            missing_concepts=missing_concepts,
            current_difficulty=current_difficulty,
            correct_streak=correct_streak,
            wrong_streak=wrong_streak,
            policy=policy
        )
        
        # Add resume context if policy is Switch Topic
        if policy == "Switch Topic" and resume_data:
            resume_text = json.dumps(resume_data, indent=2)
            prompt += f"\n\nCandidate Background:\n{resume_text}"
        
        # Log question generation input
        logger.info("")
        logger.info("=" * 70)
        logger.info("[QUESTION_GENERATION]")
        logger.info("=" * 70)
        logger.info(f"TabPFN Policy: {policy}")
        logger.info(f"Effective Difficulty: {current_difficulty}")
        logger.info(f"Current Difficulty: {current_difficulty}")
        logger.info(f"Topic: {topic}")
        logger.info(f"Previous Question: {previous_question}")
        logger.info("=" * 70)
        logger.info("")
        
        # Generate response from LLM
        try:
            response_json = self.client.generate_json(
                prompt=prompt,
                system_instruction=NEXT_QUESTION_SYSTEM_PROMPT,
                temperature=0.7
            )
            
            # Log LLM output
            logger.info("")
            logger.info("=" * 70)
            logger.info("[QUESTION_GENERATION][LLM_OUTPUT]")
            logger.info("=" * 70)
            logger.info(f"Question: {response_json.get('question', 'N/A')}")
            logger.info(f"Returned Policy: {response_json.get('policy', 'N/A')}")
            logger.info(f"Returned Difficulty: {response_json.get('difficulty', 'N/A')}")
            logger.info(f"Returned Topic: {response_json.get('topic', 'N/A')}")
            logger.info(f"Returned Source: {response_json.get('source', 'N/A')}")
            logger.info("=" * 70)
            logger.info("")
            
            # Validate that returned JSON includes and matches the TabPFN policy
            returned_policy = response_json.get("policy")
            if returned_policy != policy:
                logger.error(f"[QUESTION_GENERATION][REJECTED]")
                logger.error(f"Reason: Policy mismatch")
                logger.error(f"Expected Policy: {policy}")
                logger.error(f"Returned Policy: {returned_policy}")
                raise ValueError(
                    f"Policy validation failed: TabPFN predicted '{policy}', "
                    f"but LLM returned '{returned_policy}'. "
                    f"The LLM must follow the exact policy supplied by TabPFN."
                )
            
            # Validate that returned difficulty matches the effective difficulty
            returned_difficulty = response_json.get("difficulty")
            if returned_difficulty != current_difficulty:
                logger.error(f"[QUESTION_GENERATION][REJECTED]")
                logger.error(f"Reason: Difficulty mismatch")
                logger.error(f"Expected Difficulty: {current_difficulty}")
                logger.error(f"Returned Difficulty: {returned_difficulty}")
                raise ValueError(
                    f"Difficulty validation failed: expected '{current_difficulty}', "
                    f"but LLM returned '{returned_difficulty}'. "
                    f"The LLM must use the exact difficulty supplied by the policy application."
                )
            
            # Validate required fields
            required_fields = ["question", "difficulty", "topic", "policy", "source"]
            for field in required_fields:
                if field not in response_json:
                    logger.error(f"[QUESTION_GENERATION][REJECTED]")
                    logger.error(f"Reason: Missing required field: {field}")
                    raise ValueError(f"Missing required field in LLM response: {field}")
            
            # Validate question is not identical to previous question
            if response_json.get('question') == previous_question:
                logger.error(f"[QUESTION_GENERATION][REJECTED]")
                logger.error(f"Reason: Question identical to previous question")
                raise ValueError("Generated question is identical to previous question")
            
            # Log successful validation
            logger.info("")
            logger.info("=" * 70)
            logger.info("[QUESTION_GENERATION][VALIDATION]")
            logger.info("=" * 70)
            logger.info(f"Policy Match: True")
            logger.info(f"Difficulty Match: True")
            logger.info(f"Question Different: True")
            logger.info("=" * 70)
            logger.info("")
            
            # Log successful question generation
            logger.info("")
            logger.info("=" * 60)
            logger.info("[NEXT QUESTION GENERATION]")
            logger.info("=" * 60)
            logger.info(f"Policy received from TabPFN: {policy}")
            logger.info(f"Effective difficulty: {current_difficulty}")
            logger.info(f"Generated question: {response_json.get('question')}")
            logger.info(f"Generated difficulty: {response_json.get('difficulty')}")
            logger.info(f"Generated topic: {response_json.get('topic')}")
            logger.info(f"Returned policy: {response_json.get('policy')}")
            logger.info(f"Source: {response_json.get('source')}")
            logger.info(f"Policy validation: PASSED")
            logger.info(f"Difficulty validation: PASSED")
            logger.info("=" * 60)
            logger.info("")
            
            return response_json
            
        except Exception as e:
            raise Exception(f"Failed to generate next question: {str(e)}")
