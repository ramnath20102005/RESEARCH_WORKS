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
    
    def generate_first_question(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate the first interview question based on resume data.
        
        Args:
            resume_data: Parsed resume JSON containing skills, projects, etc.
        
        Returns:
            Dictionary with the generated question, topic, difficulty, source, and required concepts.
        """
        # Format resume data for the prompt
        resume_text = json.dumps(resume_data, indent=2)
        
        # Build the prompt
        prompt = FIRST_QUESTION_USER_PROMPT.format(resume_data=resume_text)
        
        # Generate response from LLM
        try:
            response_json = self.client.generate_json(
                prompt=prompt,
                system_instruction=FIRST_QUESTION_SYSTEM_PROMPT,
                temperature=0.7
            )
            
            # Validate required fields
            required_fields = ["question", "topic", "difficulty", "source", "required_concepts"]
            for field in required_fields:
                if field not in response_json:
                    raise ValueError(f"Missing required field in response: {field}")
            
            # Validate source field
            valid_sources = ["Project", "Internship", "Skill"]
            if response_json["source"] not in valid_sources:
                raise ValueError(f"Invalid source field: {response_json['source']}. Must be one of {valid_sources}")
            
            # Validate and fix difficulty for first question
            # Override to Easy since first question must always be Easy
            if response_json["difficulty"] != "Easy":
                logger.warning(f"[First Question] LLM returned difficulty '{response_json['difficulty']}', overriding to 'Easy'")
                response_json["difficulty"] = "Easy"
            
            # Validate required_concepts is a list
            if not isinstance(response_json["required_concepts"], list):
                raise ValueError(f"required_concepts must be a list, got: {type(response_json['required_concepts'])}")
            
            # Validate required_concepts is not empty
            if len(response_json["required_concepts"]) == 0:
                raise ValueError(f"required_concepts must contain 2-5 concepts, got empty list")
            
            # Log successful generation
            logger.info("")
            logger.info("=" * 60)
            logger.info("[FIRST QUESTION GENERATION]")
            logger.info("=" * 60)
            logger.info(f"Question: {response_json.get('question')}")
            logger.info(f"Topic: {response_json.get('topic')}")
            logger.info(f"Difficulty: {response_json.get('difficulty')}")
            logger.info(f"Source: {response_json.get('source')}")
            logger.info(f"Required Concepts: {response_json.get('required_concepts')}")
            logger.info("=" * 60)
            logger.info("")
            
            return response_json
            
        except Exception as e:
            raise Exception(f"Failed to generate first question: {str(e)}")
    
    def generate_next_question(
        self,
        policy: str,
        topic: str,
        current_difficulty: str,
        effective_difficulty: str,
        previous_question: str,
        candidate_answer: str,
        correctness_score: int,
        concept_coverage: int,
        reasoning_score: int,
        missing_concepts: int,
        missing_concept_list: List[str] = None,
        correct_streak: int = 0,
        wrong_streak: int = 0,
        resume_data: Optional[Dict[str, Any]] = None,
        available_topics: List[str] = None,
        required_concepts: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate the next interview question based on TabPFN policy prediction.

        Args:
            policy: The predicted policy from TabPFN
            topic: The current interview topic
            current_difficulty: The current difficulty level
            effective_difficulty: The effective difficulty after policy application
            previous_question: The previous question asked
            candidate_answer: The candidate's previous answer
            correctness_score: Semantic evaluation correctness score
            concept_coverage: Semantic evaluation concept coverage
            reasoning_score: Semantic evaluation reasoning score
            missing_concepts: Number of missing concepts from semantic evaluation
            missing_concept_list: List of actual missing concept names
            correct_streak: Current correct answer streak
            wrong_streak: Current wrong answer streak
            resume_data: Optional resume data for topic switching
            available_topics: List of available topics for Switch Topic policy
            required_concepts: Required concepts from the current question
        
        Returns:
            Dictionary containing the next question, topic, difficulty, and policy
        """
        # Default missing_concept_list if not provided
        if missing_concept_list is None:
            missing_concept_list = []
        
        # Default available_topics if not provided
        if available_topics is None:
            available_topics = []

        # Default required_concepts if not provided
        if required_concepts is None:
            required_concepts = []

        # Add resume context if policy is Switch Topic
        resume_context = ""
        if policy == "Switch Topic" and resume_data:
            resume_text = json.dumps(resume_data, indent=2)
            resume_context = f"\n\nCandidate Background:\n{resume_text}"

        # Log question generation input
        logger.info("")
        logger.info("=" * 70)
        logger.info("[QUESTION_GENERATION]")
        logger.info("=" * 70)
        logger.info(f"TabPFN Policy: {policy}")
        logger.info(f"Effective Difficulty: {effective_difficulty}")
        logger.info(f"Current Difficulty: {current_difficulty}")
        logger.info(f"Topic: {topic}")
        logger.info(f"Previous Question: {previous_question}")
        logger.info(f"Missing Concept List: {missing_concept_list}")
        logger.info(f"Available Topics: {available_topics}")
        logger.info("=" * 70)
        logger.info("")

        # Retry logic for question generation
        max_retries = 3
        response_json = None

        for attempt in range(max_retries):
            try:
                # Build the prompt with new parameter structure
                # Add retry instruction if this is not the first attempt
                if attempt > 0:
                    prompt_instruction = f"\n\nIMPORTANT: The previous question you generated was identical to the previous question. You MUST generate a DIFFERENT question. Do not repeat the same question. Try a different angle, different difficulty, or different concept within the same topic."
                else:
                    prompt_instruction = ""

                prompt = NEXT_QUESTION_USER_PROMPT.format(
                    policy=policy,
                    effective_difficulty=effective_difficulty,
                    current_difficulty=current_difficulty,
                    topic=topic,
                    previous_question=previous_question,
                    candidate_answer=candidate_answer,
                    correctness_score=correctness_score,
                    concept_coverage=concept_coverage,
                    reasoning_score=reasoning_score,
                    missing_concepts=missing_concepts,
                    missing_concept_list=json.dumps(missing_concept_list) if missing_concept_list else "[]",
                    required_concepts=json.dumps(required_concepts) if required_concepts else "[]",
                    available_topics=json.dumps(available_topics) if available_topics else "[]"
                ) + prompt_instruction + resume_context

                # Generate response from LLM
                response_json = self.client.generate_json(
                    prompt=prompt,
                    system_instruction=NEXT_QUESTION_SYSTEM_PROMPT,
                    temperature=0.7
                )

                # Ensure response is a dictionary
                if not isinstance(response_json, dict):
                    logger.error(f"[QUESTION_GENERATION][ERROR] Response is not a dictionary: {type(response_json)}")
                    logger.error(f"[QUESTION_GENERATION][ERROR] Response content: {str(response_json)[:500]}")
                    if attempt < max_retries - 1:
                        logger.warning(f"[QUESTION_GENERATION][RETRY] Retrying due to non-dict response")
                        continue
                    else:
                        raise ValueError(f"LLM response is not a dictionary after {max_retries} attempts")

                # Log LLM output
                logger.info("")
                logger.info("=" * 70)
                logger.info(f"[QUESTION_GENERATION][LLM_OUTPUT] (Attempt {attempt + 1}/{max_retries})")
                logger.info("=" * 70)
                logger.info(f"Question: {response_json.get('question', 'N/A')}")
                logger.info(f"Returned Policy: {response_json.get('policy', 'N/A')}")
                logger.info(f"Returned Difficulty: {response_json.get('difficulty', 'N/A')}")
                logger.info(f"Returned Topic: {response_json.get('topic', 'N/A')}")
                logger.info(f"Returned Source: {response_json.get('source', 'N/A')}")
                logger.info(f"Returned Required Concepts: {response_json.get('required_concepts', [])}")
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
                if returned_difficulty != effective_difficulty:
                    logger.error(f"[QUESTION_GENERATION][REJECTED]")
                    logger.error(f"Reason: Difficulty mismatch")
                    logger.error(f"Expected Difficulty: {effective_difficulty}")
                    logger.error(f"Returned Difficulty: {returned_difficulty}")
                    raise ValueError(
                        f"Difficulty validation failed: expected '{effective_difficulty}', "
                        f"but LLM returned '{returned_difficulty}'. "
                        f"The LLM must use the exact difficulty supplied by the policy application."
                    )

                # Validate Switch Topic: returned topic must differ from current topic
                if policy == "Switch Topic":
                    returned_topic = response_json.get("topic")
                    if returned_topic == topic:
                        logger.error(f"[QUESTION_GENERATION][REJECTED]")
                        logger.error(f"Reason: Switch Topic did not change topic")
                        logger.error(f"Current Topic: {topic}")
                        logger.error(f"Returned Topic: {returned_topic}")
                        raise ValueError(
                            f"Switch Topic validation failed: returned topic '{returned_topic}' "
                            f"is the same as current topic '{topic}'. "
                            f"The LLM must select a different topic for Switch Topic policy."
                        )

                # Validate Probe Missing Concept: question should reference a missing concept
                if policy == "Probe Missing Concept" and missing_concept_list:
                    returned_question = response_json.get("question", "").lower()
                    has_missing_concept = any(
                        concept.lower() in returned_question
                        for concept in missing_concept_list
                    )
                    if not has_missing_concept:
                        logger.warning(f"[QUESTION_GENERATION][WARNING]")
                        logger.warning(f"Reason: Probe Missing Concept may not target missing concept")
                        logger.warning(f"Missing Concepts: {missing_concept_list}")
                        logger.warning(f"Question: {returned_question}")
                    # Don't fail, just warn - the LLM might reference concepts differently

                # Validate required fields
                required_fields = ["question", "difficulty", "topic", "policy", "source"]
                missing_fields = [field for field in required_fields if field not in response_json]
                if missing_fields:
                    logger.error(f"[QUESTION_GENERATION][REJECTED]")
                    logger.error(f"Reason: Missing required fields: {missing_fields}")
                    logger.error(f"Response keys: {list(response_json.keys())}")
                    if attempt < max_retries - 1:
                        logger.warning(f"[QUESTION_GENERATION][RETRY] Retrying due to missing fields")
                        continue
                    else:
                        raise ValueError(f"Missing required fields in LLM response after {max_retries} attempts: {missing_fields}")

                # Validate question is not identical to previous question
                if response_json.get('question') == previous_question:
                    logger.warning(f"[QUESTION_GENERATION][RETRY]")
                    logger.warning(f"Reason: Question identical to previous question")
                    logger.warning(f"Previous Question: {previous_question}")
                    logger.warning(f"Generated Question: {response_json.get('question')}")
                    logger.warning(f"Retry attempt: {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        # Continue to next retry with updated prompt instruction
                        continue
                    else:
                        logger.error(f"[QUESTION_GENERATION][REJECTED]")
                        logger.error(f"Reason: Question identical to previous question after {max_retries} attempts")
                        raise ValueError("Generated question is identical to previous question after multiple retries")

                # If we get here, validation passed - break out of retry loop
                break

            except ValueError as e:
                # Allow retries for: question identical, missing fields, non-dict response
                retryable_errors = ["identical to previous question", "Missing required fields", "not a dictionary"]
                if not any(error in str(e) for error in retryable_errors):
                    raise
                # If it's the last attempt, raise the error
                if attempt == max_retries - 1:
                    raise
                # Otherwise, continue to next retry
                continue

        # If we exited the loop without returning, raise an error
        if response_json is None:
            raise Exception("Failed to generate next question after multiple retries")

        # Log successful validation
        logger.info("")
        logger.info("=" * 70)
        logger.info("[QUESTION_GENERATION][VALIDATION]")
        logger.info("=" * 70)
        logger.info(f"Policy Match: True")
        logger.info(f"Difficulty Match: True")
        logger.info(f"Question Different: True")
        if policy == "Switch Topic":
            logger.info(f"Topic Changed: True")
        if policy == "Probe Missing Concept":
            logger.info(f"Targets Missing Concept: {has_missing_concept if missing_concept_list else 'N/A'}")
        logger.info("=" * 70)
        logger.info("")

        # Log successful question generation
        logger.info("")
        logger.info("=" * 60)
        logger.info("[NEXT QUESTION GENERATION]")
        logger.info("=" * 60)
        logger.info(f"Policy received from TabPFN: {policy}")
        logger.info(f"Effective difficulty: {effective_difficulty}")
        logger.info(f"Generated question: {response_json.get('question')}")
        logger.info(f"Generated difficulty: {response_json.get('difficulty')}")
        logger.info(f"Generated topic: {response_json.get('topic')}")
        logger.info(f"Returned policy: {response_json.get('policy')}")
        logger.info(f"Source: {response_json.get('source')}")
        logger.info(f"Required Concepts: {response_json.get('required_concepts', [])}")
        logger.info(f"Policy validation: PASSED")
        logger.info(f"Difficulty validation: PASSED")
        logger.info("=" * 60)
        logger.info("")

        return response_json
