"""
Test script for all seven TabPFN policies using local Qwen 4B for question generation.

Tests:
1. Increase Difficulty
2. Maintain Difficulty
3. Reduce Difficulty
4. Probe Missing Concept
5. Ask Application Question
6. Ask Follow-up Question
7. Switch Topic
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.llm import get_llm_client
from app.llm.question_generator import QuestionGenerator
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_all_policies():
    """Test all seven TabPFN policies with local Qwen 4B."""
    
    logger.info("=" * 60)
    logger.info("ALL SEVEN POLICIES TEST (LOCAL QWEN 4B)")
    logger.info("=" * 60)
    
    try:
        # Initialize question client with local Qwen
        logger.info("\n[TEST] Initializing question LLM client (local Qwen 4B)...")
        question_client = get_llm_client(task="question")
        logger.info(f"[TEST] Question client initialized")
        logger.info(f"[TEST] Model: {question_client.model_name}")
        
        # Initialize question generator
        question_generator = QuestionGenerator(question_client)
        
        # Test policies
        policies = [
            "Increase Difficulty",
            "Maintain Difficulty",
            "Reduce Difficulty",
            "Probe Missing Concept",
            "Ask Application Question",
            "Ask Follow-up Question",
            "Switch Topic"
        ]
        
        results = []
        
        for i, policy in enumerate(policies, 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"POLICY TEST {i}: {policy}")
            logger.info(f"{'=' * 60}")
            
            start_time = time.perf_counter()
            
            try:
                # Configure test parameters based on policy
                if policy == "Increase Difficulty":
                    current_difficulty = "Medium"
                    effective_difficulty = "Hard"
                    topic = "Python"
                    previous_question = "What is a list in Python?"
                    candidate_answer = "A list is a mutable sequence that can hold multiple items."
                    missing_concept_list = []
                    available_topics = ["Python", "Databases", "Machine Learning"]
                elif policy == "Maintain Difficulty":
                    current_difficulty = "Medium"
                    effective_difficulty = "Medium"
                    topic = "Python"
                    previous_question = "What is a list in Python?"
                    candidate_answer = "A list is a mutable sequence that can hold multiple items."
                    missing_concept_list = []
                    available_topics = ["Python", "Databases", "Machine Learning"]
                elif policy == "Reduce Difficulty":
                    current_difficulty = "Hard"
                    effective_difficulty = "Medium"
                    topic = "Python"
                    previous_question = "Explain the difference between shallow copy and deep copy in Python with memory implications."
                    candidate_answer = "I'm not sure about the memory implications."
                    missing_concept_list = ["memory", "references"]
                    available_topics = ["Python", "Databases", "Machine Learning"]
                elif policy == "Probe Missing Concept":
                    current_difficulty = "Medium"
                    effective_difficulty = "Medium"
                    topic = "Databases"
                    previous_question = "Explain the ACID properties in database transactions."
                    candidate_answer = "ACID includes Atomicity and Consistency. I'm not sure about the others."
                    missing_concept_list = ["Isolation", "Durability"]
                    available_topics = ["Python", "Databases", "Machine Learning"]
                elif policy == "Ask Application Question":
                    current_difficulty = "Medium"
                    effective_difficulty = "Medium"
                    topic = "Python"
                    previous_question = "What is a decorator in Python?"
                    candidate_answer = "A decorator is a function that modifies another function."
                    missing_concept_list = []
                    available_topics = ["Python", "Databases", "Machine Learning"]
                elif policy == "Ask Follow-up Question":
                    current_difficulty = "Medium"
                    effective_difficulty = "Medium"
                    topic = "Python"
                    previous_question = "What is a list in Python?"
                    candidate_answer = "A list is a mutable sequence that can hold multiple items of different types."
                    missing_concept_list = []
                    available_topics = ["Python", "Databases", "Machine Learning"]
                elif policy == "Switch Topic":
                    current_difficulty = "Medium"
                    effective_difficulty = "Medium"
                    topic = "Python"
                    previous_question = "What is a list in Python?"
                    candidate_answer = "A list is a mutable sequence."
                    missing_concept_list = []
                    available_topics = ["Databases", "Machine Learning", "Web Development"]
                
                result = question_generator.generate_next_question(
                    policy=policy,
                    topic=topic,
                    current_difficulty=current_difficulty,
                    effective_difficulty=effective_difficulty,
                    previous_question=previous_question,
                    candidate_answer=candidate_answer,
                    correctness_score=75,
                    concept_coverage=70,
                    reasoning_score=70,
                    missing_concepts=len(missing_concept_list),
                    missing_concept_list=missing_concept_list,
                    correct_streak=0,
                    wrong_streak=0,
                    available_topics=available_topics
                )
                
                elapsed = time.perf_counter() - start_time
                
                logger.info(f"\n[RESULT] Latency: {elapsed:.2f}s")
                logger.info(f"[RESULT] Question: {result.get('question')}")
                logger.info(f"[RESULT] Returned Difficulty: {result.get('difficulty')}")
                logger.info(f"[RESULT] Returned Topic: {result.get('topic')}")
                logger.info(f"[RESULT] Returned Policy: {result.get('policy')}")
                logger.info(f"[RESULT] Source: {result.get('source')}")
                
                # Validate policy match
                policy_match = result.get('policy') == policy
                logger.info(f"[VALIDATION] Policy match: {policy_match}")
                
                # Validate difficulty match
                difficulty_match = result.get('difficulty') == effective_difficulty
                logger.info(f"[VALIDATION] Difficulty match: {difficulty_match}")
                
                # Validate topic change for Switch Topic
                if policy == "Switch Topic":
                    topic_changed = result.get('topic') != topic
                    logger.info(f"[VALIDATION] Topic changed: {topic_changed}")
                else:
                    topic_changed = True  # Not applicable
                
                results.append({
                    'policy': policy,
                    'latency': elapsed,
                    'question': result.get('question'),
                    'difficulty': result.get('difficulty'),
                    'topic': result.get('topic'),
                    'returned_policy': result.get('policy'),
                    'source': result.get('source'),
                    'policy_match': policy_match,
                    'difficulty_match': difficulty_match,
                    'topic_changed': topic_changed if policy == "Switch Topic" else True,
                    'success': policy_match and difficulty_match and (topic_changed if policy == "Switch Topic" else True)
                })
                
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                logger.error(f"[ERROR] Policy test failed: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
                results.append({
                    'policy': policy,
                    'latency': elapsed,
                    'success': False,
                    'error': str(e)
                })
        
        # Summary
        logger.info(f"\n{'=' * 60}")
        logger.info("POLICY TEST SUMMARY")
        logger.info(f"{'=' * 60}")
        
        for result in results:
            if result['success']:
                logger.info(f"\n{result['policy']}:")
                logger.info(f"  Latency: {result['latency']:.2f}s")
                logger.info(f"  Policy match: {result['policy_match']}")
                logger.info(f"  Difficulty match: {result['difficulty_match']}")
                if result['policy'] == "Switch Topic":
                    logger.info(f"  Topic changed: {result['topic_changed']}")
                logger.info(f"  Status: PASS")
            else:
                logger.info(f"\n{result['policy']}:")
                logger.info(f"  Status: FAIL")
                logger.info(f"  Error: {result.get('error', 'Validation failed')}")
        
        pass_count = sum(1 for r in results if r['success'])
        logger.info(f"\nTotal: {pass_count}/{len(results)} policies passed")
        
        return results
        
    except Exception as e:
        logger.error(f"[TEST] Fatal error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    test_all_policies()
